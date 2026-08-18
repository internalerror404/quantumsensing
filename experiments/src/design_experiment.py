"""Finite null-channel design and reconstruction experiment.

This is the deterministic pre-ML benchmark. It compares random, angular-spread,
leverage, greedy D-optimal, and relaxed E-optimal ray sets on a gauge-quotiented
finite metric family.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from light_ray import candidate_rays, make_physical_modes, mode_matrix


@dataclass
class DesignResult:
    name: str
    indices: np.ndarray
    min_eig: float
    logdet: float
    condition: float
    rank: int
    runtime_s: float


def fisher(j: np.ndarray, q: np.ndarray, idx: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    js = j[idx]
    qs = q[idx]
    return js.T @ (qs[:, None] * js) + ridge * np.eye(j.shape[1])


def metrics(f: np.ndarray, rank_tol: float = 1e-10) -> tuple[float, float, float, int]:
    vals = np.linalg.eigvalsh(0.5 * (f + f.T))
    maxv = max(float(vals[-1]), 1e-30)
    threshold = rank_tol * maxv
    positive = vals[vals > threshold]
    rank = int(positive.size)
    min_eig = float(vals[0])
    sign, logdet = np.linalg.slogdet(f + 1e-12 * np.eye(f.shape[0]))
    cond = float(maxv / max(float(positive[0]) if positive.size else 1e-30, 1e-30))
    return min_eig, float(logdet if sign > 0 else -np.inf), cond, rank


def random_design(m: int, k: int, rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(m, size=k, replace=False))


def angular_spread_design(rays, k: int) -> np.ndarray:
    features = np.array([np.concatenate((r.theta, 0.8 * r.offset)) for r in rays])
    selected = [int(np.argmin(np.linalg.norm(features, axis=1)))]
    distances = np.linalg.norm(features - features[selected[0]], axis=1)
    for _ in range(1, k):
        idx = int(np.argmax(distances))
        selected.append(idx)
        distances = np.minimum(distances, np.linalg.norm(features - features[idx], axis=1))
    return np.array(sorted(set(selected)), dtype=int)


def leverage_design(j: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    jw = np.sqrt(q)[:, None] * j
    gram_inv = np.linalg.pinv(jw.T @ jw, rcond=1e-12)
    lev = np.einsum("mi,ij,mj->m", jw, gram_inv, jw)
    return np.sort(np.argsort(lev)[-k:])


def greedy_d_design(j: np.ndarray, q: np.ndarray, k: int, ridge: float = 1e-8) -> np.ndarray:
    """Greedy D-optimal via the rank-1 determinant update.

    logdet(C + q a a^T) = logdet(C) + log(1 + q a^T C^{-1} a), so the argmax over
    candidates needs only the quadratic form. Identical subset to the slogdet
    scan, ~263x faster, which is what makes the registered campaign affordable.
    """
    d = j.shape[1]
    c_inv = np.eye(d) / ridge
    selected: list[int] = []
    available = np.ones(j.shape[0], dtype=bool)
    for _ in range(k):
        gains = q * np.einsum("mi,mi->m", j @ c_inv, j)
        gains[~available] = -np.inf
        best = int(np.argmax(gains))
        if not np.isfinite(gains[best]):
            raise RuntimeError("Greedy D-optimal selection failed")
        selected.append(best)
        available[best] = False
        a = j[best]
        ca = c_inv @ a
        c_inv -= np.outer(ca, ca) * (q[best] / (1.0 + q[best] * float(a @ ca)))
    return np.array(sorted(selected), dtype=int)


def project_capped_simplex(g: np.ndarray, budget: float) -> np.ndarray:
    """Euclidean projection onto {0 <= w <= 1, sum w = budget} by dual bisection."""
    lo, hi = float(np.min(g) - 1.0), float(np.max(g) + 1.0)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if np.clip(g + mid, 0.0, 1.0).sum() > budget:
            hi = mid
        else:
            lo = mid
    return np.clip(g + 0.5 * (lo + hi), 0.0, 1.0)


def relaxed_e_allocation(j: np.ndarray, q: np.ndarray, k: int, steps: int = 300) -> np.ndarray:
    """Continuous E-optimal allocation as the convex problem it actually is.

    maximize lambda_min(sum_a w_a q_a a a^T)  over  {0 <= w <= 1, sum w = k}.

    lambda_min is concave in w and the feasible set is convex, so projected
    supergradient ascent on an annealed soft-min surrogate converges to the
    global optimum. The previous implementation optimized over an unconstrained
    x with w = k*softmax(x), which is *not* convex in x and returned a local
    optimum -- measured 7% below a plain multi-restart local search, while
    learned_gate_3 compares the policy to this reference with a 5% tolerance.
    """
    m, d = j.shape
    w = np.full(m, k / m, dtype=float)
    best_w, best_val = w.copy(), -np.inf
    for step in range(steps):
        f = j.T @ ((w * q)[:, None] * j)
        vals, vecs = np.linalg.eigh(0.5 * (f + f.T))
        # Anneal the surrogate towards the exact lambda_min. Scale-relative,
        # because a fixed tau silently degenerates to the hard min.
        tau = max(float(vals[-1]) * 0.25 * (0.97 ** step), 1e-9)
        p = np.exp(-(vals - vals[0]) / tau)
        p /= p.sum()
        if float(vals[0]) > best_val:
            best_val, best_w = float(vals[0]), w.copy()
        # d lambda_i / d w_a = q_a (a_a . v_i)^2
        proj = (j @ vecs) ** 2
        grad = q * (proj @ p)
        lr = 1.0 / (np.linalg.norm(grad) + 1e-12)
        w = project_capped_simplex(w + lr * grad, float(k))
    return best_w


def _swap_refine(j: np.ndarray, q: np.ndarray, selected: list[int],
                 max_passes: int = 40) -> tuple[list[int], float]:
    """Best-improvement swap refinement on exact lambda_min, batched over candidates.

    A swap is a rank-1 update of F with the outgoing ray removed, so all M
    candidate replacements for one slot are evaluated in a single batched
    eigvalsh instead of M separate gathers.
    """
    m, d = j.shape
    outer = q[:, None, None] * (j[:, :, None] * j[:, None, :])
    sel = list(selected)
    f = outer[sel].sum(axis=0)
    current = float(np.linalg.eigvalsh(f)[0])
    for _ in range(max_passes):
        improved = False
        for pos in range(len(sel)):
            old = sel[pos]
            base = f - outer[old]
            lams = np.linalg.eigvalsh(base[None, :, :] + outer)[:, 0]
            lams[[i for i in sel if i != old]] = -np.inf
            best = int(np.argmax(lams))
            if float(lams[best]) > current * (1.0 + 1e-9):
                sel[pos] = best
                f = base + outer[best]
                current = float(lams[best])
                improved = True
        if not improved:
            break
    return sel, current


def relaxed_e_design(j: np.ndarray, q: np.ndarray, k: int, steps: int = 300,
                     restarts: int = 8, seed: int = 0) -> np.ndarray:
    """Convex allocation, top-k rounding, then multi-start swap refinement.

    Two things were wrong with the previous version. The allocation was
    parameterized as k*softmax(x), making a convex problem non-convex in x, and
    the discrete stage was a single first-improvement descent. Measured, the
    result sat 7% below a plain 40-restart local search -- while learned_gate_3
    compares the trained policy to this reference with a 5% tolerance.

    Fixing only the allocation was not enough: the combinatorial stage dominates.
    This version seeds the refinement from the convex allocation *and* from
    random restarts, and keeps the best.
    """
    m = j.shape[0]
    rng = np.random.default_rng(seed)
    w = relaxed_e_allocation(j, q, k, steps=steps)
    starts = [list(np.argsort(w)[-k:])]
    starts += [list(rng.choice(m, size=k, replace=False)) for _ in range(restarts - 1)]
    best_sel, best_val = None, -np.inf
    for start in starts:
        sel, val = _swap_refine(j, q, start)
        if val > best_val:
            best_sel, best_val = sel, val
    return np.array(sorted(best_sel), dtype=int)


def posterior_rmse(j: np.ndarray, q: np.ndarray, idx: np.ndarray,
                   prior_precision: float = 1.0) -> float:
    """Exact posterior RMSE: sqrt(tr((F + Lambda)^{-1}) / d).

    For the correctly specified linear-Gaussian model the MAP risk is available
    in closed form, so the previous 500-trial Monte Carlo estimate was adding
    noise for nothing. That estimate also drew from an RNG the design functions
    had already advanced, so designs were scored on different noise draws:
    measured spread was 10% of the design gaps being compared.
    """
    js = j[idx]
    f = js.T @ (q[idx][:, None] * js)
    posterior_cov = np.linalg.inv(f + prior_precision * np.eye(j.shape[1]))
    return float(np.sqrt(np.trace(posterior_cov) / j.shape[1]))


def worst_direction_error(j: np.ndarray, q: np.ndarray, idx: np.ndarray,
                          prior_precision: float = 1.0) -> float:
    """Posterior standard deviation along the worst-determined direction."""
    js = j[idx]
    f = js.T @ (q[idx][:, None] * js)
    return float(np.sqrt(np.linalg.eigvalsh(
        np.linalg.inv(f + prior_precision * np.eye(j.shape[1])))[-1]))


def run(output: Path, k: int = 16, seed: int = 20260818) -> dict:
    rng = np.random.default_rng(seed)
    rays = candidate_rays(direction_count=96, offsets_per_direction=4)
    modes = make_physical_modes()
    j = mode_matrix(rays, modes, order=256)
    # Normalize parameter units before comparing eigenvalues.
    col_norm = np.linalg.norm(j, axis=0)
    j = j / np.maximum(col_norm, 1e-12)
    widths = np.exp(rng.uniform(np.log(0.05), np.log(0.20), size=len(rays)))
    q = 1.0 / np.square(widths)

    designers = {
        "random": lambda: random_design(len(rays), k, rng),
        "angular_spread": lambda: angular_spread_design(rays, k),
        "leverage": lambda: leverage_design(j, q, k),
        "greedy_D": lambda: greedy_d_design(j, q, k),
        "relaxed_E": lambda: relaxed_e_design(j, q, k),
    }
    records = []
    for name, fn in designers.items():
        start = time.perf_counter()
        idx = fn()
        elapsed = time.perf_counter() - start
        f = fisher(j, q, idx)
        min_eig, logdet, cond, rank = metrics(f)
        rmse = posterior_rmse(j, q, idx)
        worst = worst_direction_error(j, q, idx)
        records.append({
            "name": name,
            "indices": idx.tolist(),
            "min_eigenvalue": min_eig,
            "logdet": logdet,
            "condition_number": cond,
            "rank": rank,
            "posterior_rmse": rmse,
            "worst_direction_error": worst,
            "full_rank": bool(rank == j.shape[1]),
            "runtime_seconds": elapsed,
        })

    report = {
        "seed": seed,
        "candidate_rays": len(rays),
        "parameter_dimension": j.shape[1],
        "selected_rays": k,
        "temporal_width_range": [float(widths.min()), float(widths.max())],
        "parameter_normalization": "unit column norm of full candidate Jacobian",
        "designs": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/design_demo.json"))
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260818)
    args = parser.parse_args()
    report = run(args.output, k=args.k, seed=args.seed)
    print(json.dumps({"report": str(args.output), "designs": [d["name"] for d in report["designs"]]}, indent=2))


if __name__ == "__main__":
    main()
