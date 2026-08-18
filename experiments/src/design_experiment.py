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
from scipy.optimize import minimize

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
    d = j.shape[1]
    current = ridge * np.eye(d)
    selected: list[int] = []
    available = np.ones(j.shape[0], dtype=bool)
    for _ in range(k):
        best_idx = -1
        best_score = -np.inf
        for idx in np.flatnonzero(available):
            a = j[idx]
            candidate = current + q[idx] * np.outer(a, a)
            sign, score = np.linalg.slogdet(candidate)
            if sign > 0 and score > best_score:
                best_score = float(score)
                best_idx = int(idx)
        if best_idx < 0:
            raise RuntimeError("Greedy D-optimal selection failed")
        selected.append(best_idx)
        current += q[best_idx] * np.outer(j[best_idx], j[best_idx])
        available[best_idx] = False
    return np.array(sorted(selected), dtype=int)


def relaxed_e_design(j: np.ndarray, q: np.ndarray, k: int, steps: int = 400) -> np.ndarray:
    """Continuous E-optimal allocation, top-k rounding, then one-swap refinement."""
    m, d = j.shape
    x0 = np.zeros(m)

    def soft_weights(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        p = np.exp(z)
        p /= np.sum(p)
        return k * p

    def objective(x: np.ndarray) -> float:
        w = soft_weights(x)
        f = j.T @ ((w * q)[:, None] * j)
        vals = np.linalg.eigvalsh(f + 1e-9 * np.eye(d))
        tau = max(float(vals[-1]) * 0.02, 1e-8)
        # Smooth minimum eigenvalue.
        softmin = -tau * np.log(np.sum(np.exp(-vals / tau)))
        cap_penalty = 20.0 * np.sum(np.square(np.maximum(w - 1.0, 0.0)))
        return float(-softmin + cap_penalty)

    result = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": steps, "ftol": 1e-10})
    w = soft_weights(result.x)
    selected = list(np.argsort(w)[-k:])

    def score(indices: list[int]) -> float:
        vals = np.linalg.eigvalsh(fisher(j, q, np.array(indices), ridge=1e-10))
        return float(vals[0])

    improved = True
    while improved:
        improved = False
        current_score = score(selected)
        outside = [i for i in range(m) if i not in selected]
        for pos in range(k):
            old = selected[pos]
            for new in outside:
                trial = selected.copy(); trial[pos] = new
                s = score(trial)
                if s > current_score * (1.0 + 1e-9):
                    selected = trial
                    outside.remove(new); outside.append(old)
                    current_score = s
                    improved = True
                    break
            if improved:
                break
    return np.array(sorted(selected), dtype=int)


def posterior_rmse(j: np.ndarray, q: np.ndarray, idx: np.ndarray, trials: int,
                   rng: np.random.Generator, prior_precision: float = 1.0) -> float:
    js = j[idx]
    qs = q[idx]
    f = js.T @ (qs[:, None] * js)
    precision = f + prior_precision * np.eye(j.shape[1])
    gain = np.linalg.solve(precision, js.T * qs[None, :])
    squared = []
    noise_std = 1.0 / np.sqrt(qs)
    for _ in range(trials):
        theta = rng.normal(size=j.shape[1])
        y = js @ theta + rng.normal(scale=noise_std)
        estimate = gain @ y
        squared.append(float(np.mean((estimate - theta) ** 2)))
    return float(np.sqrt(np.mean(squared)))


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
        rmse = posterior_rmse(j, q, idx, trials=500, rng=rng)
        records.append({
            "name": name,
            "indices": idx.tolist(),
            "min_eigenvalue": min_eig,
            "logdet": logdet,
            "condition_number": cond,
            "rank": rank,
            "posterior_rmse": rmse,
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
