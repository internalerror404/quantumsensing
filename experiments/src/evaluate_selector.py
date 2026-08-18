"""Held-out evaluation of the learned selector against the deterministic baselines.

Without this, the selector's own report shows `rank: 12` and a positive
lambda_min for a design that was measured at 10x worse than random selection.
Computes learned_gate_2, _3 and _4 on a held-out task set with paired bootstrap
intervals. Gates 1 and 5 need the ablation sweep and are not computed here.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from design_experiment import (angular_spread_design, fisher, greedy_d_design,
                               leverage_design, random_design, relaxed_e_design)
from light_ray import candidate_rays, make_physical_modes, mode_matrix
from train_selector import DeepSetSelector, generate_tasks, select


def lam_min(j: np.ndarray, q: np.ndarray, idx) -> float:
    return float(np.linalg.eigvalsh(fisher(j, q, np.asarray(idx)))[0])


def bootstrap_ratio_ci(policy: np.ndarray, baseline: np.ndarray, rng: np.random.Generator,
                       draws: int = 10000) -> tuple[float, float, float]:
    """Paired bootstrap over tasks on the log ratio; lambda_min is multiplicative."""
    log_ratio = np.log(policy) - np.log(baseline)
    n = log_ratio.size
    medians = np.median(log_ratio[rng.integers(0, n, size=(draws, n))], axis=1)
    return (float(np.exp(np.median(log_ratio))),
            float(np.exp(np.quantile(medians, 0.025))),
            float(np.exp(np.quantile(medians, 0.975))))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("results/selector_checkpoint.pt"))
    parser.add_argument("--output", type=Path, default=Path("results/selector_evaluation.json"))
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026, help="held-out seed; distinct from training")
    args = parser.parse_args()

    ck = torch.load(args.checkpoint, weights_only=True)
    model = DeepSetSelector(ck["input_dim"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    rng = np.random.default_rng(args.seed)
    rays = candidate_rays(direction_count=96, offsets_per_direction=4)
    base_j = mode_matrix(rays, make_physical_modes())
    base_j = base_j / np.maximum(np.linalg.norm(base_j, axis=0), 1e-12)
    q_base = 1.0 / np.square(np.exp(rng.uniform(np.log(0.05), np.log(0.20), size=len(rays))))
    ray_features = np.array([np.concatenate((r.theta, r.offset)) for r in rays], dtype=np.float32)
    directions = np.array([r.theta for r in rays], dtype=float)

    names = ("learned", "random", "angular_spread", "leverage", "greedy_D", "relaxed_E")
    scores = {n: [] for n in names}
    latency = {n: 0.0 for n in names}
    full_rank = []

    def timed(name, fn):
        start = time.perf_counter()
        idx = fn()
        latency[name] += time.perf_counter() - start
        return idx

    for _ in range(args.tasks):
        js, qs, feats, avail, d, k = generate_tasks(
            base_j, ray_features, q_base, directions, 1, rng)
        j = js[0][:, :d].astype(float)
        q = np.where(avail[0], qs[0].astype(float), 1e-12)
        usable = np.flatnonzero(avail[0])
        if usable.size < k:
            continue

        start = time.perf_counter()
        logits = select(model, feats, avail, k)[0]
        chosen = np.argsort(logits)[-k:]
        latency["learned"] += time.perf_counter() - start

        oracle = timed("relaxed_E", lambda: relaxed_e_design(j, q, k, restarts=4))
        rnd = timed("random", lambda: usable[random_design(usable.size, k, rng)])
        ang = timed("angular_spread",
                    lambda: usable[angular_spread_design([rays[i] for i in usable], k)])
        lev = timed("leverage", lambda: leverage_design(j, q, k))
        grd = timed("greedy_D", lambda: greedy_d_design(j, q, k))

        scores["learned"].append(lam_min(j, q, chosen))
        scores["random"].append(lam_min(j, q, rnd))
        scores["angular_spread"].append(lam_min(j, q, ang))
        scores["leverage"].append(lam_min(j, q, lev))
        scores["greedy_D"].append(lam_min(j, q, grd))
        scores["relaxed_E"].append(lam_min(j, q, oracle))
        full_rank.append(np.linalg.matrix_rank(j[chosen], tol=1e-10 * np.abs(j).max()) == d)

    arr = {n: np.asarray(v) for n, v in scores.items()}
    n_tasks = arr["learned"].size
    boot = np.random.default_rng(args.seed + 1)

    gate2 = {}
    for baseline in ("random", "angular_spread"):
        ratio, lo, hi = bootstrap_ratio_ci(arr["learned"], arr[baseline], boot)
        gate2[baseline] = {"median_ratio": ratio, "ci95": [lo, hi],
                           "pass": bool(ratio >= 1.15 and lo > 1.0)}
    oracle_ratio, o_lo, o_hi = bootstrap_ratio_ci(arr["learned"], arr["relaxed_E"], boot)
    ms = {n: 1000.0 * latency[n] / max(n_tasks, 1) for n in names}
    speedup = ms["relaxed_E"] / max(ms["learned"], 1e-9)

    # ---- POST-HOC practical-utility analysis (not preregistered) ----
    # The preregistered gates stand exactly as computed above. This section asks
    # a question they do not: is the policy on the runtime-quality Pareto
    # frontier of the registered baselines at all? Gate 4 compares latency only
    # against the slow per-instance optimizer, so a policy can pass all its
    # gates while a *faster* deterministic heuristic also achieves a *better*
    # objective -- strict domination that the gates never look at.
    med = {n: float(np.median(v)) for n, v in arr.items()}
    dominators = []
    for b in names:
        if b == "learned":
            continue
        ratio, lo, hi = bootstrap_ratio_ci(arr[b], arr["learned"], boot)
        if med[b] >= med["learned"] and ms[b] <= ms["learned"] and lo > 1.0:
            dominators.append({"baseline": b, "quality_ratio": ratio,
                               "ci95": [lo, hi], "latency_ms": ms[b]})
    faster = [b for b in names if b != "learned" and ms[b] <= ms["learned"]]
    best_faster = max(faster, key=lambda b: med[b]) if faster else None
    quality_at_latency = (med["learned"] / med[best_faster]) if best_faster else float("inf")
    post_hoc = {
        "post_hoc": True,
        "note": "not preregistered; preregistered gate outcomes above are unchanged",
        "latency_ms_per_task": ms,
        "pareto_dominated_by": dominators,
        "pareto_dominated": bool(dominators),
        "quality_at_latency": {
            "best_baseline_not_slower_than_policy": best_faster,
            "policy_over_that_baseline": quality_at_latency,
            "proposed_v0_3_target": 0.95,
            "pass": bool(quality_at_latency >= 0.95),
        },
    }

    report = {
        "tasks": n_tasks,
        "heldout_seed": args.seed,
        "median_lambda_min": {n: float(np.median(v)) for n, v in arr.items()},
        "learned_gate_2_vs_baselines": gate2,
        "learned_gate_2_pass": all(g["pass"] for g in gate2.values()),
        "learned_gate_3_oracle_ratio": {
            "median_ratio": oracle_ratio, "ci95": [o_lo, o_hi], "target": 0.95,
            "pass": bool(oracle_ratio >= 0.95),
            "note": "spec permits reporting the gap instead of passing",
        },
        "learned_gate_4_speedup": {
            "policy_ms_per_task": ms["learned"],
            "oracle_ms_per_task": ms["relaxed_E"],
            "speedup": speedup, "target": 20.0, "pass": bool(speedup >= 20.0),
        },
        "heldout_full_rank_rate": float(np.mean(full_rank)) if full_rank else 0.0,
        "post_hoc_practical_utility": post_hoc,
        "policy_status": "frozen as an honest negative baseline for one-shot amortized selection",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
