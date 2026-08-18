"""Deterministic design scaling study over the registered surface.

Answers the decision question left open by the selector evaluation: at the
largest scientifically relevant instance, is deterministic Fisher-aware
selection actually expensive enough to justify amortization? If greedy-D
remains in the low-millisecond range at M=4096, a learned selector has no
operational role in Paper 1 and the ML line needs a stronger justification
than latency.

Sweeps the registered surface (configs/experiment.yaml):
    M    in {256, 512, 1024, 4096}   (pool size)
    d    in {6, 8, 12, 16}           (family dimension)
    K/d  in {1.0, 1.25, 1.5, 2.0}    (budget ratio)
using the registered seed_set for the packet widths -- the first use of the
registered seeds anywhere in this package. Greedy-D is measured on every cell
and every seed; relaxed-E, being the expensive reference rather than a
deployment candidate, is measured once per (M, d) at K/d = 1.5.
"""
from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from design_experiment import fisher, greedy_d_design, relaxed_e_design
from light_ray import candidate_rays, make_physical_modes, mode_matrix

SEED_SET = [2026, 3407, 9181, 17041, 27183]
POOLS = {256: (64, 4), 512: (128, 4), 1024: (256, 4), 4096: (512, 8)}
DIMS = [6, 8, 12, 16]
BUDGET_RATIOS = [1.0, 1.25, 1.5, 2.0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/scaling_study.json"))
    parser.add_argument("--order", type=int, default=512,
                        help="quadrature order for J assembly; design latency is unaffected")
    args = parser.parse_args()

    modes16 = make_physical_modes(16)
    cells = []
    for m_target, (n_dir, n_off) in POOLS.items():
        rays = candidate_rays(direction_count=n_dir, offsets_per_direction=n_off)
        t0 = time.perf_counter()
        j_full = mode_matrix(rays, modes16, order=args.order)
        build_s = time.perf_counter() - t0
        for d in DIMS:
            j = j_full[:, :d]
            j = j / np.maximum(np.linalg.norm(j, axis=0), 1e-12)
            for ratio in BUDGET_RATIOS:
                k = int(round(d * ratio))
                lat, lam, full = [], [], []
                lam_blk, full_blk, mem_kib = [], [], []
                directions = np.array([r.theta for r in rays])
                for seed in SEED_SET:
                    rng = np.random.default_rng(seed)
                    widths = np.exp(rng.uniform(np.log(0.05), np.log(0.20), size=len(rays)))
                    q = 1.0 / np.square(widths)
                    tracemalloc.start()
                    start = time.perf_counter()
                    idx = greedy_d_design(j, q, k)
                    lat.append(time.perf_counter() - start)
                    mem_kib.append(tracemalloc.get_traced_memory()[1] / 1024.0)
                    tracemalloc.stop()
                    vals = np.linalg.eigvalsh(fisher(j, q, idx))
                    lam.append(float(vals[0]))
                    full.append(bool(vals[0] > 1e-10 * max(float(vals[-1]), 1.0)))
                    # Registered worst-case blocking: a contiguous 20% angular
                    # sector around a random axis, pinned at the cap rather
                    # than drawn from U(0, 0.2), so this measures the hardest
                    # registered case instead of an easier random draw.
                    axis = rng.normal(size=3)
                    axis /= np.linalg.norm(axis)
                    proj = directions @ axis
                    blocked = proj >= np.quantile(proj, 0.8)
                    q_blk = np.where(blocked, 1e-12, q)
                    idx_b = greedy_d_design(j, q_blk, k)
                    vals_b = np.linalg.eigvalsh(fisher(j, q_blk, idx_b))
                    lam_blk.append(float(vals_b[0]))
                    full_blk.append(bool(vals_b[0] > 1e-10 * max(float(vals_b[-1]), 1.0)))
                cell = {
                    "M": len(rays), "d": d, "K": k, "K_over_d": ratio,
                    "greedy_D": {
                        "median_latency_ms": float(np.median(lat) * 1000),
                        "max_latency_ms": float(np.max(lat) * 1000),
                        "median_lambda_min": float(np.median(lam)),
                        "full_rank_rate": float(np.mean(full)),
                        "peak_memory_kib": float(np.max(mem_kib)),
                        "seeds": SEED_SET,
                    },
                    "greedy_D_blocked_20pct_sector": {
                        "median_lambda_min": float(np.median(lam_blk)),
                        "full_rank_rate": float(np.mean(full_blk)),
                    },
                }
                if ratio == 1.5:
                    rng = np.random.default_rng(SEED_SET[0])
                    widths = np.exp(rng.uniform(np.log(0.05), np.log(0.20), size=len(rays)))
                    q = 1.0 / np.square(widths)
                    restarts = 4 if len(rays) <= 1024 else 2
                    start = time.perf_counter()
                    idx = relaxed_e_design(j, q, k, restarts=restarts)
                    elapsed = time.perf_counter() - start
                    cell["relaxed_E"] = {
                        "latency_s": float(elapsed),
                        "lambda_min": float(np.linalg.eigvalsh(fisher(j, q, idx))[0]),
                        "restarts": restarts,
                        "seed": SEED_SET[0],
                    }
                cells.append(cell)
        cells.append({"M": len(rays), "jacobian_build_seconds": float(build_s),
                      "quadrature_order": args.order})

    # Order-stability check: the runtime sweep builds J at --order (512 by
    # default) for speed, but the certified operator uses order 1024 (the
    # compact-gauge residual at 512 is ~1e-10 -- at the pinned gate, not
    # comfortably under it). Verify on the boundary cells that the design
    # conclusions do not depend on the order: same selected subsets, or
    # negligible lambda_min shift, between 512 and 1024.
    stability = []
    for m_target, d, ratio in ((256, 6, 1.5), (1024, 12, 1.5), (4096, 16, 2.0)):
        n_dir, n_off = POOLS[m_target]
        rays = candidate_rays(direction_count=n_dir, offsets_per_direction=n_off)
        k = int(round(d * ratio))
        per_order = {}
        for order in (512, 1024):
            j = mode_matrix(rays, modes16, order=order)[:, :d]
            j = j / np.maximum(np.linalg.norm(j, axis=0), 1e-12)
            sel, lam = [], []
            for seed in SEED_SET:
                rng = np.random.default_rng(seed)
                widths = np.exp(rng.uniform(np.log(0.05), np.log(0.20), size=len(rays)))
                q = 1.0 / np.square(widths)
                idx = greedy_d_design(j, q, k)
                sel.append(tuple(idx.tolist()))
                lam.append(float(np.linalg.eigvalsh(fisher(j, q, idx))[0]))
            per_order[order] = (sel, lam)
        identical = [a == b for a, b in zip(per_order[512][0], per_order[1024][0])]
        rel = [abs(a - b) / max(abs(b), 1e-30)
               for a, b in zip(per_order[512][1], per_order[1024][1])]
        stability.append({
            "M": len(rays), "d": d, "K": k,
            "subsets_identical_per_seed": identical,
            "all_subsets_identical": bool(all(identical)),
            "max_rel_lambda_min_shift": float(max(rel)),
        })

    worst = max((c for c in cells if "greedy_D" in c),
                key=lambda c: c["greedy_D"]["max_latency_ms"])
    report = {
        "registered_seed_set": SEED_SET,
        "surface": {"M": list(POOLS), "d": DIMS, "K_over_d": BUDGET_RATIOS},
        "parameter_normalization": (
            "columns of J whitened to unit norm: R = diag(||J_:,1||^2, ..., ||J_:,d||^2), "
            "J_tilde = J R^{-1/2}. Reported eigenvalues are of the normalized visibility "
            "operator, not raw physical QFI; rank and blocked/unblocked ratios are "
            "normalization-independent, absolute cross-family lambda_min values are not."
        ),
        "quadrature_note": (
            "runtime sweep at --order (default 512) for J-build speed; design-conclusion "
            "stability vs the certified order 1024 is verified on boundary cells below. "
            "Final paper tables must use order 1024."
        ),
        "order_stability": stability,
        "cells": cells,
        "worst_case_greedy": {
            "M": worst["M"], "d": worst["d"], "K": worst["K"],
            "max_latency_ms": worst["greedy_D"]["max_latency_ms"],
        },
        "decision_question": "is deterministic selection expensive enough to justify amortization?",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"worst_case_greedy": report["worst_case_greedy"],
                      "order_stability": stability}, indent=2))


if __name__ == "__main__":
    main()
