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
                for seed in SEED_SET:
                    rng = np.random.default_rng(seed)
                    widths = np.exp(rng.uniform(np.log(0.05), np.log(0.20), size=len(rays)))
                    q = 1.0 / np.square(widths)
                    start = time.perf_counter()
                    idx = greedy_d_design(j, q, k)
                    lat.append(time.perf_counter() - start)
                    vals = np.linalg.eigvalsh(fisher(j, q, idx))
                    lam.append(float(vals[0]))
                    full.append(bool(vals[0] > 1e-10 * max(float(vals[-1]), 1.0)))
                cell = {
                    "M": len(rays), "d": d, "K": k, "K_over_d": ratio,
                    "greedy_D": {
                        "median_latency_ms": float(np.median(lat) * 1000),
                        "max_latency_ms": float(np.max(lat) * 1000),
                        "median_lambda_min": float(np.median(lam)),
                        "full_rank_rate": float(np.mean(full)),
                        "seeds": SEED_SET,
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

    worst = max((c for c in cells if "greedy_D" in c),
                key=lambda c: c["greedy_D"]["max_latency_ms"])
    report = {
        "registered_seed_set": SEED_SET,
        "surface": {"M": list(POOLS), "d": DIMS, "K_over_d": BUDGET_RATIOS},
        "cells": cells,
        "worst_case_greedy": {
            "M": worst["M"], "d": worst["d"], "K": worst["K"],
            "max_latency_ms": worst["greedy_D"]["max_latency_ms"],
        },
        "decision_question": "is deterministic selection expensive enough to justify amortization?",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["worst_case_greedy"], indent=2))


if __name__ == "__main__":
    main()
