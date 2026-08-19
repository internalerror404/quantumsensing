"""Registered deterministic full-metric campaign (order 1024).

The scaling study established feasibility (runtime, rank, memory, blocking);
this campaign prices the phenomenon: conditioning, posterior cost in both
declared coordinate conventions, blocking retention, and per-cell timing, over
the registered surface and seed set.

Execution rules (registered before the run):
- each candidate Jacobian is built ONCE per pool size M at order 1024, cached,
  and reused across all dimensions, budget ratios, and seeds -- the seed
  changes the packet widths q, not the geometry;
- greedy-D runs on every cell x seed; relaxed-E runs on the preregistered
  representative cells only, as a quality reference, not a deployment method;
- posterior RMSE and worst-direction error are analytic; Monte-Carlo coverage
  is a calibration check on the representative cells only;
- seed-wise records are stored, not only pooled medians.
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
# Representative cells for the relaxed-E reference and MC coverage (registered).
REPRESENTATIVE = ((256, 12, 1.5), (1024, 12, 1.5), (4096, 16, 1.5))
LOGDET_EPS = 1e-12


def rank_of(f: np.ndarray, rtol: float = 1e-10) -> int:
    vals = np.linalg.eigvalsh(0.5 * (f + f.T))
    lam_max = max(float(vals[-1]), 1e-30)
    threshold = max(rtol * rtol * lam_max, 100.0 * np.finfo(float).eps * lam_max * f.shape[0])
    return int(np.count_nonzero(vals > threshold))


def cell_metrics(j_w: np.ndarray, j_raw: np.ndarray, q: np.ndarray, idx: np.ndarray) -> dict:
    """All per-design metrics; posterior blocks in both declared conventions."""
    f_w = fisher(j_w, q, idx)
    vals = np.linalg.eigvalsh(0.5 * (f_w + f_w.T))
    lam_min, lam_max = float(vals[0]), float(vals[-1])
    sign, logdet = np.linalg.slogdet(f_w + LOGDET_EPS * np.eye(f_w.shape[0]))
    d = j_w.shape[1]
    cov_w = np.linalg.inv(f_w + np.eye(d))
    f_raw = fisher(j_raw, q, idx)
    cov_raw = np.linalg.inv(f_raw + np.eye(d))
    return {
        "rank": rank_of(f_w),
        "lambda_min_whitened": lam_min,
        "logdet_whitened": float(logdet if sign > 0 else -np.inf),
        "condition_number": float(lam_max / max(lam_min, 1e-300)),
        "posterior_whitened": {
            "rmse": float(np.sqrt(np.trace(cov_w) / d)),
            "worst_direction_std": float(np.sqrt(np.linalg.eigvalsh(cov_w)[-1])),
        },
        "posterior_raw": {
            "rmse": float(np.sqrt(np.trace(cov_raw) / d)),
            "worst_direction_std": float(np.sqrt(np.linalg.eigvalsh(cov_raw)[-1])),
        },
    }


def run(output: Path, order: int = 1024) -> dict:
    modes16 = make_physical_modes(16)
    campaign: dict = {
        "registered_seed_set": SEED_SET,
        "surface": {"M": list(POOLS), "d": DIMS, "K_over_d": BUDGET_RATIOS},
        "representative_cells": [list(c) for c in REPRESENTATIVE],
        "conventions": {
            "quadrature_order": order,
            "parameter_metric": "R = diag(||J_col||^2) of the full candidate pool per (M, d); "
                                "whitened metrics use J R^{-1/2}; raw metrics use J unchanged",
            "prior": "standard normal in each declared coordinate convention",
            "qfi_weights": "q = 1/s^2, packet widths log-uniform(0.05, 0.20) per registered seed",
            "blocking": "pinned worst-case 20% contiguous angular sector per seed",
            "logdet_ridge": LOGDET_EPS,
            "coverage": "MC calibration on representative cells only (300 trials, raw coords)",
        },
        "cells": [],
        "representative_relaxed_E": [],
        "representative_coverage": [],
    }

    for m_target, (n_dir, n_off) in POOLS.items():
        rays = candidate_rays(direction_count=n_dir, offsets_per_direction=n_off)
        directions = np.array([r.theta for r in rays])
        t0 = time.perf_counter()
        j_full = mode_matrix(rays, modes16, order=order)   # built ONCE per M
        build_s = time.perf_counter() - t0
        campaign["cells"].append({"M": len(rays), "jacobian_build_seconds": build_s,
                                  "quadrature_order": order})
        for d in DIMS:
            j_raw = j_full[:, :d]
            col = np.linalg.norm(j_raw, axis=0)
            j_w = j_raw / np.maximum(col, 1e-15)[None, :]
            for ratio in BUDGET_RATIOS:
                k = int(round(d * ratio))
                seed_records = []
                for seed in SEED_SET:
                    rng = np.random.default_rng(seed)
                    widths = np.exp(rng.uniform(np.log(0.05), np.log(0.20), size=len(rays)))
                    q = 1.0 / np.square(widths)
                    tracemalloc.start()
                    t0 = time.perf_counter()
                    idx = greedy_d_design(j_w, q, k)
                    select_s = time.perf_counter() - t0
                    peak_kib = tracemalloc.get_traced_memory()[1] / 1024.0
                    tracemalloc.stop()
                    record = {"seed": seed, "selection_seconds": select_s,
                              "peak_memory_kib": peak_kib, **cell_metrics(j_w, j_raw, q, idx)}
                    # Blocking robustness: pinned 20% sector, same seed stream.
                    axis = rng.normal(size=3)
                    axis /= np.linalg.norm(axis)
                    proj = directions @ axis
                    q_blk = np.where(proj >= np.quantile(proj, 0.8), 1e-12, q)
                    idx_b = greedy_d_design(j_w, q_blk, k)
                    f_b = fisher(j_w, q_blk, idx_b)
                    lam_b = float(np.linalg.eigvalsh(0.5 * (f_b + f_b.T))[0])
                    record["blocked"] = {
                        "lambda_min_whitened": lam_b,
                        "retention": lam_b / max(record["lambda_min_whitened"], 1e-300),
                        "full_rank": bool(rank_of(f_b) == d),
                    }
                    seed_records.append(record)
                campaign["cells"].append({"M": len(rays), "d": d, "K": k,
                                          "K_over_d": ratio, "seeds": seed_records})

    # Relaxed-E reference and MC coverage on the representative cells.
    for m_target, d, ratio in REPRESENTATIVE:
        n_dir, n_off = POOLS[m_target]
        rays = candidate_rays(direction_count=n_dir, offsets_per_direction=n_off)
        j_raw = mode_matrix(rays, modes16, order=order)[:, :d]
        j_w = j_raw / np.maximum(np.linalg.norm(j_raw, axis=0), 1e-15)[None, :]
        k = int(round(d * ratio))
        rng = np.random.default_rng(SEED_SET[0])
        widths = np.exp(rng.uniform(np.log(0.05), np.log(0.20), size=len(rays)))
        q = 1.0 / np.square(widths)
        restarts = 4 if len(rays) <= 1024 else 2
        t0 = time.perf_counter()
        idx_e = relaxed_e_design(j_w, q, k, restarts=restarts)
        e_seconds = time.perf_counter() - t0
        idx_g = greedy_d_design(j_w, q, k)
        lam = lambda i: float(np.linalg.eigvalsh(fisher(j_w, q, np.asarray(i)))[0])
        campaign["representative_relaxed_E"].append({
            "M": len(rays), "d": d, "K": k, "seed": SEED_SET[0],
            "relaxed_E_lambda_min": lam(idx_e), "relaxed_E_seconds": e_seconds,
            "greedy_D_lambda_min": lam(idx_g), "restarts": restarts,
        })
        # Coverage calibration (correctly specified model -> ~0.95 expected).
        cov_hits = 0
        trials = 300
        f_raw = fisher(j_raw, q, idx_g)
        cov_post = np.linalg.inv(f_raw + np.eye(d))
        gain = cov_post @ (j_raw[idx_g].T * q[idx_g][None, :])
        half = 1.959963984540054 * np.sqrt(np.diag(cov_post))
        rng_c = np.random.default_rng(SEED_SET[0] + 1)
        for _ in range(trials):
            theta = rng_c.normal(size=d)
            y = j_raw[idx_g] @ theta + rng_c.normal(size=k) / np.sqrt(q[idx_g])
            estimate = gain @ y
            cov_hits += int(np.count_nonzero(np.abs(estimate - theta) <= half))
        campaign["representative_coverage"].append({
            "M": len(rays), "d": d, "K": k, "trials": trials,
            "coverage_95": cov_hits / (trials * d),
        })

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(campaign, indent=2), encoding="utf-8")
    return campaign


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/full_metric_campaign.json"))
    parser.add_argument("--order", type=int, default=1024)
    args = parser.parse_args()
    campaign = run(args.output, order=args.order)
    data_cells = [c for c in campaign["cells"] if "seeds" in c]
    lam = [s["lambda_min_whitened"] for c in data_cells for s in c["seeds"]]
    fr = [s["rank"] == c["d"] for c in data_cells for s in c["seeds"]]
    ret = [s["blocked"]["retention"] for c in data_cells for s in c["seeds"]]
    print(json.dumps({
        "report": str(args.output),
        "cells": len(data_cells), "cell_x_seed": len(lam),
        "full_rank_rate": float(np.mean(fr)),
        "lambda_min_whitened_range": [float(np.min(lam)), float(np.max(lam))],
        "blocking_retention_median": float(np.median(ret)),
        "coverage": campaign["representative_coverage"],
    }, indent=2))


if __name__ == "__main__":
    main()
