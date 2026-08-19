"""Ablation W-1: QFI vs uniform selection weights x homogeneous vs heterogeneous
packet widths, with cross-evaluation and controlled total information.

The quantum-statistical content of the design framework is the claim that the
packet-derived weight q_a = 1/s_a^2 is a *meaningful* statistical weight, not a
notational one. This ablation isolates it on the three registered
representative cells, under the constraint (1/M) sum_a q_a = 1 in every
condition, so no arm wins merely by holding more total information.

2x2 design per (cell, seed):
    packet resources: homogeneous s_a = s_0 (matched: 1/s_0^2 = mean(1/s_a^2))
                      vs heterogeneous s_a ~ log-uniform(0.05, 0.20);
    selection weighting: uniform (designer pretends q = 1)
                      vs QFI (designer sees the true q).
Under homogeneous resources the two weightings coincide (q is constant), so
that condition contributes one reference arm. Under heterogeneous resources
both designs are CROSS-EVALUATED under the true QFI-weighted objective --
scoring each design only on the objective it optimized would be circular.
No predetermined success criterion: whatever the comparison shows is the result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from design_experiment import fisher, greedy_d_design
from light_ray import candidate_rays, make_physical_modes, mode_matrix

SEED_SET = [2026, 3407, 9181, 17041, 27183]
POOLS = {256: (64, 4), 1024: (256, 4), 4096: (512, 8)}
CELLS = ((256, 12, 18), (1024, 12, 18), (4096, 16, 24))


def evaluate(j_w: np.ndarray, j_raw: np.ndarray, q_true: np.ndarray, idx) -> dict:
    idx = np.asarray(idx)
    f = fisher(j_w, q_true, idx)
    vals = np.linalg.eigvalsh(0.5 * (f + f.T))
    d = j_w.shape[1]
    cov_w = np.linalg.inv(f + np.eye(d))
    cov_raw = np.linalg.inv(fisher(j_raw, q_true, idx) + np.eye(d))
    return {
        "lambda_min_whitened": float(vals[0]),
        "logdet_whitened": float(np.linalg.slogdet(f + 1e-12 * np.eye(d))[1]),
        "posterior_rmse_whitened": float(np.sqrt(np.trace(cov_w) / d)),
        "posterior_rmse_raw": float(np.sqrt(np.trace(cov_raw) / d)),
        "worst_direction_std_whitened": float(np.sqrt(np.linalg.eigvalsh(cov_w)[-1])),
    }


def run(output: Path, order: int = 1024) -> dict:
    modes16 = make_physical_modes(16)
    cells_out = []
    for m_target, d, k in CELLS:
        n_dir, n_off = POOLS[m_target]
        rays = candidate_rays(direction_count=n_dir, offsets_per_direction=n_off)
        j_raw = mode_matrix(rays, modes16, order=order)[:, :d]
        j_w = j_raw / np.maximum(np.linalg.norm(j_raw, axis=0), 1e-15)[None, :]
        m = len(rays)
        seed_records = []
        for seed in SEED_SET:
            rng = np.random.default_rng(seed)
            widths = np.exp(rng.uniform(np.log(0.05), np.log(0.20), size=m))
            q_het = 1.0 / np.square(widths)
            q_het = q_het / q_het.mean()          # total-information control
            q_homo = np.ones(m)                    # matched: 1/s_0^2 = mean(q_het) = 1

            # Designs. Under homogeneous q the weighting distinction vanishes.
            sel_homo = greedy_d_design(j_w, q_homo, k)
            sel_het_uniform = greedy_d_design(j_w, q_homo, k)   # designer blind to q
            sel_het_qfi = greedy_d_design(j_w, q_het, k)        # designer sees q

            overlap = len(set(sel_het_uniform.tolist()) & set(sel_het_qfi.tolist())) / k
            record = {
                "seed": seed,
                "selected_set_overlap_uniform_vs_qfi": overlap,
                # Reference: homogeneous resources (evaluated under q_homo).
                "homogeneous_resources": evaluate(j_w, j_raw, q_homo, sel_homo),
                # Heterogeneous resources: both designs cross-evaluated under
                # the TRUE heterogeneous objective.
                "het_uniform_selected_eval_qfi": evaluate(j_w, j_raw, q_het, sel_het_uniform),
                "het_qfi_selected_eval_qfi": evaluate(j_w, j_raw, q_het, sel_het_qfi),
                # And the reverse cross-check under the uniform objective.
                "het_qfi_selected_eval_uniform": evaluate(j_w, j_raw, q_homo, sel_het_qfi),
                "het_uniform_selected_eval_uniform": evaluate(j_w, j_raw, q_homo, sel_het_uniform),
            }
            record["qfi_weighting_gain_lambda_min"] = (
                record["het_qfi_selected_eval_qfi"]["lambda_min_whitened"]
                / max(record["het_uniform_selected_eval_qfi"]["lambda_min_whitened"], 1e-300))
            record["heterogeneity_effect_lambda_min"] = (
                record["het_qfi_selected_eval_qfi"]["lambda_min_whitened"]
                / max(record["homogeneous_resources"]["lambda_min_whitened"], 1e-300))
            seed_records.append(record)
        cells_out.append({"M": m, "d": d, "K": k, "seeds": seed_records})

    report = {
        "experiment": "qfi_vs_uniform_weights_x_homogeneous_vs_heterogeneous_packets",
        "conventions": {
            "quadrature_order": order,
            "total_information_control": "(1/M) sum_a q_a = 1 in every condition",
            "homogeneous_matching": "1/s_0^2 = mean(1/s_a^2) over the heterogeneous draw",
            "designer": "greedy D-optimal (deployment method)",
            "cross_evaluation": "every design scored under the TRUE objective of its resource condition",
            "success_criterion": "none preregistered; the comparison itself is the result",
        },
        "registered_seed_set": SEED_SET,
        "cells": cells_out,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/weight_packet_ablation.json"))
    parser.add_argument("--order", type=int, default=1024)
    args = parser.parse_args()
    r = run(args.output, order=args.order)
    summary = []
    for c in r["cells"]:
        gains = [s["qfi_weighting_gain_lambda_min"] for s in c["seeds"]]
        het = [s["heterogeneity_effect_lambda_min"] for s in c["seeds"]]
        ov = [s["selected_set_overlap_uniform_vs_qfi"] for s in c["seeds"]]
        summary.append({"M": c["M"], "d": c["d"], "K": c["K"],
                        "median_qfi_weighting_gain": float(np.median(gains)),
                        "median_heterogeneity_effect": float(np.median(het)),
                        "median_set_overlap": float(np.median(ov))})
    print(json.dumps({"report": str(args.output), "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
