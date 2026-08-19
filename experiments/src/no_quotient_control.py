"""Ablation NQ-1: no-gauge-quotient negative control.

Augments the physical family with explicit gauge columns (conformal phi*eta and
potential 2 d^s V) WITHOUT quotienting, and verifies that the unquotiented
Jacobian and Fisher matrix stay singular at every ray budget. The singularity
is expected -- this is an implementation negative control demonstrating that
the quotient is operationally necessary, not notation.

The analysis separates three magnitude scales and refuses to conflate them:
    exact conformal zeros            (pointwise null contraction, ~1e-18);
    quadrature-level gauge residuals (potential columns at order 1024, ~1e-12);
    genuinely visible physical singular values (~1e-2).
Quadrature noise is never counted as recovered gauge information: the rank
threshold is pinned ABOVE the measured gauge floor and BELOW the smallest
physical singular value, and both are reported.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from light_ray import (
    candidate_rays, conformal_contraction, gauge_contraction,
    make_physical_modes, mode_matrix, ray_integral_contraction,
)

SEED = 20260818
BUDGETS = [16, 32, 64, 128, 288]


def run(output: Path, order: int = 1024) -> dict:
    rays = candidate_rays(direction_count=72, offsets_per_direction=4)
    modes = make_physical_modes(12)
    j_phys = mode_matrix(rays, modes, order=order)

    rng = np.random.default_rng(SEED)
    gauge_cols, gauge_labels = [], []
    for i in range(2):  # conformal phi*eta columns
        c = rng.uniform(-0.5, 0.5, 4); r = rng.uniform(0.5, 0.9, 4)
        gauge_cols.append([ray_integral_contraction(
            ray, lambda p, k: conformal_contraction(p, k, c, r), order=order) for ray in rays])
        gauge_labels.append(f"conformal_{i}")
    for i in range(2):  # potential 2 d^s V columns
        c = rng.uniform(-0.5, 0.5, 4); r = rng.uniform(0.5, 0.9, 4); a = rng.normal(size=4)
        gauge_cols.append([ray_integral_contraction(
            ray, lambda p, k: gauge_contraction(p, k, c, r, a), order=order) for ray in rays])
        gauge_labels.append(f"potential_{i}")
    j_gauge = np.asarray(gauge_cols, dtype=float).T
    j_unq = np.column_stack((j_phys, j_gauge))
    dim_unq = j_unq.shape[1]

    conformal_floor = float(np.max(np.abs(j_gauge[:, :2])))
    potential_floor = float(np.max(np.abs(j_gauge[:, 2:])))
    gauge_floor = max(conformal_floor, potential_floor)

    per_budget = []
    all_singular = True
    sigma_min_phys_full = None
    for m in BUDGETS:
        j_m = j_unq[:m]
        widths = np.exp(np.random.default_rng(SEED + m).uniform(
            np.log(0.05), np.log(0.20), size=m))
        q = 1.0 / np.square(widths)
        b = np.sqrt(q)[:, None] * j_m
        singular = np.linalg.svd(b, compute_uv=False)
        sigma_min_phys = float(np.linalg.svd(b[:, :12], compute_uv=False).min())
        sigma_min_phys_full = sigma_min_phys
        # Threshold pinned between the gauge floor and the physical spectrum.
        threshold = float(np.sqrt(gauge_floor * sigma_min_phys))
        rank = int(np.count_nonzero(singular > threshold))
        f = b.T @ b
        eig = np.linalg.eigvalsh(0.5 * (f + f.T))
        f_rank = int(np.count_nonzero(eig > threshold ** 2))
        per_budget.append({
            "rays": m, "rank_J": rank, "rank_F": f_rank,
            "nullity": dim_unq - rank, "threshold": threshold,
            "sigma_min_physical": sigma_min_phys,
            "smallest_4_singular_values": singular[-4:].tolist(),
        })
        all_singular &= (rank == 12 and f_rank == 12 and dim_unq - rank == 4)

    checks = {
        "unquotiented_rank_deficit": {
            "pass": bool(all_singular),
            "statement": "rank(J_unquotiented) = 12 < 16 = dim at EVERY ray budget; "
                         "F = J^T W J singular regardless of budget; nullity = #gauge columns = 4",
        },
        "scale_separation": {
            "pass": bool(conformal_floor < 1e-14 and potential_floor < 1e-9
                         and sigma_min_phys_full > 1e3 * potential_floor),
            "exact_conformal_zero_floor": conformal_floor,
            "quadrature_potential_gauge_floor": potential_floor,
            "smallest_physical_singular_value": sigma_min_phys_full,
            "statement": "conformal columns are exact zeros; potential columns are "
                         "quadrature-limited; neither is counted as signal",
        },
    }
    all_pass = all(c["pass"] for c in checks.values())
    report = {
        "experiment": "no_gauge_quotient_negative_control",
        "conventions": {"quadrature_order": order, "seed": SEED,
                        "gauge_columns": gauge_labels,
                        "rank_threshold": "sqrt(gauge_floor * sigma_min_physical), pinned "
                                          "above quadrature noise and below physics"},
        "dimensions": {"physical": 12, "gauge_columns": 4, "unquotiented": dim_unq},
        "per_budget": per_budget,
        "checks": checks,
        "all_pass": bool(all_pass),
        "paper_claim": "Without the gauge quotient the Fisher matrix is singular at every "
                       "ray budget; the quotient is operationally necessary, not notation.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all_pass:
        raise SystemExit(f"STOP: no-quotient control failed. See {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/no_quotient_control.json"))
    parser.add_argument("--order", type=int, default=1024)
    args = parser.parse_args()
    r = run(args.output, order=args.order)
    print(json.dumps({"all_pass": r["all_pass"],
                      "nullity_by_budget": {str(b["rays"]): b["nullity"] for b in r["per_budget"]},
                      "floors": {k: r["checks"]["scale_separation"][k] for k in
                                 ("exact_conformal_zero_floor", "quadrature_potential_gauge_floor",
                                  "smallest_physical_singular_value")}}, indent=2))


if __name__ == "__main__":
    main()
