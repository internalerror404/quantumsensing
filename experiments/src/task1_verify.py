"""Task 1 theorem-verification suite.

Stops on any deviation from the seven pinned gates. The suite does not silently
change tolerances or patch a failing identity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from light_ray import (
    DEFAULT_ORDER,
    ETA,
    Ray,
    candidate_rays,
    conformal_contraction,
    conformal_contraction_scale,
    endpoint_gauge_contraction,
    endpoint_gauge_value,
    gauge_contraction,
    gaussian_packet_frequency_variance,
    make_physical_modes,
    mode_matrix,
    ray_integral_contraction,
)


def numerical_rank(a: np.ndarray, rel_tol: float) -> tuple[int, np.ndarray, float]:
    s = np.linalg.svd(a, compute_uv=False)
    threshold = rel_tol * (s[0] if s.size else 1.0)
    return int(np.count_nonzero(s > threshold)), s, float(threshold)


def consistent_weighted_ranks(j: np.ndarray, widths: np.ndarray, rel_tol: float) -> tuple[dict, np.ndarray]:
    """Compare rank(J), rank(B), and rank(B^T B) with matched tolerances.

    Here B=W^{1/2}J and W is strictly positive diagonal, so rank(B)=rank(J)
    exactly. Numerically, J and B are each ranked by a relative SVD tolerance.
    If tau_B is the singular-value threshold for B, the corresponding
    eigenvalue threshold for F=B^T B is tau_B^2, subject to an explicit
    floating-point roundoff floor.
    """
    singular_j = np.linalg.svd(j, compute_uv=False)
    tau_j = rel_tol * (singular_j[0] if singular_j.size else 1.0)
    rank_j = int(np.count_nonzero(singular_j > tau_j))

    b = (1.0 / widths)[:, None] * j
    singular_b = np.linalg.svd(b, compute_uv=False)
    tau_b = rel_tol * (singular_b[0] if singular_b.size else 1.0)
    rank_b = int(np.count_nonzero(singular_b > tau_b))

    f = b.T @ b
    eig = np.linalg.eigvalsh(0.5 * (f + f.T))[::-1]
    lambda_max = max(float(eig[0]) if eig.size else 1.0, 1.0)
    roundoff_floor = 100.0 * np.finfo(float).eps * lambda_max * max(f.shape)
    f_threshold = max(tau_b * tau_b, roundoff_floor)
    rank_f = int(np.count_nonzero(eig > f_threshold))
    return {
        "rank_J": rank_j,
        "rank_weighted_J": rank_b,
        "rank_F": rank_f,
        "J_threshold": float(tau_j),
        "weighted_J_threshold": float(tau_b),
        "F_threshold": float(f_threshold),
        "F_threshold_components": {"matched_square": float(tau_b * tau_b), "roundoff_floor": float(roundoff_floor)},
        "J_singular_values": singular_j.tolist(),
        "weighted_J_singular_values": singular_b.tolist(),
        "F_eigenvalues": eig.tolist(),
    }, f


def qfim(j: np.ndarray, temporal_widths: np.ndarray) -> np.ndarray:
    w = 1.0 / np.square(temporal_widths)
    return j.T @ (w[:, None] * j)


def gauge_reference_block(rays, rng: np.random.Generator, n_potential: int = 60,
                          n_conformal: int = 20) -> np.ndarray:
    """Sampled numerical span of G = {2 d^s V} + {phi eta} on the given rays.

    Used by gate 8 to check that the physical mode family is independent
    *modulo gauge*, which is the claim that makes "d = 12" meaningful.
    """
    cols = []
    for _ in range(n_potential):
        c = rng.uniform(-0.6, 0.6, 4)
        rr = rng.uniform(0.4, 0.9, 4)
        a = rng.normal(size=4)
        cols.append([ray_integral_contraction(
            r, lambda p, k: gauge_contraction(p, k, c, rr, a)) for r in rays])
    for _ in range(n_conformal):
        c = rng.uniform(-0.6, 0.6, 4)
        rr = rng.uniform(0.4, 0.9, 4)
        cols.append([ray_integral_contraction(
            r, lambda p, k: conformal_contraction(p, k, c, rr)) for r in rays])
    return np.asarray(cols, dtype=float).T


def run(output: Path, svd_tol: float = 1e-10) -> dict:
    rng = np.random.default_rng(20260818)
    ray = Ray(theta=np.array([0.31, -0.52, 0.795]), offset=np.array([0.08, 0.12, 0.04]))
    center = np.array([0.05, -0.12, 0.08, 0.03])
    radii = np.array([0.85, 0.75, 0.75, 0.75])

    # Gate 1: conformal annihilation, as a *relative* residual of the assembled
    # tensor contraction. An absolute threshold would silently track |phi|.
    k = ray.k
    null_residual = abs(float(k @ ETA @ k))
    conf_value = abs(ray_integral_contraction(
        ray, lambda pts, kk: conformal_contraction(pts, kk, center, radii)))
    conf_scale = abs(ray_integral_contraction(
        ray, lambda pts, kk: conformal_contraction_scale(pts, kk, center, radii)))
    conf_relative = conf_value / max(conf_scale, 1e-300)
    gate1 = max(null_residual, conf_relative) < 1e-14

    # Gate 2: compact-support pure gauge cancellation on a single ray.
    covector = np.array([0.4, -0.7, 0.2, 0.55])
    gauge_value = abs(ray_integral_contraction(
        ray, lambda pts, kk: gauge_contraction(pts, kk, center, radii, covector)))
    gate2 = gauge_value < 1e-10

    # Convergence evidence for the gate-2/3 tolerance, so the threshold is
    # justified by observed behaviour rather than asserted.
    convergence = {}
    for order in (128, 256, 512, 1024):
        convergence[str(order)] = abs(ray_integral_contraction(
            ray, lambda pts, kk: gauge_contraction(pts, kk, center, radii, covector),
            order=order))

    # Gates 3 and 4: gauge columns stay null as rays are added; ranks agree.
    rays = candidate_rays(direction_count=72, offsets_per_direction=4)
    modes = make_physical_modes()
    j_phys = mode_matrix(rays, modes)
    conf_col = np.array([ray_integral_contraction(
        r, lambda pts, kk: conformal_contraction(pts, kk, center, radii)) for r in rays])
    # Gate 3 integrates the gauge direction on every ray in the pool, at the
    # same quadrature order used to build J. Substituting the analytic endpoint
    # formula here would make the gate vacuous: the bump is identically zero at
    # the ray endpoints, so the column would be exactly zero by construction.
    gauge_col = np.array([ray_integral_contraction(
        r, lambda pts, kk: gauge_contraction(pts, kk, center, radii, covector)) for r in rays])
    j_aug = np.column_stack((j_phys, conf_col, gauge_col))
    prefix_sizes = [8, 16, 32, 64, 128, len(rays)]
    widths_all = np.exp(rng.uniform(np.log(0.05), np.log(0.20), size=len(rays)))
    gauge_prefix_max = []
    rank_checks = []
    for m in prefix_sizes:
        j_m = j_aug[:m]
        rank_record, _ = consistent_weighted_ranks(j_m, widths_all[:m], svd_tol)
        gauge_prefix_max.append(float(np.max(np.abs(j_m[:, -2:]))))
        rank_record["m"] = m
        rank_checks.append(rank_record)
    gate3 = max(gauge_prefix_max) < 1e-10
    gate4 = all(item["rank_J"] == item["rank_weighted_J"] == item["rank_F"]
                for item in rank_checks)

    # Gate 5: the s^{-2} law, derived from the packet rather than assumed.
    # Var(nu-hat) is computed by quadrature from psi_s and must equal 1/(4 s^2);
    # the QFI is then 4 Var (A u)(A v), and only then is the slope fitted.
    s_grid = np.geomspace(0.04, 0.40, 17)
    variances = np.array([gaussian_packet_frequency_variance(float(sv)) for sv in s_grid])
    variance_rel_err = float(np.max(np.abs(variances * 4.0 * s_grid ** 2 - 1.0)))
    a_scalar = float(j_phys[0, 0])
    fisher_curve = 4.0 * variances * a_scalar ** 2
    slope, _ = np.polyfit(np.log(s_grid), np.log(fisher_curve), deg=1)
    gate5 = abs(float(slope) + 2.0) <= 0.01 and variance_rel_err <= 1e-12

    # Gate 6: endpoint term with a non-polynomial V, so quadrature error is real.
    finite_ray = Ray(theta=np.array([0.22, 0.61, -0.761]), offset=np.zeros(3),
                     lam_min=-1.0, lam_max=1.0)
    endpoint_numeric = ray_integral_contraction(
        finite_ray, lambda pts, kk: endpoint_gauge_contraction(pts, kk, covector))
    endpoint_exact = (endpoint_gauge_value(1.0, finite_ray.k, covector)
                      - endpoint_gauge_value(-1.0, finite_ray.k, covector))
    endpoint_rel = abs(endpoint_numeric - endpoint_exact) / max(abs(endpoint_exact), 1e-15)
    gate6 = endpoint_rel <= 1e-8

    # Gate 8: the physical family is independent modulo gauge. Without this the
    # dimension d = 12 is asserted rather than tested, and the no-gauge-quotient
    # negative control has nothing to be a control against.
    gauge_block = gauge_reference_block(rays, np.random.default_rng(7))
    gauge_floor = float(np.max(np.abs(gauge_block)))
    sigma_phys = np.linalg.svd(j_phys, compute_uv=False)
    quotient_margin = float(sigma_phys.min() / max(gauge_floor, 1e-300))
    gate8 = quotient_margin > 1e3

    results = {
        "conventions": {
            "A_gamma": "0.5 * integral h(k,k) dlambda",
            "metric": "diag(-1,1,1,1)",
            "svd_relative_tolerance": svd_tol,
            "quadrature_order": DEFAULT_ORDER,
            "k_normalization": "k=(1,theta), |theta|=1; A_gamma is not affine-reparametrization invariant",
        },
        "gate_1_conformal": {
            "pass": gate1, "null_residual": null_residual, "A_abs": conf_value,
            "A_relative": conf_relative, "integrand_scale": conf_scale,
            "threshold": 1e-14, "threshold_kind": "relative",
        },
        "gate_2_compact_gauge": {
            "pass": gate2, "A_abs": gauge_value, "threshold": 1e-10,
            "order_convergence": convergence,
        },
        "gate_3_gauge_persistence": {
            "pass": gate3, "prefix_max_abs": gauge_prefix_max, "threshold": 1e-10,
            "rays_integrated": len(rays), "order": DEFAULT_ORDER,
            "method": "gauge direction integrated on every ray; no analytic substitution",
        },
        "gate_4_rank_identity": {"pass": gate4, "checks": rank_checks},
        "gate_5_packet_scaling": {
            "pass": gate5, "loglog_slope": float(slope), "target": -2.0, "tolerance": 0.01,
            "variance_from_quadrature_rel_err": variance_rel_err,
            "method": "Var(nu) computed from psi_s by quadrature, checked against 1/(4 s^2)",
        },
        "gate_6_endpoint_term": {
            "pass": gate6, "numeric": float(endpoint_numeric), "exact": float(endpoint_exact),
            "relative_error": float(endpoint_rel), "threshold": 1e-8,
            "profile": "non-polynomial: 0.35 exp(0.4 sin 1.7t) + 0.18 cos 0.9t",
        },
        "gate_7_no_patch_policy": {
            "pass": True,
            "statement": "Any failed gate raises SystemExit; tolerances are not modified at runtime.",
        },
        "gate_8_gauge_quotient": {
            "pass": gate8, "sigma_min_physical": float(sigma_phys.min()),
            "gauge_floor": gauge_floor, "margin": quotient_margin, "threshold": 1e3,
            "statement": "physical mode family is independent modulo G = {2 d^s V} + {phi eta}",
        },
    }
    results["all_pass"] = all(
        results[name]["pass"]
        for name in (
            "gate_1_conformal",
            "gate_2_compact_gauge",
            "gate_3_gauge_persistence",
            "gate_4_rank_identity",
            "gate_5_packet_scaling",
            "gate_6_endpoint_term",
            "gate_7_no_patch_policy",
            "gate_8_gauge_quotient",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not results["all_pass"]:
        failed = [k for k, v in results.items() if isinstance(v, dict) and v.get("pass") is False]
        raise SystemExit(f"STOP: pinned theorem gate(s) failed: {failed}. See {output}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/task1_verification.json"))
    parser.add_argument("--svd-tol", type=float, default=1e-10)
    args = parser.parse_args()
    results = run(args.output, args.svd_tol)
    print(json.dumps({"all_pass": results["all_pass"], "report": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
