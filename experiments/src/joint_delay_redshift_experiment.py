"""Experiment D+R-1: clock-assisted conformal rank restoration in a joint
stationary tensor family.

The central physical distinction under test:

    Additional samples of a channel cannot reveal a direction annihilated by
    that channel; an additional observable can.

Family: h = sum_i theta_i e_i + sum_j alpha_j phi_j eta, with d_p stationary
physical modes e_i (all with h_0i = 0, inside the static ansatz) and the four
localized conformal modes of the section-6 experiment. The joint forward
operator has block form

    J_joint = [[D_p, 0], [R_p, R_c]],

so if D_p has full column rank, rank(J_joint) = d_p + rank(R_c) -- an identity
that holds even when the physical modes have nonzero redshift responses R_p.

Arms (per d_p in {6, 8, 12, 16}):
    A: K selected delay rays only               -> rank d_p
    B: same rays + 3 greedy clock links         -> rank d_p + 3
    C: same rays + 4 greedy clock links         -> rank d_p + 4
    D: same rays + 4 MORE delay rays            -> rank d_p   (negative control)
    E: same rays + all 10 clock links           -> rank d_p + 4
    F: arm E + constant endpoint conformal mode -> that mode stays invisible

Ten locked gates; any failure writes the full report and terminates.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from design_experiment import greedy_d_design
from light_ray import (
    DEFAULT_ORDER,
    ETA,
    StaticTensorMode,
    candidate_rays,
    gauss_legendre_interval,
    static_tensor_delay_matrix,
    static_tensor_redshift_matrix,
)
from static_redshift_experiment import (
    make_endpoint_fixed_gauge_modes,
    make_static_problem,
    numerical_rank,
)

D_P_VALUES = (6, 8, 12, 16)
GAUGE_ORDERS = (1024, 2048, 4096)
GAUGE_SELECTED_ORDER = 2048


class ConstantConformalMode:
    """h = c * eta with constant scalar profile.

    Not compactly supported, but the delay contraction vanishes pointwise
    (eta(k,k)=0), and every endpoint difference is exactly zero. This is the
    globally constant conformal factor that no clock network can see.
    """

    label = "phi_const"

    def __init__(self, amplitude: float = 1.0) -> None:
        self.amplitude = float(amplitude)

    def phi(self, spatial_points: np.ndarray) -> np.ndarray:
        return np.full(np.asarray(spatial_points).shape[0], self.amplitude)

    def tensor(self, spatial_points: np.ndarray) -> np.ndarray:
        n = np.asarray(spatial_points).shape[0]
        return self.amplitude * np.repeat(ETA[None, :, :], n, axis=0)

    def contraction(self, spacetime_points: np.ndarray, k: np.ndarray) -> np.ndarray:
        h = self.tensor(np.asarray(spacetime_points)[:, 1:])
        return np.einsum("nij,i,j->n", h, k, k)


def make_static_physical_modes(count: int = 16) -> list[StaticTensorMode]:
    """Stationary physical modes with h_0i = 0, staying inside the static
    ansatz so the endpoint redshift row is valid for them too.

    Four centers (the section-6 clock sites, where the conformal modes and
    clocks live) each carry a scalar mode (nonzero h_00, hence nonzero R_p --
    the rank law must survive R_p != 0) and three spatial trace-free modes
    (h_00 = 0).
    """
    centers = np.array([
        [-0.60, -0.55, -0.45],
        [-0.60,  0.55,  0.45],
        [ 0.60, -0.55,  0.45],
        [ 0.60,  0.55, -0.45],
    ], dtype=float)
    radii = np.array([0.55, 0.55, 0.55])
    p_scalar = np.diag([-2.0, -2.0, -2.0, -2.0])
    p1 = np.zeros((4, 4)); p1[1, 1] = 1.0; p1[2, 2] = -1.0
    p2 = np.zeros((4, 4)); p2[1, 2] = p2[2, 1] = 1.0
    p3 = np.zeros((4, 4)); p3[1, 1] = 1.0; p3[3, 3] = -1.0
    modes: list[StaticTensorMode] = []
    for j, center in enumerate(centers):
        for name, pol in (("scalar", p_scalar), ("xx-yy", p1), ("xy", p2), ("xx-zz", p3)):
            modes.append(StaticTensorMode(center, radii, pol, f"e_{j}_{name}"))
    if count > len(modes):
        raise ValueError(f"at most {len(modes)} static physical modes available")
    return modes[:count]


def whitened_metrics(j: np.ndarray, r_diag: np.ndarray) -> dict:
    jw = j / np.sqrt(r_diag)[None, :]
    f = jw.T @ jw
    vals = np.linalg.eigvalsh(0.5 * (f + f.T))
    return {"lambda_min_whitened": float(vals[0]), "lambda_max_whitened": float(vals[-1]),
            "whitened_eigenvalues": vals.tolist()}


def whitened_posterior(j: np.ndarray, r_diag: np.ndarray, d_p: int) -> dict:
    """Posterior blocks in full-bank-whitened coordinates.

    Invariant to diagonal rescaling of the mode basis, unlike the raw-coordinate
    block; the report carries both conventions explicitly.
    """
    jw = j / np.sqrt(r_diag)[None, :]
    d = j.shape[1]
    cov = np.linalg.inv(jw.T @ jw + np.eye(d))
    n_conf = d - d_p
    return {
        "posterior_rmse_physical": float(np.sqrt(np.trace(cov[:d_p, :d_p]) / d_p)),
        "posterior_rmse_conformal": (
            float(np.sqrt(np.trace(cov[d_p:, d_p:]) / n_conf)) if n_conf else None),
        "worst_direction_posterior_std": float(np.sqrt(np.linalg.eigvalsh(cov)[-1])),
    }


def clock_precision_sweep(j_delay: np.ndarray, j_clock: np.ndarray,
                          r_diag: np.ndarray, d_p: int,
                          rho_grid: np.ndarray) -> dict:
    """Conformal posterior std versus relative clock precision rho = sigma_D/sigma_R.

    Computed in full-bank-whitened coordinates:
        C_rho = [I + Jd~^T Jd~ + rho^2 Jr~^T Jr~]^{-1},
        u_conf(rho) = sqrt(tr(P_conf C_rho P_conf^T)/4).
    Rank restoration is exact at every rho; this curve is the separate
    precision statement.
    """
    scale = np.sqrt(r_diag)[None, :]
    jd = j_delay / scale
    jr = j_clock / scale
    d = jd.shape[1]
    base = np.eye(d) + jd.T @ jd
    ftr = jr.T @ jr
    n_conf = d - d_p
    curve = []
    for rho in rho_grid:
        cov = np.linalg.inv(base + float(rho) ** 2 * ftr)
        curve.append(float(np.sqrt(np.trace(cov[d_p:, d_p:]) / n_conf)))
    return {"rho": rho_grid.tolist(), "u_conf": curve,
            "marked_points_rho": [1.0, 10.0, 50.0],
            "marked_points_note": "rho = 1/s_clock at unit delay noise; s_clock in {1.0, 0.1, 0.02}"}


def posterior_stats(j: np.ndarray, d_p: int, rng: np.random.Generator,
                    trials: int = 400) -> dict:
    """Closed-form posterior blocks plus Monte-Carlo coverage.

    Unit-noise channels and standard normal prior (declared, not hidden): the
    experiment isolates rank geometry, not packet weighting.
    """
    d = j.shape[1]
    cov = np.linalg.inv(j.T @ j + np.eye(d))
    gain = cov @ j.T
    rmse_phys = float(np.sqrt(np.trace(cov[:d_p, :d_p]) / d_p))
    n_conf = d - d_p
    rmse_conf = float(np.sqrt(np.trace(cov[d_p:, d_p:]) / n_conf)) if n_conf else None
    worst = float(np.sqrt(np.linalg.eigvalsh(cov)[-1]))
    half = 1.959963984540054 * np.sqrt(np.diag(cov))
    covered = 0
    for _ in range(trials):
        theta = rng.normal(size=d)
        y = j @ theta + rng.normal(size=j.shape[0])
        estimate = gain @ y
        covered += int(np.count_nonzero(np.abs(estimate - theta) <= half))
    return {
        "posterior_rmse_physical": rmse_phys,
        "posterior_rmse_conformal": rmse_conf,
        "worst_direction_posterior_std": worst,
        "coverage_95": covered / (trials * d),
        "conformal_posterior_std_mean": (
            float(np.mean(np.sqrt(np.diag(cov)[d_p:]))) if n_conf else None),
    }


def run(output: Path, *, order: int = DEFAULT_ORDER, svd_tol: float = 1e-10,
        seed: int = 2026) -> dict:
    observers, links, conf_modes = make_static_problem()
    phys_modes = make_static_physical_modes(16)
    const_mode = ConstantConformalMode()
    rays = candidate_rays(direction_count=48, offsets_per_direction=3)
    rng = np.random.default_rng(seed)

    # Full candidate bank at the certified order: all 144 rays and 10 links,
    # all 16 physical + 4 conformal + constant columns. Sliced per d_p below.
    t0 = time.perf_counter()
    delay_phys_all = static_tensor_delay_matrix(rays, phys_modes, order=order)
    delay_conf_all = static_tensor_delay_matrix(rays, conf_modes, order=order)
    delay_const_all = static_tensor_delay_matrix(rays, [const_mode], order=order)
    build_seconds = time.perf_counter() - t0
    red_phys_all = static_tensor_redshift_matrix(links, phys_modes)
    red_conf_all = static_tensor_redshift_matrix(links, conf_modes)
    red_const_all = static_tensor_redshift_matrix(links, [const_mode])

    # Gate 2: conformal delay kernel over the whole pool.
    conf_delay_max = float(np.max(np.abs(np.column_stack((delay_conf_all, delay_const_all)))))
    conf_delay_rank, _, _ = numerical_rank(delay_conf_all, rtol=0.0, atol=1e-14)
    gate2 = conf_delay_max < 1e-14 and conf_delay_rank == 0

    # Gate 8: combined gauge invariance, machine-readable convergence table.
    gauge_modes = make_endpoint_fixed_gauge_modes(observers)
    gauge_delay_by_order = {}
    for g_order in GAUGE_ORDERS:
        worst = 0.0
        for ray in rays:
            lam, w = gauss_legendre_interval(ray.lam_min, ray.lam_max, order=g_order)
            pts = ray.points(lam)
            for mode in gauge_modes:
                h = mode.tensor(pts[:, 1:])
                val = 0.5 * float(w @ np.einsum("nij,i,j->n", h, ray.k, ray.k))
                worst = max(worst, abs(val))
        gauge_delay_by_order[g_order] = worst
    gauge_redshift_max = float(np.max(np.abs(
        static_tensor_redshift_matrix(links, gauge_modes))))
    gate8 = (gauge_delay_by_order[GAUGE_SELECTED_ORDER] < 1e-10
             and gauge_redshift_max < 1e-14)
    gauge_block = {
        "orders": list(GAUGE_ORDERS),
        "delay_max_abs": [gauge_delay_by_order[o] for o in GAUGE_ORDERS],
        "redshift_max_abs": gauge_redshift_max,
        "selected_order": GAUGE_SELECTED_ORDER,
        "threshold": 1e-10,
        "pass": bool(gate8),
    }

    per_dp = {}
    gate_flags = {n: True for n in
                  ("gate_1_physical_delay_rank", "gate_3_rank_law",
                   "gate_4_four_link_restoration", "gate_5_three_link_bound",
                   "gate_6_more_delay_negative_control", "gate_7_constant_mode_invisible",
                   "gate_9_no_physical_degradation")}

    for d_p in D_P_VALUES:
        dp_slice = delay_phys_all[:, :d_p]
        rp_slice = red_phys_all[:, :d_p]
        k_rays = d_p + 4

        # Ray selection by greedy-D on the physical delay block (unit weights).
        t0 = time.perf_counter()
        ray_sel = greedy_d_design(dp_slice, np.ones(len(rays)), k_rays, ridge=1e-12)
        ray_time = time.perf_counter() - t0
        # Arm D extras: a highest-sensitivity-norm heuristic (not a conditional
        # D/E-optimal continuation -- irrelevant to the conclusion, since every
        # delay row has identically zero conformal columns; the full-144-ray
        # bank check below closes the question completely).
        remaining = np.setdiff1d(np.arange(len(rays)), ray_sel)
        extra_scores = np.linalg.norm(dp_slice[remaining], axis=1)
        ray_extra = remaining[np.argsort(extra_scores)[-4:]]

        # Link selection by greedy-D on the conformal endpoint block.
        t0 = time.perf_counter()
        links4 = greedy_d_design(red_conf_all, np.ones(len(links)), 4, ridge=1e-12)
        links3 = greedy_d_design(red_conf_all, np.ones(len(links)), 3, ridge=1e-12)
        link_time = time.perf_counter() - t0

        def joint(ray_idx, link_idx, with_const=False):
            top = np.column_stack((
                dp_slice[ray_idx], delay_conf_all[ray_idx]))
            if with_const:
                top = np.column_stack((top, delay_const_all[ray_idx]))
            if len(link_idx):
                bottom = np.column_stack((
                    rp_slice[link_idx], red_conf_all[link_idx]))
                if with_const:
                    bottom = np.column_stack((bottom, red_const_all[link_idx]))
                return np.vstack((top, bottom))
            return top

        arms = {
            "A_delay_only": (joint(ray_sel, []), d_p),
            "B_plus_3_links": (joint(ray_sel, links3), d_p + 3),
            "C_plus_4_links": (joint(ray_sel, links4), d_p + 4),
            "D_plus_4_more_rays": (joint(np.concatenate((ray_sel, ray_extra)), []), d_p),
            "E_plus_all_links": (joint(ray_sel, np.arange(len(links))), d_p + 4),
        }

        # Fixed parameter metric from the FULL candidate bank, not the design.
        bank = np.vstack((
            np.column_stack((dp_slice, delay_conf_all)),
            np.column_stack((rp_slice, red_conf_all)),
        ))
        r_param = np.square(np.linalg.norm(bank, axis=0))
        r_param = np.maximum(r_param, 1e-30)

        arm_reports = {}
        for name, (matrix, expected) in arms.items():
            rank, singular, threshold = numerical_rank(matrix, rtol=svd_tol, atol=1e-14)
            report = {
                "rows": matrix.shape[0], "columns": matrix.shape[1],
                "expected_rank": expected, "observed_rank": rank,
                "rank_threshold": threshold,
                "nullity": matrix.shape[1] - rank,
                **whitened_metrics(matrix, r_param),
                "posterior_raw_mode_coordinates": posterior_stats(matrix, d_p, rng),
                "posterior_full_bank_whitened_coordinates": whitened_posterior(
                    matrix, r_param, d_p),
            }
            arm_reports[name] = report

        # Clock-precision curve: rank restoration is exact at every rho, but
        # how much conformal *uncertainty* the clocks remove is a precision
        # statement. A log-spaced sweep in whitened coordinates (not a single
        # favorable point); the canonical unit-noise value rho=1 stays visible
        # and the earlier declared points s_clock in {1.0, 0.1, 0.02} are marked.
        n_delay_rows = len(ray_sel)
        rho_grid = np.geomspace(0.1, 100.0, 25)
        for arm_name in ("C_plus_4_links", "E_plus_all_links"):
            matrix = arms[arm_name][0]
            arm_reports[arm_name]["clock_precision_sweep"] = clock_precision_sweep(
                matrix[:n_delay_rows], matrix[n_delay_rows:], r_param, d_p, rho_grid)

        # Gate 1: physical delay rank on pool and selected rays.
        pool_rank, _, _ = numerical_rank(dp_slice, rtol=svd_tol, atol=1e-14)
        sel_rank, sel_singular, _ = numerical_rank(dp_slice[ray_sel], rtol=svd_tol, atol=1e-14)
        gate_flags["gate_1_physical_delay_rank"] &= (pool_rank == d_p and sel_rank == d_p)

        # Gate 3: exact rank law on arms with clock rows.
        for arm_name, link_idx in (("B_plus_3_links", links3),
                                   ("C_plus_4_links", links4),
                                   ("E_plus_all_links", np.arange(len(links)))):
            rc_rank, _, _ = numerical_rank(red_conf_all[link_idx], rtol=svd_tol, atol=1e-14)
            law = d_p + rc_rank
            arm_reports[arm_name]["rank_law_prediction"] = law
            gate_flags["gate_3_rank_law"] &= (arm_reports[arm_name]["observed_rank"] == law)

        gate_flags["gate_4_four_link_restoration"] &= (
            arm_reports["C_plus_4_links"]["observed_rank"] == d_p + 4)
        gate_flags["gate_5_three_link_bound"] &= (
            arm_reports["B_plus_3_links"]["observed_rank"] == d_p + 3)
        full_bank_delay = np.column_stack((dp_slice, delay_conf_all))
        full_rank_bank, _, _ = numerical_rank(full_bank_delay, rtol=svd_tol, atol=1e-14)
        arm_reports["D_plus_4_more_rays"]["full_bank_144_ray_rank"] = full_rank_bank
        arm_reports["D_plus_4_more_rays"]["full_bank_144_ray_nullity"] = (
            full_bank_delay.shape[1] - full_rank_bank)
        gate_flags["gate_6_more_delay_negative_control"] &= (
            arm_reports["D_plus_4_more_rays"]["observed_rank"] == d_p
            and arm_reports["D_plus_4_more_rays"]["nullity"] == 4
            and full_rank_bank == d_p)

        # Arm F / gate 7: constant endpoint mode stays in the joint kernel.
        matrix_f = joint(ray_sel, np.arange(len(links)), with_const=True)
        rank_f, singular_f, _ = numerical_rank(matrix_f, rtol=svd_tol, atol=1e-14)
        _, _, vt = np.linalg.svd(matrix_f)
        kernel_vec = vt[-1]
        const_weight = float(abs(kernel_vec[-1]))
        arm_reports["F_plus_constant_mode"] = {
            "rows": matrix_f.shape[0], "columns": matrix_f.shape[1],
            "expected_rank": d_p + 4, "observed_rank": rank_f,
            "kernel_weight_on_constant_mode": const_weight,
        }
        gate_flags["gate_7_constant_mode_invisible"] &= (
            rank_f == d_p + 4 and const_weight > 0.99)

        # Gate 9: appending clock rows cannot shrink the physical block spectrum.
        phys_cols_joint = np.vstack((dp_slice[ray_sel], rp_slice))
        s_delay = np.linalg.svd(dp_slice[ray_sel], compute_uv=False)
        s_joint = np.linalg.svd(phys_cols_joint, compute_uv=False)
        degradation = float(np.max(s_delay - s_joint))
        gate_flags["gate_9_no_physical_degradation"] &= (degradation <= 1e-12)

        per_dp[str(d_p)] = {
            "selected_rays": ray_sel.tolist(),
            "extra_rays": ray_extra.tolist(),
            "links_4": [links[i].label for i in links4],
            "links_3": [links[i].label for i in links3],
            "ray_selection_seconds": ray_time,
            "link_selection_seconds": link_time,
            "physical_pool_rank": pool_rank,
            "selected_physical_singular_values": sel_singular.tolist(),
            "physical_degradation_max": degradation,
            "arms": arm_reports,
        }

    checks = {
        "gate_2_conformal_delay_kernel": {
            "pass": bool(gate2), "max_abs": conf_delay_max, "rank": conf_delay_rank,
            "threshold": 1e-14,
        },
        "gate_8_combined_static_gauge_invariance": gauge_block,
        **{name: {"pass": bool(flag)} for name, flag in gate_flags.items()},
        "gate_10_stop_on_failure": {
            "pass": True,
            "kind": "process control, not a measurement",
            "statement": "Any failed check raises SystemExit after the full report is written; "
                         "tolerances are not modified at runtime.",
        },
    }
    all_pass = all(c["pass"] for c in checks.values())
    report = {
        "experiment": "joint_delay_redshift_rank_restoration",
        "principle": ("additional samples of a channel cannot reveal a direction "
                      "annihilated by that channel; an additional observable can"),
        "conventions": {
            "metric": "diag(-1,1,1,1)",
            "delay": "A_gamma h = 0.5 integral h(k,k) d lambda",
            "redshift": "R_AB h = 0.5 [h_00(B)-h_00(A)] (static ansatz, all modes have h_0i=0)",
            "channel_weights": "unit noise on every channel; standard normal prior (declared)",
            "parameter_metric": "R = diag(||column||^2) of the FULL candidate bank per d_p",
            "quadrature_order": order,
            "gauge_stress_orders": list(GAUGE_ORDERS),
            "svd_relative_tolerance": svd_tol,
            "zero_matrix_absolute_tolerance": 1e-14,
            "seed": seed,
        },
        "dimensions": {
            "physical_dimensions": list(D_P_VALUES),
            "conformal_modes": len(conf_modes),
            "candidate_rays": len(rays),
            "candidate_links": len(links),
            "jacobian_build_seconds": build_seconds,
        },
        "per_dp": per_dp,
        "checks": checks,
        "all_pass": bool(all_pass),
        "paper_claim": (
            "Across stationary physical families of dimensions 6, 8, 12 and 16, the "
            "delay-only Jacobian had rank d_p and retained a four-dimensional conformal "
            "kernel. Three independent static-clock comparisons increased the joint rank "
            "to d_p+3, while four increased it to d_p+4. Adding four further delay "
            "observations did not change the conformal nullity. Thus the observed rank "
            "obeyed rank(J_joint) = d_p + rank(R_c) in every tested family."
        ),
        "statistical_qualification": (
            "Under the canonical unit-noise, standard-normal raw-coordinate prior, the "
            "added clock rows produced limited posterior contraction because the "
            "synthetic endpoint responses were small relative to the assumed noise. A "
            "declared clock-precision sweep showed increasing conformal contraction "
            "without affecting the exact rank result."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all_pass:
        failed = [k for k, v in checks.items() if not v["pass"]]
        raise SystemExit(f"STOP: joint delay-redshift gate(s) failed: {failed}. See {output}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("results/joint_delay_redshift.json"))
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER)
    parser.add_argument("--svd-tol", type=float, default=1e-10)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = run(args.output, order=args.order, svd_tol=args.svd_tol, seed=args.seed)
    summary = {
        "all_pass": report["all_pass"],
        "report": str(args.output),
        "ranks": {dp: {arm: r.get("observed_rank")
                       for arm, r in block["arms"].items()}
                  for dp, block in report["per_dp"].items()},
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
