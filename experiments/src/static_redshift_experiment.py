"""Static-clock redshift and conformal-rank restoration experiment.

This implements the restricted static-observer channel used by the paper's
redshift corollary.  It deliberately keeps the interior null-delay transform
and the endpoint clock functional as distinct forward blocks:

    D_gamma(phi eta) = 0,
    R_AB(phi eta) = 1/2 [phi(A) - phi(B)].

The experiment verifies the formula, the delay-only conformal kernel, exact
rank restoration after adding clock links, invariance under endpoint-fixed
stationary gauge transformations, and a small clock-network design problem.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from design_experiment import fisher, greedy_d_design
from light_ray import (
    DEFAULT_ORDER,
    StaticClockLink,
    StaticConformalMode,
    StaticGaugeMode,
    StaticObserver,
    candidate_rays,
    static_conformal_delay_matrix,
    static_gauge_redshift_matrix,
    static_redshift_formula_matrix,
    static_redshift_matrix,
)


def numerical_rank(matrix: np.ndarray, *, rtol: float = 1e-10,
                   atol: float = 0.0) -> tuple[int, np.ndarray, float]:
    """SVD rank with an explicit absolute floor.

    The absolute floor is load-bearing for analytically zero delay matrices:
    scaling a relative threshold by roundoff would spuriously classify the
    roundoff itself as signal.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.size == 0:
        return 0, np.array([], dtype=float), float(atol)
    singular = np.linalg.svd(matrix, compute_uv=False)
    leading = float(singular[0]) if singular.size else 0.0
    threshold = max(float(atol), float(rtol) * leading)
    rank = int(np.count_nonzero(singular > threshold))
    return rank, singular, threshold


def make_static_problem() -> tuple[
        list[StaticObserver], list[StaticClockLink], list[StaticConformalMode]]:
    """Five clocks, ten candidate links, and four localized conformal modes."""
    centers = np.array([
        [-0.60, -0.55, -0.45],
        [-0.60,  0.55,  0.45],
        [ 0.60, -0.55,  0.45],
        [ 0.60,  0.55, -0.45],
    ], dtype=float)
    reference = StaticObserver(np.zeros(3), "O0")
    mode_observers = [
        StaticObserver(center, f"O{j + 1}")
        for j, center in enumerate(centers)
    ]
    observers = [reference, *mode_observers]
    links = [
        StaticClockLink(observers[i], observers[j])
        for i, j in itertools.combinations(range(len(observers)), 2)
    ]
    radii = np.array([0.38, 0.38, 0.38])
    modes = [
        StaticConformalMode(center, radii, f"phi_{j + 1}")
        for j, center in enumerate(centers)
    ]
    return observers, links, modes


def make_endpoint_fixed_gauge_modes(
        observers: list[StaticObserver]) -> list[StaticGaugeMode]:
    """Nonzero stationary gauge tensors whose vectors vanish at all clocks."""
    covectors = [
        np.array([0.0, 1.0, 0.35, -0.20]),
        np.array([0.0, -0.30, 1.0, 0.25]),
        np.array([0.0, 0.20, -0.40, 1.0]),
        np.array([0.0, 0.80, 0.10, -0.55]),
    ]
    modes: list[StaticGaugeMode] = []
    for j, observer in enumerate(observers[1:]):
        # Support is local to this clock.  The linear factor fixes the gauge
        # vector at the endpoint while leaving a nonzero derivative there.
        center = observer.position + np.array([0.12, 0.0, 0.0])
        modes.append(StaticGaugeMode(
            center=center,
            radii=np.array([0.30, 0.30, 0.30]),
            covector=covectors[j],
            label=f"V_{j + 1}",
            zero_point=observer.position,
            zero_axis=0,
        ))
    return modes


def endpoint_incidence(
        observers: list[StaticObserver],
        links: list[StaticClockLink]) -> np.ndarray:
    index = {observer.label: i for i, observer in enumerate(observers)}
    incidence = np.zeros((len(links), len(observers)), dtype=float)
    for ell, link in enumerate(links):
        incidence[ell, index[link.emitter.label]] = 1.0
        incidence[ell, index[link.receiver.label]] = -1.0
    return incidence


def run(output: Path, *, order: int = DEFAULT_ORDER,
        svd_tol: float = 1e-10) -> dict:
    observers, links, modes = make_static_problem()

    # Delay block: enough rays to exercise directions and transverse offsets,
    # but the conformal contraction must vanish pointwise for every one.
    rays = candidate_rays(direction_count=48, offsets_per_direction=3)
    delay = static_conformal_delay_matrix(rays, modes, order=order)
    delay_rank, delay_singular, delay_threshold = numerical_rank(
        delay, rtol=0.0, atol=1e-14
    )
    delay_max = float(np.max(np.abs(delay)))

    # Endpoint clock block through two independent code paths: full h_00 tensor
    # extraction and the closed endpoint-difference formula.
    redshift = static_redshift_matrix(links, modes)
    formula = static_redshift_formula_matrix(links, modes)
    formula_error_abs = float(np.max(np.abs(redshift - formula)))
    formula_error_rel = formula_error_abs / max(
        float(np.max(np.abs(formula))), 1e-300
    )

    # Independent finite-difference check of the static lapse formula.  For
    # g_00=-(1+epsilon phi), N=sqrt(1+epsilon phi) and
    # zeta_AB=log N(A)-log N(B).  A centered derivative must converge
    # quadratically to 1/2[phi(A)-phi(B)].
    epsilons = np.array([0.5, 0.25, 0.125, 0.0625], dtype=float)
    linearization_errors = []
    for epsilon in epsilons:
        numeric = np.empty_like(formula)
        for ell, link in enumerate(links):
            point_a = link.emitter.position[None, :]
            point_b = link.receiver.position[None, :]
            for j, mode in enumerate(modes):
                phi_a = float(mode.phi(point_a)[0])
                phi_b = float(mode.phi(point_b)[0])
                z_plus = 0.5 * (
                    np.log1p(epsilon * phi_a)
                    - np.log1p(epsilon * phi_b)
                )
                z_minus = 0.5 * (
                    np.log1p(-epsilon * phi_a)
                    - np.log1p(-epsilon * phi_b)
                )
                numeric[ell, j] = (z_plus - z_minus) / (2.0 * epsilon)
        linearization_errors.append(float(np.max(np.abs(numeric - formula))))
    linearization_errors_arr = np.asarray(linearization_errors)
    linearization_slope = float(np.polyfit(
        np.log(epsilons), np.log(linearization_errors_arr), 1
    )[0])

    redshift_rank, redshift_singular, redshift_threshold = numerical_rank(
        redshift, rtol=svd_tol, atol=1e-14
    )
    combined = np.vstack((delay, redshift))
    combined_rank, combined_singular, combined_threshold = numerical_rank(
        combined, rtol=svd_tol, atol=1e-14
    )

    # Graph form R = 1/2 B Phi, where B is the oriented link incidence matrix
    # and Phi stores mode values at clock nodes.  It also makes the globally
    # constant conformal endpoint mode visibly lie in the graph kernel.
    incidence = endpoint_incidence(observers, links)
    phi_nodes = np.column_stack([
        mode.phi(np.vstack([o.position for o in observers]))
        for mode in modes
    ])
    incidence_redshift = 0.5 * incidence @ phi_nodes
    incidence_error = float(np.max(np.abs(redshift - incidence_redshift)))
    constant_endpoint_response = 0.5 * incidence @ np.ones(len(observers))

    # Orientation convention: reversing a link must reverse the log-redshift.
    reversed_links = [
        StaticClockLink(link.receiver, link.emitter, f"rev({link.label})")
        for link in links
    ]
    reversal_error = float(np.max(np.abs(
        static_redshift_matrix(reversed_links, modes) + redshift
    )))

    # Stationary gauge test.  The gauge vector fixes every endpoint, and the
    # assembled tensor is intentionally nonzero there, so a zero redshift is
    # not produced by a zero perturbation.
    gauge_modes = make_endpoint_fixed_gauge_modes(observers)
    gauge_redshift = static_gauge_redshift_matrix(links, gauge_modes)
    endpoint_positions = np.vstack([observer.position for observer in observers])
    gauge_vector_max = 0.0
    gauge_tensor_norm_max = 0.0
    gauge_h00_max = 0.0
    gauge_h0i_max = 0.0
    for mode in gauge_modes:
        vector = mode.covector_field(endpoint_positions)
        tensor = mode.tensor(endpoint_positions)
        gauge_vector_max = max(gauge_vector_max, float(np.max(np.abs(vector))))
        gauge_tensor_norm_max = max(
            gauge_tensor_norm_max,
            float(np.max(np.linalg.norm(tensor.reshape(len(observers), -1), axis=1))),
        )
        gauge_h00_max = max(gauge_h00_max, float(np.max(np.abs(tensor[:, 0, 0]))))
        gauge_h0i_max = max(gauge_h0i_max, float(np.max(np.abs(tensor[:, 0, 1:]))))
    gauge_redshift_max = float(np.max(np.abs(gauge_redshift)))

    # Small finite clock-network design.  Four scalar endpoint channels are the
    # row-count lower bound for a four-dimensional conformal family.  Greedy-D
    # should select four independent links and restore all four directions.
    q_clock = np.ones(len(links), dtype=float)
    selected = greedy_d_design(redshift, q_clock, k=len(modes), ridge=1e-12)
    selected_redshift = redshift[selected]
    selected_rank, selected_singular, selected_threshold = numerical_rank(
        selected_redshift, rtol=svd_tol, atol=1e-14
    )
    selected_fisher = fisher(
        redshift, q_clock, selected, ridge=0.0
    )
    selected_fisher_eigs = np.linalg.eigvalsh(
        0.5 * (selected_fisher + selected_fisher.T)
    )

    selected_three = greedy_d_design(
        redshift, q_clock, k=len(modes) - 1, ridge=1e-12
    )
    three_rank, _, _ = numerical_rank(
        redshift[selected_three], rtol=svd_tol, atol=1e-14
    )

    checks = {
        "endpoint_formula": {
            "pass": formula_error_rel < 1e-13,
            "max_abs_error": formula_error_abs,
            "max_relative_error": formula_error_rel,
            "threshold": 1e-13,
        },
        "nonlinear_lapse_linearization": {
            "pass": abs(linearization_slope - 2.0) < 0.05
                    and linearization_errors[-1] < linearization_errors[0] / 50.0,
            "epsilons": epsilons.tolist(),
            "max_abs_errors": linearization_errors,
            "observed_loglog_slope": linearization_slope,
            "target_slope": 2.0,
            "slope_tolerance": 0.05,
        },
        "delay_only_conformal_rank_zero": {
            "pass": delay_max < 1e-14 and delay_rank == 0,
            "max_abs": delay_max,
            "rank": delay_rank,
            "singular_values": delay_singular.tolist(),
            "rank_threshold": delay_threshold,
            "rays": len(rays),
            "order": order,
        },
        "combined_rank_equals_endpoint_rank": {
            "pass": redshift_rank == len(modes) and combined_rank == redshift_rank,
            "redshift_rank": redshift_rank,
            "combined_rank": combined_rank,
            "redshift_singular_values": redshift_singular.tolist(),
            "combined_singular_values": combined_singular.tolist(),
            "redshift_rank_threshold": redshift_threshold,
            "combined_rank_threshold": combined_threshold,
        },
        "incidence_identity_and_constant_mode": {
            "pass": incidence_error < 1e-14
                    and float(np.max(np.abs(constant_endpoint_response))) < 1e-14,
            "incidence_identity_max_abs_error": incidence_error,
            "constant_mode_max_abs": float(np.max(np.abs(constant_endpoint_response))),
        },
        "link_reversal_antisymmetry": {
            "pass": reversal_error < 1e-14,
            "max_abs_error": reversal_error,
        },
        "allowed_static_gauge_invariance": {
            "pass": gauge_vector_max < 1e-14
                    and gauge_tensor_norm_max > 1e-4
                    and gauge_h00_max < 1e-14
                    and gauge_h0i_max < 1e-14
                    and gauge_redshift_max < 1e-14,
            "endpoint_gauge_vector_max_abs": gauge_vector_max,
            "endpoint_tensor_norm_max": gauge_tensor_norm_max,
            "h00_max_abs": gauge_h00_max,
            "h0i_max_abs": gauge_h0i_max,
            "redshift_response_max_abs": gauge_redshift_max,
            "statement": (
                "V fixes every clock endpoint, is stationary with V_0=0, "
                "and produces a nonzero spatial gauge tensor but zero redshift."
            ),
        },
        "four_link_rank_restoration": {
            "pass": selected_rank == len(modes) and three_rank <= len(modes) - 1,
            "selected_indices": selected.tolist(),
            "selected_labels": [links[i].label for i in selected],
            "selected_rank": selected_rank,
            "selected_singular_values": selected_singular.tolist(),
            "selected_rank_threshold": selected_threshold,
            "selected_fisher_eigenvalues": selected_fisher_eigs.tolist(),
            "three_link_rank": three_rank,
            "row_count_lower_bound": len(modes),
        },
    }
    all_pass = all(check["pass"] for check in checks.values())
    report = {
        "experiment": "static_redshift_conformal_rank_restoration",
        "conventions": {
            "metric": "diag(-1,1,1,1)",
            "delay": "A_gamma h = 0.5 integral h(k,k) d lambda",
            "redshift": "R_AB h = 0.5 [h_00(B)-h_00(A)]",
            "conformal_redshift": "R_AB(phi eta) = 0.5 [phi(A)-phi(B)]",
            "allowed_static_gauge": (
                "stationary V with V_0=0 and V=0 at every clock endpoint"
            ),
            "svd_relative_tolerance": svd_tol,
            "zero_matrix_absolute_tolerance": 1e-14,
        },
        "dimensions": {
            "conformal_modes": len(modes),
            "observers": len(observers),
            "candidate_clock_links": len(links),
            "delay_rays": len(rays),
        },
        "observer_positions": {
            observer.label: observer.position.tolist() for observer in observers
        },
        "mode_labels": [mode.label for mode in modes],
        "link_labels": [link.label for link in links],
        "checks": checks,
        "all_pass": all_pass,
        "paper_claim": (
            "Delay channels annihilate stationary conformal modes, while "
            "endpoint clock comparisons restore exactly the rank carried by "
            "the endpoint-difference matrix."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not all_pass:
        failed = [name for name, check in checks.items() if not check["pass"]]
        raise SystemExit(
            f"STOP: static-redshift check(s) failed: {failed}. See {output}"
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/static_redshift_experiment.json"),
    )
    parser.add_argument("--order", type=int, default=DEFAULT_ORDER)
    parser.add_argument("--svd-tol", type=float, default=1e-10)
    args = parser.parse_args()
    report = run(args.output, order=args.order, svd_tol=args.svd_tol)
    print(json.dumps({
        "all_pass": report["all_pass"],
        "report": str(args.output),
        "selected_links": report["checks"]["four_link_rank_restoration"]["selected_labels"],
    }, indent=2))


if __name__ == "__main__":
    main()
