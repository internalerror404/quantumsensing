#!/usr/bin/env python3
"""P1-E0 -- reproduce and independently audit the v0.1 toy experiment.

Blocker, stated up front and carried into the gate file: the v0.1 generator and
the v0.1 manuscript are **not present in this session**.  Only the registered
dimensions (H=44, K=6, W=24, M=2, N_max=5, D=4, Gamma=0.6, a 24-dimensional
smooth restricted model, seed 42) are available.  Therefore:

  * G1 (original-versus-reimplementation agreement) is recorded NOT_RUN.  It is
    the one gate E0 exists to run, and no substitute is reported in its place.
  * Everything E0 can do without the original -- the independent operator, the
    three maximum-order rows, the operator gates, and the oracle-versus-
    deployable comparison -- is executed and reported.

Symbol pinning.  Four of the seven registered symbols are pinned by an exact
arithmetic identity: H = W + N_max*D, i.e. 44 = 24 + 5*4.  The deepest order's
window ends precisely at the end of the history, so the history length is
exactly what the order stack consumes.  The remaining pair (K, M) is not pinned
by the registered list, and the two readings are *not* equivalent -- one of
them makes the experiment vacuous.  Both are computed below.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phrt.audits import gates
from phrt.audits.rank import spectrum_of
from phrt.config import load_registry, repo_root
from phrt.inverse.base import nrmse, ridge_solve, select_hyperparameter, split_visible_null
from phrt.io.manifests import Gate, RunManifest, make_run_id, merge_gate_file
from phrt.io.tables import write_table
from phrt.operators import mixing
from phrt.operators.historical import structured_operator
from phrt.operators.structured import ToySpec, build_order_blocks
from phrt.operators.whitening import NoiseModel, gram
from phrt.audits.subspaces import visible_subspace
from phrt.seeds import SeedNamespaces
from phrt.sources.toy_classes import ClassNotConstructible, source_class

E0_SEED = 42
REGISTERED = dict(H=44, K=6, W=24, M=2, N_max=5, D=4, Gamma=0.6, restricted_dim=24)

# The two candidate readings of the unpinned pair (K, M).
READINGS = {
    "A_screen6_cells2": dict(n_screen=6, n_cells=2,
                             gloss="K=6 screen samples, M=2 source-plane cells"),
    "B_cells6_screen2": dict(n_screen=2, n_cells=6,
                             gloss="K=6 source-plane cells, M=2 screen channels"),
}

ARMS = (
    ("resolved_identical", "resolved", "identical"),
    ("resolved_diverse", "resolved", "rotation_shear"),
    ("unresolved_diverse", "unresolved", "rotation_shear"),
)


def spec_for(reading: str, spatial: str) -> ToySpec:
    r = READINGS[reading]
    return ToySpec(history_length=REGISTERED["H"], window=REGISTERED["W"],
                   n_screen=r["n_screen"], n_cells=r["n_cells"],
                   max_order=REGISTERED["N_max"], delay_step=REGISTERED["D"],
                   gamma=REGISTERED["Gamma"], delay="constant", spatial=spatial,
                   attenuation="exponential")


def arm_row(reading: str, arm: str, readout: str, spatial: str) -> dict:
    spec = spec_for(reading, spatial)
    blocks = build_order_blocks(spec, E0_SEED)
    n_orders = spec.max_order + 1
    mixer = mixing.resolved(n_orders) if readout == "resolved" else mixing.unresolved(n_orders)
    A, B = mixer.apply(blocks), mixer.whiten(blocks)
    full = spectrum_of(B, spec.source_dimension)
    try:
        Q = source_class("smooth_separable", spec.n_cells, spec.history_length)
    except ClassNotConstructible as exc:
        # Not a numerical failure: at these dimensions the registered
        # 24-dimensional smooth separable class does not exist.
        return {
            "reading": reading, "arm": arm, "readout": readout, "spatial": spatial,
            "source_dimension": spec.source_dimension,
            "data_dimension": int(A.shape[0]),
            "restricted_dimension": -1,
            "numerical_rank": full.numerical_rank,
            "nullity": full.nullity,
            "sigma_max": full.sigma_max,
            "sigma_min_positive": full.sigma_min_positive,
            "kappa_positive": full.kappa_positive,
            "stable_rank": full.stable_rank,
            "effective_rank": full.effective_rank,
            "trace_information": full.trace_information,
            "restricted_rank": -1, "restricted_nullity": -1,
            "restricted_sigma_min": float("nan"),
            "restricted_kappa_positive": float("nan"),
            "restricted_status": f"NOT_CONSTRUCTIBLE: {exc}",
        }
    restricted = spectrum_of(B @ Q, Q.shape[1])
    return {
        "restricted_status": "ok",
        "reading": reading, "arm": arm, "readout": readout, "spatial": spatial,
        "source_dimension": spec.source_dimension,
        "data_dimension": int(A.shape[0]),
        "restricted_dimension": int(Q.shape[1]),
        "numerical_rank": full.numerical_rank,
        "nullity": full.nullity,
        "sigma_max": full.sigma_max,
        "sigma_min_positive": full.sigma_min_positive,
        "kappa_positive": full.kappa_positive,
        "stable_rank": full.stable_rank,
        "effective_rank": full.effective_rank,
        "trace_information": full.trace_information,
        "restricted_rank": restricted.numerical_rank,
        "restricted_nullity": restricted.nullity,
        "restricted_sigma_min": restricted.sigma_min_positive,
        "restricted_kappa_positive": restricted.kappa_positive,
    }


def reconstruction_diagnostics(reading: str) -> list[dict]:
    """Noise-free and noisy recovery on the restricted class, oracle vs deployable.

    The oracle rule reads the truth and is an optimistic ceiling.  The
    deployable rule (GCV) never sees it.  The registered expectation is
    deployable >= oracle error; a deployable rule that beat the oracle would
    mean the oracle grid or the error metric was misconfigured.
    """
    out = []
    ns = SeedNamespaces.from_registry(load_registry())
    grid = np.logspace(-10, 2, 60)
    for arm, readout, spatial in ARMS:
        spec = spec_for(reading, spatial)
        blocks = build_order_blocks(spec, E0_SEED)
        n_orders = spec.max_order + 1
        mixer = mixing.resolved(n_orders) if readout == "resolved" else mixing.unresolved(n_orders)
        A = mixer.apply(blocks)
        Q = source_class("smooth_separable", spec.n_cells, spec.history_length)
        for snr in (None, 100.0, 10.0):
            nm = NoiseModel.from_snr(A, snr)
            # scale by the mixer's own noise propagation so readouts are
            # compared at equal per-order detector noise, not equal per-channel
            rows_per_channel = A.shape[0] // mixer.n_channels
            nm = NoiseModel(nm.sigma * np.repeat(mixer.noise_scale(), rows_per_channel),
                            nm.name + "_mixerscaled")
            B = nm.whiten(A) @ Q
            vis = visible_subspace(B)
            for idx in range(8):
                sseed = ns.seed("test_id", idx)
                srng = np.random.default_rng(sseed)
                # smooth coefficients with controlled spectral decay
                coef = srng.normal(size=Q.shape[1]) / (1.0 + np.arange(Q.shape[1])) ** 0.5
                nrng = np.random.default_rng([sseed, 7])
                noise = np.zeros(A.shape[0]) if snr is None else nm.sample(nrng)
                y = nm.whiten_data(A @ (Q @ coef) + noise)
                noise_norm = float(np.linalg.norm(nm.whiten_data(noise)))
                for rule in ("oracle", "gcv", "discrepancy"):
                    if rule == "oracle":
                        lam = select_hyperparameter(B, y, grid, "oracle", ridge_solve, truth=coef)
                    elif rule == "gcv":
                        lam = select_hyperparameter(B, y, grid, "gcv", ridge_solve)
                    else:
                        if snr is None:
                            continue
                        lam = select_hyperparameter(B, y, grid, "discrepancy", ridge_solve,
                                                    noise_level=max(noise_norm, 1e-12))
                    xh = ridge_solve(B, y, lam)
                    v, nul = split_visible_null(coef, vis)
                    vh, nh = split_visible_null(xh, vis)
                    out.append({
                        "reading": reading, "arm": arm, "readout": readout,
                        "snr": -1.0 if snr is None else float(snr),
                        "source_seed": int(sseed), "rule": rule,
                        "hyperparameter": float(lam),
                        "nrmse": nrmse(xh, coef),
                        "visible_component_error": nrmse(vh, v) if np.linalg.norm(v) > 0 else 0.0,
                        "null_component_error": float(np.linalg.norm(nh - nul)),
                        "data_residual": float(np.linalg.norm(B @ xh - y)),
                    })
    return out


def main() -> int:
    t0 = time.time()
    reg = load_registry()
    T = reg.data["correctness_gates"]
    root = repo_root()
    run_id = make_run_id("E0", reg.sha256)
    man = RunManifest(run_id=run_id, experiment_id="E0",
                      seeds={"operator_seed": E0_SEED, "registered": REGISTERED})
    man.add_input(reg.path)

    # -- registered rows, both readings ------------------------------------
    rows = [arm_row(rd, arm, ro, sp) for rd in READINGS for arm, ro, sp in ARMS]

    # -- gates on the operative reading ------------------------------------
    spec = spec_for("B_cells6_screen2", "rotation_shear")
    n_orders = spec.max_order + 1
    blocks = build_order_blocks(spec, E0_SEED)
    op = structured_operator(spec, E0_SEED)
    dense = mixing.resolved(n_orders).apply(blocks)
    nm = NoiseModel.homoscedastic(dense.shape[0])
    Bw = nm.whiten(dense)

    man.add_gate(Gate(
        "G1_v01_reproduction_relative", "NOT_RUN",
        threshold=T["G1_v01_reproduction_relative"],
        note=("v0.1 generator and v0.1 manuscript absent from this session; "
              "no original output exists to compare against. The independent "
              "reimplementation ran and is reported, but agreement with the "
              "original is untested and is NOT claimed."),
    ))
    man.add_gate(gates.gate_dense_parity(op, dense, T["G2_dense_operator_relative"]))
    man.add_gate(gates.gate_adjoint(op, T["G3_adjoint_relative"], seed=E0_SEED))
    man.add_gate(gates.gate_order_collapse(
        blocks, mixing.unresolved(n_orders).apply(blocks), T["G4_order_collapse_relative"]))
    man.add_gate(gates.gate_kernel_injection(Bw, T["G5_kernel_normalized_residual"], seed=E0_SEED))
    cumulative = []
    for n in range(n_orders):
        sub = np.vstack(blocks[: n + 1])
        cumulative.append(gram(NoiseModel.homoscedastic(sub.shape[0]).whiten(sub)))
    man.add_gate(gates.gate_gram_monotonicity(
        cumulative, T["G6_monotonicity_relative_negative_eigenvalue"]))
    man.add_gate(gates.gate_seed_splits(SeedNamespaces.from_registry(reg)))
    man.add_gate(gates.gate_replay(
        lambda: np.vstack(build_order_blocks(spec, E0_SEED)), 0.0, n_repeats=3))
    man.add_gate(Gate(
        "G11_cpu_mps_inference_relative", "NOT_RUN",
        threshold=T["G11_cpu_mps_inference_relative"],
        note="no MPS device on this host; see protocol_deviations D2_no_mps",
    ))

    # -- reconstruction ----------------------------------------------------
    recon = reconstruction_diagnostics("B_cells6_screen2")

    # oracle must not be beaten by any deployable rule
    import pandas as pd
    rdf = pd.DataFrame(recon)
    key = ["arm", "snr", "source_seed"]
    piv = rdf.pivot_table(index=key, columns="rule", values="nrmse")
    worst_violation = 0.0
    for rule in ("gcv", "discrepancy"):
        if rule in piv.columns:
            gap = (piv["oracle"] - piv[rule]).max()
            worst_violation = max(worst_violation, float(gap if gap == gap else 0.0))
    man.add_gate(gates.gate_from_tolerance(
        "E0_oracle_is_upper_bound", worst_violation, 1e-12,
        note="max amount by which a deployable rule beat the oracle-tuned curve; "
             "must be non-positive up to rounding"))

    # -- artifacts ---------------------------------------------------------
    tbl = write_table(rows, "e0_reproduction")
    rec = write_table(recon, "e0_reconstruction")
    outdir = root / "artifacts" / "e0_reproduction"
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "run_id": run_id,
        "registered_symbols": REGISTERED,
        "arithmetic_identity": {
            "claim": "H == W + N_max * D",
            "values": f"{REGISTERED['H']} == {REGISTERED['W']} + {REGISTERED['N_max']}*{REGISTERED['D']}",
            "holds": REGISTERED["H"] == REGISTERED["W"] + REGISTERED["N_max"] * REGISTERED["D"],
        },
        "readings": READINGS,
        "rows": rows,
        "gates": {g.name: g.to_dict() for g in man.gates},
    }
    (outdir / "e0_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    for p in (tbl, rec, outdir / "e0_metrics.json"):
        man.add_output(p)
    mp = man.write(reg.path, reg.sha256, runtime_seconds=time.time() - t0)
    merge_gate_file(man.gates, run_id)

    # -- console summary ---------------------------------------------------
    print(f"run_id {run_id}")
    print(f"identity H == W + N_max*D : {metrics['arithmetic_identity']['holds']} "
          f"({metrics['arithmetic_identity']['values']})")
    for rd in READINGS:
        sub = [r for r in rows if r["reading"] == rd]
        ranks = {r["arm"]: r["numerical_rank"] for r in sub}
        rranks = {r["arm"]: r["restricted_rank"] for r in sub}
        distinct = len(set(ranks.values())) == len(ranks)
        print(f"\nreading {rd}  ({READINGS[rd]['gloss']})")
        print(f"  source dim {sub[0]['source_dimension']}, three rows distinct: {distinct}")
        for r in sub:
            head = (f"    {r['arm']:20s} rank {r['numerical_rank']:4d}/"
                    f"{r['source_dimension']:<4d}")
            if r["restricted_status"] == "ok":
                tail = (f"  restricted {r['restricted_rank']:3d}/"
                        f"{r['restricted_dimension']:<3d}"
                        f"  sigma_min {r['restricted_sigma_min']:.4e}"
                        f"  kappa+ {r['restricted_kappa_positive']:.3e}")
            else:
                tail = "  restricted class NOT CONSTRUCTIBLE at these dimensions"
            print(head + tail)
    print("\ngates")
    for g in man.gates:
        m = "" if g.measured is None else f" measured={g.measured:.3e}" if isinstance(g.measured, float) else f" measured={g.measured}"
        print(f"  {g.name:46s} {g.status:8s}{m}")
    print(f"\nmanifest {mp}")
    failed = man.failed_gates
    if failed:
        print(f"FAILED GATES: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
