#!/usr/bin/env python3
"""P1-E2 -- null-space and near-null mode atlas.

Rank alone is not a physical statement.  This turns each null and near-null
direction into a labelled object: which retarded epoch it lives in, how fast it
oscillates, which source-plane harmonic it carries, and whether it is invisible
because the operator annihilates it or merely because it is faint.

Every candidate null vector is verified by an actual matvec before it is called
null, and every one is injected as x0 +- alpha v to confirm the predicted data
indistinguishability.  A vector that fails either check is reported, not
dropped.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.fft as sfft

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phrt.audits.rank import (lapack_rank_threshold, numerical_null_basis,
                              smallest_visible_modes, spectrum_of)
from phrt.audits.subspaces import principal_angles, subspace_report, visible_subspace
from phrt.config import load_registry
from phrt.io.manifests import RunManifest, Gate, gate_from_tolerance, make_run_id, merge_gate_file
from phrt.io.tables import write_table
from phrt.operators import mixing
from phrt.operators.structured import ToySpec, build_order_blocks
from phrt.sources.toy_classes import source_class

N_CELLS, N_SCREEN, HISTORY, WINDOW = 6, 2, 44, 24
OPERATIONAL_THRESHOLD = 1.0
SEED = 9000
N_NEAR = 20
ALPHA = 1.0

ARMS = [
    ("direct_only", "constant", "rotation_shear"),
    ("resolved", "constant", "identical"),
    ("resolved", "none", "rotation_shear"),
    ("resolved", "constant", "rotation_shear"),
    ("partial", "constant", "rotation_shear"),
    ("unresolved_sum", "constant", "rotation_shear"),
]


def spec_for(delay: str, spatial: str) -> ToySpec:
    return ToySpec(history_length=HISTORY, window=WINDOW, n_screen=N_SCREEN,
                   n_cells=N_CELLS, max_order=5, delay_step=4, gamma=0.6,
                   delay=delay, spatial=spatial, attenuation="exponential")


def mixer_for(readout: str, n: int):
    if readout == "direct_only":
        return mixing.OrderMixer(np.eye(n)[:1], "direct_only")
    if readout == "resolved":
        return mixing.resolved(n)
    if readout == "partial":
        return mixing.partial(n, 0.10)
    if readout == "unresolved_sum":
        return mixing.unresolved(n)
    raise ValueError(readout)


def label_mode(v_source: np.ndarray) -> dict:
    """Physical labels for one direction, in the flattened (cell, time) space.

    The history axis is indexed by retarded age (see
    ``phrt.operators.structured``): index 0 is the most recent source state and
    increasing index goes deeper into the past, because order ``n`` carries
    delay ``Delta_n`` and therefore samples ages ``[Delta_n, Delta_n+W)``.  The
    energy centre of mass along that axis *is* the mode's retarded age; it needs
    no reflection.  Labels are computed on the energy profile, so they do not
    depend on the sign of the singular vector, which is arbitrary.
    """
    X = v_source.reshape(N_CELLS, HISTORY)
    energy_t = (X ** 2).sum(axis=0)
    energy_c = (X ** 2).sum(axis=1)
    total = max(float(energy_t.sum()), 1e-300)

    spec_t = np.abs(sfft.rfft(X, axis=1)) ** 2
    freq_profile = spec_t.sum(axis=0)
    dom_freq = int(np.argmax(freq_profile))
    nyq = HISTORY // 2

    spec_c = np.abs(sfft.rfft(X, axis=0)) ** 2
    dom_harm = int(np.argmax(spec_c.sum(axis=1)))

    t_index = np.arange(HISTORY, dtype=float)
    com = float((energy_t * t_index).sum() / total)
    spread = float(np.sqrt(max((energy_t * (t_index - com) ** 2).sum() / total, 0.0)))
    # radial/cell support: how many cells hold 90 percent of the energy
    order = np.sort(energy_c)[::-1]
    cum = np.cumsum(order) / max(order.sum(), 1e-300)
    support = int(np.searchsorted(cum, 0.90) + 1)
    return {
        "dominant_temporal_frequency": dom_freq,
        "dominant_temporal_frequency_normalised": dom_freq / max(nyq, 1),
        "temporal_centre_of_mass": com,
        "retarded_age": float(com),
        "temporal_spread": spread,
        "dominant_azimuthal_harmonic": dom_harm,
        "cell_support_90pct": support,
        "max_cell_fraction": float(energy_c.max() / max(energy_c.sum(), 1e-300)),
    }


def atlas_rows(run_id: str, cfg_hash: str) -> tuple[list[dict], list[dict], list[dict]]:
    modes: list[dict] = []
    inject: list[dict] = []
    onset: list[dict] = []
    Q = source_class("smooth_separable", N_CELLS, HISTORY)
    rng = np.random.default_rng(SEED)

    for readout, delay, spatial in ARMS:
        spec = spec_for(delay, spatial)
        blocks = build_order_blocks(spec, SEED)
        m = mixer_for(readout, len(blocks))
        B = m.whiten(blocks)
        BQ = B @ Q
        arm = f"{readout}|{delay}|{spatial}"

        for space, Bmat, basis in (("restricted", BQ, Q), ("full", B, None)):
            sp = spectrum_of(Bmat, Bmat.shape[1],
                             operational_threshold=OPERATIONAL_THRESHOLD)
            V = numerical_null_basis(Bmat)
            svals, Vis = smallest_visible_modes(Bmat, N_NEAR)

            def to_source(vec):
                return basis @ vec if basis is not None else vec

            for j in range(V.shape[1]):
                lab = label_mode(to_source(V[:, j]))
                modes.append({"run_id": run_id, "config_hash": cfg_hash, "arm": arm,
                              "readout": readout, "delay_structure": delay,
                              "spatial_structure": spatial, "space": space,
                              "kind": "exact_null", "index": j,
                              "singular_value": 0.0, **lab})
            for j in range(Vis.shape[1]):
                lab = label_mode(to_source(Vis[:, j]))
                modes.append({"run_id": run_id, "config_hash": cfg_hash, "arm": arm,
                              "readout": readout, "delay_structure": delay,
                              "spatial_structure": spatial, "space": space,
                              "kind": "near_null", "index": j,
                              "singular_value": float(svals[j]), **lab})

            # injection: x0 +- alpha v must be indistinguishable (null) or
            # ambiguous at the stated noise level (near-null)
            x0 = rng.normal(size=Bmat.shape[1])
            base = max(float(np.linalg.norm(Bmat @ x0)), 1e-300)
            for kind, vecs, vals in (("exact_null", V, np.zeros(V.shape[1])),
                                     ("near_null", Vis, svals)):
                for j in range(vecs.shape[1]):
                    v = vecs[:, j]
                    dy = Bmat @ (ALPHA * v)
                    inject.append({
                        "run_id": run_id, "arm": arm, "space": space, "kind": kind,
                        "index": j, "singular_value": float(vals[j]),
                        "source_separation": float(np.linalg.norm(2 * ALPHA * v)),
                        "data_separation": float(np.linalg.norm(2 * dy)),
                        "data_separation_relative": float(np.linalg.norm(2 * dy) / base),
                        "mahalanobis_pair_distance": float(np.linalg.norm(2 * dy)),
                    })

        # mode onset with retained order, on the restricted class
        prev_alg = prev_op = 0
        prev_vis = np.zeros((Q.shape[1], 0))
        for n in range(spec.max_order + 1):
            mn = mixer_for(readout, n + 1) if readout != "direct_only" else mixer_for("direct_only", n + 1)
            Bn = mn.whiten(blocks[: n + 1]) @ Q
            sp = spectrum_of(Bn, Q.shape[1], operational_threshold=OPERATIONAL_THRESHOLD)
            vis = visible_subspace(Bn)
            ang = principal_angles(prev_vis, vis) if prev_vis.shape[1] else np.zeros(0)
            newly = []
            if vis.shape[1] > prev_vis.shape[1]:
                # directions in the new visible space orthogonal to the old one
                resid = vis - (prev_vis @ (prev_vis.T @ vis) if prev_vis.shape[1] else 0.0)
                norms = np.linalg.norm(resid, axis=0)
                for j in np.argsort(norms)[::-1][: vis.shape[1] - prev_vis.shape[1]]:
                    newly.append(label_mode(Q @ vis[:, j]))
            onset.append({
                "run_id": run_id, "arm": arm, "max_order": n,
                "algebraic_rank": sp.numerical_rank, "operational_rank": sp.operational_rank,
                "new_algebraic": sp.numerical_rank - prev_alg,
                "new_operational": sp.operational_rank - prev_op,
                "sigma_min_positive": sp.sigma_min_positive,
                "max_principal_angle_to_previous_rad": float(ang[-1]) if ang.size else float("nan"),
                "mean_new_mode_retarded_age": float(np.mean([x["retarded_age"] for x in newly])) if newly else float("nan"),
                "mean_new_mode_frequency": float(np.mean([x["dominant_temporal_frequency_normalised"] for x in newly])) if newly else float("nan"),
            })
            prev_alg, prev_op, prev_vis = sp.numerical_rank, sp.operational_rank, vis
    return modes, inject, onset


def main() -> int:
    t0 = time.time()
    reg = load_registry()
    run_id = make_run_id("E2", reg.sha256)
    man = RunManifest(run_id=run_id, experiment_id="E2",
                      seeds={"seed": SEED, "alpha": ALPHA,
                             "operational_threshold": OPERATIONAL_THRESHOLD})
    man.add_input(reg.path)

    modes, inject, onset = atlas_rows(run_id, reg.sha256)
    import pandas as pd
    idf = pd.DataFrame(inject)

    # every exact null vector must move the source and not the data
    ex = idf[idf.kind == "exact_null"]
    if len(ex):
        man.add_gate(gate_from_tolerance(
            "E2_null_injection_invisible", float(ex.data_separation_relative.max()), 1e-8,
            note=f"{len(ex)} injected exact-null vectors; worst relative data change"))
        man.add_gate(gate_from_tolerance(
            "E2_null_injection_moves_source", float(-ex.source_separation.min()), -1e-6,
            note="every exact-null injection must actually change the source"))
    else:
        man.add_gate(Gate("E2_null_injection_invisible", "NOT_RUN",
                          note="no arm in this atlas has an exact null space"))

    # near-null modes must be strictly visible, however faintly
    nn = idf[idf.kind == "near_null"]
    man.add_gate(gate_from_tolerance(
        "E2_near_null_is_not_null", float(-nn.singular_value.min()), 0.0,
        note="smallest retained near-null singular value must be strictly positive"))

    for p in (write_table(modes, "e2_mode_atlas"),
              write_table(inject, "e2_injection"),
              write_table(onset, "e2_mode_onset")):
        man.add_output(p)
    mp = man.write(reg.path, reg.sha256, runtime_seconds=time.time() - t0)
    merge_gate_file(man.gates, run_id)

    mdf = pd.DataFrame(modes)
    print(f"atlas rows {len(mdf)}  injections {len(idf)}  onset rows {len(onset)}")
    print("\nexact null dimension per arm (restricted class, 24-dim):")
    r = mdf[(mdf.space == "restricted") & (mdf.kind == "exact_null")]
    for arm, sub in r.groupby("arm"):
        print(f"  {arm:44s} nullity {len(sub):3d}  "
              f"median retarded age {sub.retarded_age.median():6.2f}  "
              f"median freq {sub.dominant_temporal_frequency_normalised.median():.3f}")
    print("\ngates")
    for g in man.gates:
        print(f"  {g.name:36s} {g.status:8s} measured={g.measured}")
    print(f"\nmanifest {mp}\ntotal {time.time()-t0:.0f}s")
    return 1 if man.failed_gates else 0


if __name__ == "__main__":
    raise SystemExit(main())
