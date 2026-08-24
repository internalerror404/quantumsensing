#!/usr/bin/env python3
"""P1-E1 -- structured finite-dimensional factorial.

Replaces the single favourable random-projection example with a controlled
factorial that separates delay diversity, spatial diversity, attenuation, and
order collapse.  Random operators appear only as the matched negative control,
never as the base construction: an iid Gaussian block supplies delay and
spatial diversity simultaneously and so answers neither question.

Operational threshold.  A mode counts as operationally visible when its
whitened singular value reaches 1.0, i.e. when a unit-amplitude source produces
a response at the noise level.  This is a unit, not a tuned cut: it is fixed
here before any main-grid number is inspected and is reported alongside the
algebraic rank everywhere, never instead of it.
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from phrt.audits.rank import spectrum_of
from phrt.config import load_registry
from phrt.io.manifests import RunManifest, gate_from_tolerance, make_run_id, merge_gate_file
from phrt.io.tables import write_table
from phrt.operators import mixing
from phrt.operators.structured import (ToySpec, build_order_blocks, duplicate_order,
                                       shuffle_delays, zero_amplitude_high_orders)
from phrt.operators.whitening import NoiseModel
from phrt.sources.toy_classes import ClassNotConstructible, source_class

OPERATIONAL_THRESHOLD = 1.0
N_SEEDS = 20
SEED0 = 9000

N_CELLS, N_SCREEN = 6, 2
DELAYS = ("none", "constant", "perturbed", "cell_dependent")
SPATIALS = ("identical", "rotation", "rotation_shear", "independent")
ATTENUATIONS = (("equalized", 0.0), ("exponential", 0.3),
                ("exponential", 0.6), ("exponential", 0.9))
CLASSES = ("full", "smooth_separable", "localized_atoms", "orbit_tangent")
READOUTS = ("direct_only", "resolved_012", "partial_leakage", "unresolved_sum")
LEAKAGE_FOR_PARTIAL = 0.10


def base_spec(delay: str, spatial: str, atten: str, gamma: float) -> ToySpec:
    return ToySpec(history_length=44, window=24, n_screen=N_SCREEN, n_cells=N_CELLS,
                   max_order=5, delay_step=4, gamma=gamma, delay=delay,
                   spatial=spatial, attenuation=atten)


def readout_operator(blocks, readout: str) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (raw operator, whitened operator, name).

    Whitening goes through the mixer so that the noise a channel actually
    carries -- sigma * ||L[c,:]||_2 -- is the noise it is divided by.  Using a
    flat sigma for every readout would credit the unresolved channel with a
    sqrt(n_orders) gain purely for summing orders together.
    """
    n = len(blocks)
    if readout == "direct_only":
        m = mixing.OrderMixer(np.eye(n)[:1], "direct_only")
    elif readout == "resolved_012":
        m = mixing.resolved(n)
    elif readout == "partial_leakage":
        m = mixing.partial(n, LEAKAGE_FOR_PARTIAL)
    elif readout == "unresolved_sum":
        m = mixing.unresolved(n)
    else:
        raise ValueError(readout)
    return m.apply(blocks), m.whiten(blocks), m.name


def class_bases(seed: int) -> dict[str, np.ndarray]:
    out = {}
    for c in CLASSES:
        try:
            out[c] = source_class(c, N_CELLS, 44, seed=seed)
        except ClassNotConstructible:
            continue
    return out


def evaluate(B: np.ndarray, Q: np.ndarray) -> dict:
    sp = spectrum_of(B @ Q, Q.shape[1], operational_threshold=OPERATIONAL_THRESHOLD)
    s = sp.summary()
    return {
        "numerical_rank": s["numerical_rank"], "operational_rank": s["operational_rank"],
        "nullity": s["nullity"], "operational_nullity": s["operational_nullity"],
        "sigma_max": s["sigma_max"], "sigma_min_positive": s["sigma_min_positive"],
        "restricted_sigma_min": s["sigma_min_positive"],
        "kappa_positive": s["kappa_positive"], "stable_rank": s["stable_rank"],
        "effective_rank": s["effective_rank"], "trace_information": s["trace_information"],
        "rank_rel_1e-08": s["rank_rel_1e-08"], "rank_rel_1e-10": s["rank_rel_1e-10"],
        "rank_rel_1e-12": s["rank_rel_1e-12"],
    }


def factorial_rows(run_id: str, cfg_hash: str) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    onset: list[dict] = []
    combos = list(itertools.product(DELAYS, SPATIALS, ATTENUATIONS, range(N_SEEDS)))
    t0 = time.time()
    for i, (delay, spatial, (atten, gamma), si) in enumerate(combos):
        seed = SEED0 + si
        spec = base_spec(delay, spatial, atten, gamma)
        blocks = build_order_blocks(spec, seed)
        bases = class_bases(seed)
        for readout in READOUTS:
            A, B, readout_name = readout_operator(blocks, readout)
            for cname, Q in bases.items():
                rows.append({
                    "run_id": run_id, "config_hash": cfg_hash,
                    "delay_structure": delay, "spatial_structure": spatial,
                    "attenuation": atten, "gamma": gamma, "seed": seed,
                    "readout": readout_name, "source_class": cname,
                    "source_dimension": int(Q.shape[1]),
                    "data_dimension": int(A.shape[0]),
                    "max_order": spec.max_order, "arm": "factorial",
                    **evaluate(B, Q),
                })
        if si == 0:
            Qs = bases["smooth_separable"]
            prev_rank = prev_op = 0
            for n in range(spec.max_order + 1):
                Bn = mixing.resolved(n + 1).whiten(blocks[: n + 1]) @ Qs
                sp = spectrum_of(Bn, Qs.shape[1],
                                 operational_threshold=OPERATIONAL_THRESHOLD)
                onset.append({
                    "delay_structure": delay, "spatial_structure": spatial,
                    "attenuation": atten, "gamma": gamma, "seed": seed,
                    "max_order": n, "numerical_rank": sp.numerical_rank,
                    "operational_rank": sp.operational_rank,
                    "new_algebraic_modes": sp.numerical_rank - prev_rank,
                    "new_operational_modes": sp.operational_rank - prev_op,
                    "sigma_min_positive": sp.sigma_min_positive,
                    "kappa_positive": sp.kappa_positive,
                })
                prev_rank, prev_op = sp.numerical_rank, sp.operational_rank
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(combos)} cells, {time.time()-t0:.0f}s", flush=True)
    return rows, onset


def control_rows(run_id: str, cfg_hash: str) -> list[dict]:
    """Negative controls, on the same grid geometry as the factorial."""
    rows = []
    Q = source_class("smooth_separable", N_CELLS, 44)
    ref = base_spec("constant", "rotation_shear", "exponential", 0.6)
    for si in range(N_SEEDS):
        seed = SEED0 + si
        controls = {
            "control_gaussian_spatial": build_order_blocks(
                base_spec("constant", "gaussian", "exponential", 0.6), seed),
            "control_shuffled_delays": shuffle_delays(ref, seed),
            "control_duplicate_order": duplicate_order(ref, seed),
            "control_zero_amplitude_high": zero_amplitude_high_orders(ref, seed),
            "reference_structured": build_order_blocks(ref, seed),
        }
        for name, blocks in controls.items():
            for readout in ("resolved_012", "unresolved_sum"):
                A, B, readout_name = readout_operator(blocks, readout)
                rows.append({
                    "run_id": run_id, "config_hash": cfg_hash, "arm": name,
                    "delay_structure": "constant", "spatial_structure": "rotation_shear",
                    "attenuation": "exponential", "gamma": 0.6, "seed": seed,
                    "readout": readout_name, "source_class": "smooth_separable",
                    "leakage_level": float("nan"),
                    "mixer_condition_number": float("nan"), "mixer_rank": -1,
                    "source_dimension": int(Q.shape[1]),
                    "data_dimension": int(A.shape[0]), "max_order": 5,
                    **evaluate(B, Q),
                })
    for eps in load_registry().data["leakage_levels"]:
        for si in range(N_SEEDS):
            seed = SEED0 + si
            blocks = build_order_blocks(ref, seed)
            m = mixing.partial(len(blocks), float(eps))
            A, B = m.apply(blocks), m.whiten(blocks)
            rows.append({
                "run_id": run_id, "config_hash": cfg_hash, "arm": "leakage_sweep",
                "delay_structure": "constant", "spatial_structure": "rotation_shear",
                "attenuation": "exponential", "gamma": 0.6, "seed": seed,
                "readout": m.name, "leakage_level": float(eps),
                "mixer_condition_number": m.condition_number(),
                "mixer_rank": int(np.linalg.matrix_rank(m.L)),
                "source_class": "smooth_separable",
                "source_dimension": int(Q.shape[1]),
                "data_dimension": int(A.shape[0]), "max_order": 5,
                **evaluate(B, Q),
            })
    return rows


def main() -> int:
    t0 = time.time()
    reg = load_registry()
    run_id = make_run_id("E1", reg.sha256)
    man = RunManifest(run_id=run_id, experiment_id="E1",
                      seeds={"seed0": SEED0, "n_seeds": N_SEEDS,
                             "operational_threshold": OPERATIONAL_THRESHOLD})
    man.add_input(reg.path)

    print(f"E1 factorial: {len(DELAYS)}x{len(SPATIALS)}x{len(ATTENUATIONS)}"
          f"x{N_SEEDS} operator cells x {len(READOUTS)} readouts x {len(CLASSES)} classes")
    rows, onset = factorial_rows(run_id, reg.sha256)
    print(f"  factorial rows: {len(rows)}  ({time.time()-t0:.0f}s)")
    ctrl = control_rows(run_id, reg.sha256)
    print(f"  control rows:   {len(ctrl)}")

    import pandas as pd
    df, odf, cdf = pd.DataFrame(rows), pd.DataFrame(onset), pd.DataFrame(ctrl)

    worst = 0.0
    for _, sub in odf.groupby(["delay_structure", "spatial_structure",
                               "attenuation", "gamma"]):
        sub = sub.sort_values("max_order")
        drops = np.diff(sub.numerical_rank.values)
        if drops.size:
            worst = max(worst, float(max(0.0, -drops.min())))
    man.add_gate(gate_from_tolerance(
        "E1_rank_monotone_in_order", worst, 0.0,
        note="largest decrease in cumulative restricted rank when an order is added"))

    dup = cdf[(cdf.arm == "control_duplicate_order") & (cdf.readout == "resolved_012")]
    ref_direct = df[(df.readout == "direct_only") & (df.source_class == "smooth_separable")
                    & (df.delay_structure == "constant")
                    & (df.spatial_structure == "rotation_shear")
                    & (df.attenuation == "exponential") & (df.gamma == 0.6)]
    excess = float(dup.numerical_rank.max() - ref_direct.numerical_rank.max())
    man.add_gate(gate_from_tolerance(
        "E1_duplicate_order_adds_no_rank", max(0.0, excess), 0.0,
        note="rank of the scaled-duplicate control minus rank of direct-only; "
             "a positive value would mean the factorial credits redundancy as information"))

    zero = cdf[(cdf.arm == "control_zero_amplitude_high") & (cdf.readout == "resolved_012")]
    zexcess = float(zero.numerical_rank.max() - ref_direct.numerical_rank.max())
    man.add_gate(gate_from_tolerance(
        "E1_zero_amplitude_adds_no_rank", max(0.0, zexcess), 0.0,
        note="rank when higher orders carry no signal, minus direct-only rank"))

    for p in (write_table(rows, "e1_identifiability_factorial"),
              write_table(onset, "e1_mode_onset"),
              write_table(ctrl, "e1_controls")):
        man.add_output(p)
    mp = man.write(reg.path, reg.sha256, runtime_seconds=time.time() - t0)
    merge_gate_file(man.gates, run_id)

    print("\ngates")
    for g in man.gates:
        print(f"  {g.name:44s} {g.status:8s} measured={g.measured} thr={g.threshold}")
    print(f"\nmanifest {mp}\ntotal {time.time()-t0:.0f}s")
    return 1 if man.failed_gates else 0


if __name__ == "__main__":
    raise SystemExit(main())
