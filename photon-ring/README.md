# Photon-Ring Retarded-Time Tomography I — Mac protocol v0.2, executed

Reproducible, matrix-free computational campaign for **Paper I: Null Spaces,
Identifiability, and Stability of Historical Inversion from Near-Critical Null
Geodesics**, implementing `docs/Paper_I_Comprehensive_Experiment_Protocol_Mac_v0.2.md`
and `docs/Paper_I_Mac_Coding_Agent_Handoff_v0.2.md`.

Theory and controlled synthetic computation only. No telescope detection, no
laboratory result, and no recovery from a resolved real photon ring is claimed
anywhere in this tree.

## Status

| phase | state |
|---|---|
| P0 governance, environment lock, provenance | **done** |
| P1 E0 toy reproduction | **partial — G1 blocked**, see below |
| P2 matrix-free operator library + gates G2–G6, G9, G13 | **done**, 73 tests |
| P3 E1 factorial + E2 mode atlas | in progress |
| P4+ physical Kerr (E3–E10) | blocked on G1 |

## Two live blockers

**B1 — the v0.1 generator is absent.** E0 requires running the supplied v0.1
generator unchanged and comparing it against an independent reimplementation.
Neither the generator nor the v0.1 manuscript is in this session. Gate G1 is
recorded `NOT_RUN`, not passed, and no reproduction is claimed. The protocol's
own rule blocks the physical Kerr phases until it can run. The reimplementation
itself is complete and reported.

A consequence: two of the seven registered E0 symbols (K and M) are not pinned
by the registered list. `artifacts/reports/P1_E0_REPRODUCTION.md` gives three
independent consistency arguments for the operative reading and labels it an
inference. One line of the v0.1 generator settles it.

**B2 — no independent geodesic tracer.** AART 2.1.10 installs and imports from
PyPI, so the primary ray tracer is available. `kgeo` is not on PyPI, so the
registered cross-tracer gate G8 needs a vendored or hand-written second
implementation before any physical ray map is trusted.

## Protocol deviations

Recorded in every manifest under `protocol_deviations`, and in
`artifacts/reports/TASK0_ENVIRONMENT.md`:

- **D1_platform** — the protocol targets macOS; this runs on Linux x86_64. All
  float64 CPU numerics are platform-portable; runtime and peak-RSS rows describe
  this host, and no macOS-specific claim is made.
- **D2_no_mps** — no Apple-Silicon device, so gate G11 (CPU/MPS inference
  parity) is `NOT_RUN`. CUDA is deliberately **not** substituted for MPS in any
  registered gate.
- **D3_missing_v01_generator** — see B1.

## Layout

```
configs/     frozen registry (sha256 travels in every manifest)
schemas/     run-manifest JSON schema
src/phrt/    library: config, provenance, seeds, operators, audits, inverse, io
scripts/     one runner per experiment, plus the report generator
tests/       73 tests, including gate tests that assert the gates catch failures
artifacts/   manifests, gates, tables (parquet + csv twin), reports, figures
docs/        the protocol and handoff, vendored unchanged
```

## Running

```bash
cd photon-ring
python -m pytest tests -q
python scripts/reproduce_v01.py        # E0
python scripts/build_report.py         # regenerate reports from artifacts
```

Every reported number is read back out of a canonical table; none is typed into
a report or a plotting script by hand.
