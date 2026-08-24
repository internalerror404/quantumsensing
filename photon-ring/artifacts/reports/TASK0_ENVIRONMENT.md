# TASK 0 — ENVIRONMENT AND GOVERNANCE

## Identity
- branch: `claude/experiment-review-mac-rthiz1`
- commit: `79fbb89d109131ab7454f45094d01720d5397bc6`  (source tree dirty: True)
- config: `paper1_experiment_registry_v0.2.yaml`  sha256 `2ba66f0209fe1cdec97b8cf5862494c22fb94704318f9601a7a1f9eb4b783796`
- environment sha256: `2a20e241e1761eea7769c0c2d6d1b2c0a47388deba3f9875fe68cba909b1c9dd`
- hardware: Linux x86_64, 4 cores, 15.7 GiB
- python 3.11.15; numpy 2.4.6, scipy 1.17.1, torch 2.13.0, aart 2.1.10

## Mechanical gate result
**PASS** for G0 (environment captured, package manifest present, hardware
reported). Two protocol deviations are recorded below and travel inside every
manifest this repository writes.

## Inputs
- `configs/paper1_experiment_registry_v0.2.yaml` — sha256 `2ba66f0209fe1cdec97b8cf5862494c22fb94704318f9601a7a1f9eb4b783796`
- `schemas/run_manifest_schema_v0.2.json` — sha256 `85ad0ab03ee27409a93da3f1010b80d8609559a9b37f283ea029cef2617825a3`
- `environments/physics.yml`, `environments/ml.yml` — vendored unchanged

## Results

Registered execution profiles loaded and validated:

| profile | rays/order | observer times | source dim | geometries | latent dim |
|---|---:|---:|---:|---:|---:|
| `smoke` | 256 | 8 | 36 | 1 | 4 |
| `core` | 1536 | 24 | 224 | 12 | 16 |
| `stress` | 4096 | 32 | 480 | 4 | 24 |
| `refine` | 8192 | 24 | 224 | 2 | None |

Geometry grid: 12 geometries
(spin [0.0, 0.5, 0.9, 0.98] x inclination
[20.0, 50.0, 75.0] deg), orders [0, 1, 2].

Source-seed namespaces are disjoint (gate G9): `train` 100000–109999, `validation` 200000–201999, `test_id` 300000–303999, `test_ood` 400000–403999, `null_pair` 500000–500999

Installed package versions entering the environment hash:

| package | version |
|---|---|
| `numpy` | 2.4.6 |
| `scipy` | 1.17.1 |
| `pandas` | 3.0.5 |
| `pyarrow` | 25.0.1 |
| `h5py` | 3.16.0 |
| `matplotlib` | 3.11.1 |
| `scikit-learn` | 1.9.0 |
| `scikit-image` | 0.26.0 |
| `pyyaml` | 6.0.1 |
| `torch` | 2.13.0 |
| `aart` | 2.1.10 |
| `imageio` | 2.37.4 |

## Diagnostics
- The registry is hashed as raw bytes before parsing, so the recorded sha256 is
  the hash of the file the reviewer can open, not of a YAML round-trip.
- Dirty-tree status is scoped to `src/`, `scripts/`, `configs/`, `tests/`. A
  whole-tree check makes every emitter after the first report a dirty run purely
  because an earlier emitter wrote its own output.
- AART 2.1.10 imports and is available for the E3 physical
  phase. `kgeo` is not on PyPI; the registered independent geodesic cross-check
  (G8) will need a vendored or hand-written second tracer.

## Deviations
**D1_platform** — registered: macos_native (execution_target in registry); actual: Linux x86_64.

  Effect: No macOS-specific or Apple-Silicon-specific result can be claimed. All float64 CPU numerics are platform-portable and unaffected; runtime and peak-RSS rows describe this Linux host, not a Mac.

**D2_no_mps** — registered: optional Apple-Silicon MPS for compact neural models; actual: no MPS device present.

  Effect: Gate G11 (CPU/MPS inference parity) cannot be executed and is recorded NOT_RUN. Neural training runs on CPU float32. CUDA is not substituted for MPS in any registered gate.

## Claim effect
Permits: every float64 CPU numerical result in this repository, which is
platform-portable.
Forbids: any macOS-specific or Apple-Silicon-specific claim, and any use of
gate G11 as evidence. Runtime and peak-RSS rows describe this Linux host.

## Artifacts
- `artifacts/gates/correctness_gates.json`
- `artifacts/manifests/*.json`

## Next authorized step
P1 — E0 toy reproduction.
