# TASK 1 — P1-E0 TOY REPRODUCTION AND INDEPENDENT AUDIT

## Identity
- branch: `claude/experiment-review-mac-rthiz1`
- commit: `fdd6cfad88ddcc668f0ef297308400b6b0732029`  (source tree dirty: True)
- config: `paper1_experiment_registry_v0.2.yaml`  sha256 `2ba66f0209fe1cdec97b8cf5862494c22fb94704318f9601a7a1f9eb4b783796`
- environment sha256: `2a20e241e1761eea7769c0c2d6d1b2c0a47388deba3f9875fe68cba909b1c9dd`
- hardware: Linux x86_64, 4 cores, 15.7 GiB
- python 3.11.15; numpy 2.4.6, scipy 1.17.1, torch 2.13.0, aart 2.1.10
- run_id: `E0_20260824T215711Z_2ba66f02`

## Mechanical gate result
**STOP on G1 / PASS on everything E0 can execute without the original.**

G1 is the gate E0 exists to run, and it is recorded `NOT_RUN`. The v0.1
generator and the v0.1 manuscript are not present in this session, so there is
no original output to compare the reimplementation against. No substitute
number is reported in its place, and no claim of reproduction is made.

Per the protocol's own rule ("Failure blocks all later phases"), the physical
Kerr phases E3 and beyond are blocked on the v0.1 generator arriving. The
mathematical phases E1 and E2 do not depend on it and proceed.

## Inputs
- registered symbols: `{'H': 44, 'K': 6, 'W': 24, 'M': 2, 'N_max': 5, 'D': 4, 'Gamma': 0.6, 'restricted_dim': 24}`
- operator seed: 42

## Results

### Symbol pinning

Four of the seven registered symbols are pinned by an exact arithmetic identity:

> **H == W + N_max * D** — 44 == 24 + 5*4 — holds: **True**

The deepest order's window ends exactly at the end of the history, so the
registered history length is precisely what the order stack consumes. Nothing
is truncated and nothing is padded. This fixes H, W, N_max and D.

The remaining pair (K, M) is **not** pinned by the registered list. The two
readings are not equivalent, and only one of them yields a usable experiment.

#### Reading A — K = 6 screen samples, M = 2 source-plane cells

| arm | data dim | full rank / source dim | restricted rank / dim | restricted sigma_min | restricted kappa+ |
|---|---:|---:|---:|---:|---:|
| `resolved_identical` | 864 | 88 / 88 | NOT CONSTRUCTIBLE | – | – |
| `resolved_diverse` | 864 | 88 / 88 | NOT CONSTRUCTIBLE | – | – |
| `unresolved_diverse` | 144 | 48 / 88 | NOT CONSTRUCTIBLE | – | – |

Reading A is rejected on three independent grounds:

1. the three registered maximum-order rows are **not distinct** — two of them
   collapse to the same rank, so a v0.1 table with three separate rows could
   not have been produced this way;
2. the resolved arms are full rank at 88/88, making the null-space experiment
   vacuous — there is nothing to be identifiable *about*;
3. the registered 24-dimensional smooth restricted model **cannot be built at
   all**: a separable class over 2 source cells admits at most 2 spatial modes,
   so 4 x 6 = 24 does not exist.

#### Reading B — K = 6 source-plane cells, M = 2 screen channels

| arm | data dim | full rank / source dim | restricted rank / dim | restricted sigma_min | restricted kappa+ |
|---|---:|---:|---:|---:|---:|
| `resolved_identical` | 288 | 88 / 264 | 12 / 24 | 7.5961e-02 | 1.6407e+01 |
| `resolved_diverse` | 288 | 216 / 264 | 24 / 24 | 9.8798e-03 | 1.3225e+02 |
| `unresolved_diverse` | 48 | 48 / 264 | 22 / 24 | 1.2839e-07 | 1.6840e+07 |

Reading B produces three distinct rows, a non-trivial null space in every arm,
and a 24-dimensional smooth class that exists exactly (4 spatial x 6 temporal).
It is adopted as the operative reading and is flagged as an inference, not a
fact: it is confirmed or refuted the moment the v0.1 generator arrives.

### What the three rows show

Under reading B the registered rows separate the paper's own claim hierarchy:

- `resolved_identical` — delay diversity alone. Restricted rank 12 of 24: half
  the smooth class is invisible however much the data are collected.
- `resolved_diverse` — delay plus spatial diversity. Restricted rank 24 of 24,
  a strict **C1 structural gain**: 12 directions that no amount of order-zero
  data could constrain become identifiable.
- `unresolved_diverse` — the same physics, order labels destroyed. Restricted
  rank 22 of 24 but restricted sigma_min 1.28e-07 and kappa+ 1.68e+07. This is
  the distinction the paper turns on: the collapse is **not** mainly a rank
  loss, it is a conditioning catastrophe. Reporting rank 22 alone would read as
  "almost everything is still visible", which is false in any operational sense.

Note that `resolved_identical` has a *larger* restricted sigma_min (7.60e-02)
than `resolved_diverse` (9.88e-03). These are not comparable as quality scores:
they are minima over supports of different size (12 versus 24). Spatial
diversity converts twelve strictly invisible directions into visible but
weakly-constrained ones. That is a gain in identifiability bought at a cost in
conditioning, and it must be reported as both.

### Reconstruction diagnostics

Median NRMSE on the 24-dimensional smooth class, 8 registered `test_id`
sources, ridge with three hyperparameter rules:

| arm | SNR | discrepancy | gcv | oracle |
|---|---:|---:|---:|---:|
| `resolved_diverse` | noise-free | – | 0.00000 | 0.00000 |
| `resolved_diverse` | 10 | 0.52430 | 0.52238 | 0.49786 |
| `resolved_diverse` | 100 | 0.20265 | 0.13730 | 0.09564 |
| `resolved_identical` | noise-free | – | 0.48335 | 0.48335 |
| `resolved_identical` | 10 | 0.58167 | 0.58607 | 0.55569 |
| `resolved_identical` | 100 | 0.50328 | 0.48595 | 0.47004 |
| `unresolved_diverse` | noise-free | – | 0.41449 | 0.41449 |
| `unresolved_diverse` | 10 | 0.71524 | 0.73020 | 0.68355 |
| `unresolved_diverse` | 100 | 0.61955 | 0.70079 | 0.61523 |

Median error of the **null component** of the same reconstructions:

| arm | SNR | discrepancy | gcv | oracle |
|---|---:|---:|---:|---:|
| `resolved_diverse` | noise-free | – | 0.00000 | 0.00000 |
| `resolved_diverse` | 10 | 0.00000 | 0.00000 | 0.00000 |
| `resolved_diverse` | 100 | 0.00000 | 0.00000 | 0.00000 |
| `resolved_identical` | noise-free | – | 0.80140 | 0.80140 |
| `resolved_identical` | 10 | 0.80140 | 0.80140 | 0.80140 |
| `resolved_identical` | 100 | 0.80140 | 0.80140 | 0.76804 |
| `unresolved_diverse` | noise-free | – | 0.52496 | 0.52496 |
| `unresolved_diverse` | 10 | 0.52497 | 0.52497 | 0.52497 |
| `unresolved_diverse` | 100 | 0.52497 | 0.52497 | 0.52497 |

In the two rank-deficient arms the null-component error is invariant across
every *deployable* rule and every SNR — 0.8014 for `resolved_identical` and
0.5250 for `unresolved_diverse`, identical from noise-free through SNR 10 to
SNR 100. That is the E0 result worth carrying into the manuscript: the
unidentified component is not estimated badly, it is not estimated at all.
Whatever appears there was chosen by the regulariser, and adding photons does
not change it.

There is exactly one exception in the table, and it is diagnostic rather than
contradictory: `resolved_identical` at SNR 100 under the **oracle** rule reads
0.76804 instead of 0.80140. The oracle rule selects lambda by looking at the
truth, so it can pull the regularised solution slightly toward the true null
component. No rule that lacks the answer can do this. The exception is
therefore a direct measurement of how much of an oracle-tuned result comes from
the oracle rather than from the data, and it is the reason the oracle curve is
reported only as a ceiling.

`resolved_diverse` reaches exactly 0.0 null error because its restricted
operator has no null space to begin with.

The oracle-tuned curve is never beaten by a deployable rule
(`E0_oracle_is_upper_bound`, measured 0.0), which is the registered expectation.

## Diagnostics
| gate | status | measured | threshold | note |
|---|---|---|---|---|
| G1_v01_reproduction_relative | **NOT_RUN** | - | 1e-08 | v0.1 generator and v0.1 manuscript absent from this session; no original output exists to compare against. The independent reimplementation ran and is reported, but agreement with the original is untested and is NOT claimed. |
| G2_dense_operator_relative | **PASS** | 0 | 1e-10 |  |
| G3_adjoint_relative | **PASS** | 2.91588e-14 | 1e-08 | worst of 20 probes, seed 42 |
| G4_order_collapse_relative | **PASS** | 0 | 1e-10 |  |
| G5_kernel_normalized_residual | **PASS** | 5.51347e-17 | 1e-08 | null dim 48, source moved by 0.064 relative |
| G6_monotonicity_relative_negative_eigenvalue | **PASS** | 1.67537e-16 | 1e-10 | min relative increment eigenvalue per order: [-1.68e-16, -4.9e-17, -5.9e-17, -3.7e-17, -2e-17] |
| G9_source_split_disjoint | **PASS** | 0 | 0 | all namespaces disjoint |
| G13_replay | **PASS** | 0 | 0 | 3 builds compared |
| G11_cpu_mps_inference_relative | **NOT_RUN** | - | 0.0001 | no MPS device on this host; see protocol_deviations D2_no_mps |
| E0_oracle_is_upper_bound | **PASS** | 0 | 1e-12 | max amount by which a deployable rule beat the oracle-tuned curve; must be non-positive up to rounding |

## Deviations
**D1_platform** — registered: macos_native (execution_target in registry); actual: Linux x86_64.

  Effect: No macOS-specific or Apple-Silicon-specific result can be claimed. All float64 CPU numerics are platform-portable and unaffected; runtime and peak-RSS rows describe this Linux host, not a Mac.

**D2_no_mps** — registered: optional Apple-Silicon MPS for compact neural models; actual: no MPS device present.

  Effect: Gate G11 (CPU/MPS inference parity) cannot be executed and is recorded NOT_RUN. Neural training runs on CPU float32. CUDA is not substituted for MPS in any registered gate.

**D3_missing_v01_generator** — registered: run the supplied v0.1 generator
unchanged and compare. Actual: neither the generator nor the v0.1 manuscript is
present in this session.

  Effect: G1 is NOT_RUN. The reimplementation is reported on its own terms.
  The (K, M) reading is an inference from three consistency arguments, not a
  reproduction. Physical Kerr claims stay blocked until G1 can run.

## Claim effect
Permits: the operator library, the gate battery, and E1/E2 as mathematical
computation.
Demotes: nothing yet.
Forbids: describing any of this as "reproducing v0.1"; describing reading B as
established; any black-hole language whatsoever at this stage.

## Artifacts
- `artifacts/tables/e0_reproduction.parquet` (+ `.csv`)
- `artifacts/tables/e0_reconstruction.parquet` (+ `.csv`)
- `artifacts/e0_reproduction/e0_metrics.json`
- `artifacts/gates/correctness_gates.json`
- `artifacts/manifests/E0_20260824T215711Z_2ba66f02.json`

## Next authorized step
P3 — E1 structured factorial and E2 null-mode atlas. (P2, the operator library
and its gates, is complete and is exercised by 73 passing tests.)
