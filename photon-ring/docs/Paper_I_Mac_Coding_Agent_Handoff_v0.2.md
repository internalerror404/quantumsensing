# Coding-Agent Handoff — Paper I Mac-Only Research Program v0.2

## Mission

Build a reproducible, matrix-free computational research repository for **Photon-Ring Retarded-Time Tomography I: Null Spaces, Identifiability, and Stability of Historical Inversion from Near-Critical Null Geodesics**.

The repository must determine whether order-resolved Kerr channels add genuinely identifiable historical source modes and whether any ML reconstruction is supported by the likelihood rather than merely selected by a prior.

This is a theory-and-synthetic-computation project. Do not claim real photon-ring detection, recovery from current telescope data, or access to all past moments.

---

## 1. Non-negotiable rules

1. **Read the manuscript and protocol before coding.** Preserve its distinction among full-space rank, restricted injectivity, stability, and prior-driven selection.
2. **Do not start with ML.** Complete E0 and all operator correctness gates first.
3. **Physics/SVD calculations run in float64 on CPU.** Neural training may use float32 on `mps`; all final operator metrics are recomputed on CPU.
4. **Never construct a large dense physical operator unless a smoke-test instance explicitly requests it.** Use `scipy.sparse.linalg.LinearOperator`.
5. **Implement the adjoint explicitly.** Autodiff is not a substitute for the registered inner-product gate.
6. **Cache ray maps.** Source sweeps must never rerun geodesics unnecessarily.
7. **Use native macOS environments.** Do not make Docker the canonical execution path.
8. **No test-set tuning.** Validation selects regularization and stopping. Test truths are read only by scoring code.
9. **No silent fallback or changed profile.** A reduced run requires a new config and run ID.
10. **No result from one random seed becomes a paper claim.**
11. **No ML null-space lifting claim is allowed.** If a network appears to distinguish an exact null pair, assume leakage, hidden source restrictions, numerical error, or train/test contamination until disproved.
12. **A failed gate stops the dependent phase.** Preserve failed artifacts and write a failure report; do not loosen tolerances after seeing the failure.
13. **Do not modify AART internally unless necessary.** If a publication depends on an AART modification, isolate it, document it, and prepare it for public release under the project’s citation/license expectations.
14. **All figures are generated from canonical tables.** Do not manually copy numbers into plotting scripts.
15. **Negative outcomes are first-class outputs.** Do not redesign the source class after seeing main-grid failures.

---

## 2. Repository layout

Create:

```text
photon-ring-history/
  README.md
  CITATION.cff
  LICENSE
  pyproject.toml
  Makefile
  .gitignore
  environments/
    physics.yml
    ml.yml
  configs/
    paper1_experiment_registry_v0.2.yaml
    smoke.yaml
    pilot.yaml
    core.yaml
    stress.yaml
  src/phrt/
    __init__.py
    cli.py
    config.py
    provenance.py
    geometry/
      raymap.py
      aart_adapter.py
      kgeo_adapter.py
      convergence.py
    sources/
      basis.py
      linear_families.py
      hotspot.py
      gaussian_field.py
      ood.py
      splits.py
    operators/
      historical.py
      mixing.py
      visibility.py
      whitening.py
      dense_reference.py
    audits/
      adjoint.py
      kernel.py
      monotonicity.py
      rank.py
      subspaces.py
      tangent.py
      secant.py
      leakage.py
    inverse/
      base.py
      tsvd.py
      ridge.py
      smoothness.py
      tv.py
      state_space.py
      uncertainty.py
    ml/
      device.py
      autoencoder.py
      latent_inverse.py
      neural_field.py
      training.py
    metrics/
      reconstruction.py
      identifiability.py
      prior_dominance.py
      runtime.py
    io/
      hdf5.py
      tables.py
      manifests.py
  scripts/
    reproduce_v01.py
    build_raymaps.py
    run_e1_factorial.py
    run_e2_mode_atlas.py
    run_e3_raymap_validation.py
    run_e4_mechanism.py
    run_e5_leakage.py
    run_e6_geometry_mismatch.py
    run_e7_linear_reconstruction.py
    train_e8_autoencoder.py
    run_e8_tangent_secant.py
    run_e9_prior_audit.py
    run_e10_instrument.py
    build_figures.py
    build_report.py
  tests/
    test_dense_operator_parity.py
    test_adjoint.py
    test_mixing_identity.py
    test_kernel_injection.py
    test_gram_monotonicity.py
    test_source_splits.py
    test_manifest.py
  artifacts/
    manifests/
    gates/
    raymaps/
    tables/
    figures/
    reports/
    logs/
  notebooks/
    exploratory_only/
```

Notebooks are not authoritative. Every notebook operation needed by the paper must also exist in `src/` or `scripts/`.

---

## 3. Native Mac environment

### 3.1 One-time system setup

```bash
xcode-select --install
```

Use Miniforge/Mambaforge or another native arm64 conda-compatible manager on Apple Silicon. Intel Macs use the x86_64 variant.

### 3.2 Physics environment

```bash
conda env create -f environments/physics.yml
conda activate phrt-physics
python -m pytest tests/test_dense_operator_parity.py -q
```

The physics environment owns:

- AART;
- NumPy/SciPy float64 operators;
- HDF5 ray maps;
- singular/eigenvalue audits;
- source generation;
- linear reconstruction;
- figure generation.

### 3.3 ML environment

```bash
conda env create -f environments/ml.yml
conda activate phrt-ml
python - <<'PY'
import torch
print('torch', torch.__version__)
print('mps built', torch.backends.mps.is_built())
print('mps available', torch.backends.mps.is_available())
PY
```

Device policy:

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

Do not require MPS. Intel Macs and unsupported operations must remain valid on CPU.

### 3.4 Precision policy

- ray maps and operator audits: NumPy float64;
- Gram/eigen/SVD calculations: CPU float64;
- neural weights and batches: torch float32;
- frozen neural outputs are copied to CPU float64 before scientific singular-spectrum calculations;
- do not attempt float64 tensors on MPS.

### 3.5 Optional packages

`eht-imaging`, `ngehtsim`, and `finufft` are optional E10 dependencies. Do not install or debug them during E0–E6 unless the core environment already passes.

---

## 4. Core data model

### 4.1 `RayMap`

Implement an immutable dataclass with fields:

```python
@dataclass(frozen=True)
class RayMap:
    geometry_id: str
    order: int
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    source_r: NDArray[np.float64]
    source_phi: NDArray[np.float64]
    delay: NDArray[np.float64]
    redshift: NDArray[np.float64]
    transfer_weight: NDArray[np.float64]
    pixel_area: NDArray[np.float64]
    valid: NDArray[np.bool_]
    metadata: dict[str, Any]
```

The adapter must record the coordinate conventions and whether `delay` is absolute, relative, or sign-flipped from the backend. Convert all maps to one repository convention before storage.

### 4.2 HDF5 schema

Each file contains:

```text
/meta/
  geometry_id
  spin
  inclination_deg
  profile
  backend
  backend_version
  code_commit
  convention_json
/screen/
  alpha
  beta
/orders/n0/
  source_r
  source_phi
  delay
  redshift
  transfer_weight
  pixel_area
  valid
/orders/n1/...
/orders/n2/...
```

Attach SHA-256 and input config hash in HDF5 attributes.

### 4.3 `SourceBasis`

Interface:

```python
class SourceBasis(Protocol):
    dimension: int
    def evaluate(self, r, phi, t) -> NDArray[np.float64]: ...
    def coefficients_to_values(self, x, r, phi, t) -> NDArray[np.float64]: ...
    def accumulate_adjoint(self, residual, r, phi, t, weight) -> NDArray[np.float64]: ...
    def labels(self) -> list[ModeLabel]: ...
```

The registered 224D basis is 4 radial B-splines × 7 real azimuthal Fourier modes × 8 temporal DCT modes. Orthonormalize numerically under the declared source quadrature and store the Gram correction.

### 4.4 `HistoricalOperator`

Subclass `scipy.sparse.linalg.LinearOperator`.

`matvec(x)`:

1. for each observer time and order, compute emission time `t_obs - delay`;
2. evaluate source basis at `(source_r, source_phi, emission_time)`;
3. multiply by redshift/transfer/pixel weights;
4. aggregate into the declared image or ray samples;
5. return resolved stack or feed through `OrderMixer`.

`rmatvec(y)`:

1. unmix/route residuals to order blocks as required by the linear readout;
2. multiply by identical forward weights;
3. accumulate basis adjoint at the corresponding source coordinates and retarded times;
4. sum over rays and observer times.

The `rmatvec` implementation must not be generated by explicitly materializing the full matrix in core runs.

### 4.5 Whitening

Represent the scientific operator as `B = W @ A`, where `W` applies `C^{-1/2}`. All singular values and Fisher metrics use `B`, not the unwhitened operator.

---

## 5. Implementation phases

## Task 0 — Initialize repository and provenance

### Actions

- create repository layout;
- add environment files;
- implement `phrt provenance` command;
- record macOS version, architecture, RAM, CPU/GPU, Python, package versions, git commit, config hash;
- add deterministic seed manager;
- add Parquet/HDF5 helpers;
- add manifest and SHA-256 generation;
- import the original v0.1 generator under `archive/` without editing it.

### Required output

`artifacts/reports/TASK0_ENVIRONMENT.md`

### Stop rule

Do not proceed if the environment cannot replay a simple NumPy/SciPy float64 SVD and write/read HDF5 exactly.

---

## Task 1 — E0 reproduction

### Actions

- run original generator;
- implement independent toy operator;
- compare ranks, singular values, and reconstruction errors;
- implement the first five unit tests;
- produce a machine-readable gate file.

### Acceptance

All E0 gates pass exactly as defined in the protocol.

### Required report

`artifacts/reports/P1_E0_REPRODUCTION.md`

The report must include a side-by-side table with original, reimplementation, absolute error, relative error, and pass/fail.

---

## Task 2 — Matrix-free operator library

### Actions

- implement basis and dense reference;
- implement `HistoricalOperator` and adjoint;
- implement resolved/unresolved/partial order mixers;
- implement row whitening;
- implement explicit dense export only for smoke sizes;
- implement rank/SVD utilities that accept `LinearOperator`;
- compute Gram matrices by streaming when source dimension is modest.

### Required tests

1. dense vs matrix-free forward;
2. dense vs matrix-free adjoint;
3. inner-product adjoint test over 20 seeds;
4. resolved-to-unresolved mixing identity;
5. Gram monotonicity;
6. exact null-vector injection on constructed rank-deficient operators.

### Engineering note

For \(d\le 600\), it is usually cheaper and more stable to stream columns or accumulate \(G=A^\mathsf TC^{-1}A\) and diagonalize the \(d\times d\) Gram matrix than to store all rows. Preserve the ability to recover right singular vectors from the eigenvectors of \(G\).

---

## Task 3 — E1 structured factorial and E2 atlas

### Actions

- implement rotation/shear/delay structured toy operators;
- run all registered factors and seeds;
- compute rank and stability metrics;
- compute null and near-null modes;
- render mode atlas;
- create matched Gaussian, duplicate, shuffled-delay, and zero-amplitude controls.

### Required report

`artifacts/reports/P1_E1_E2_STRUCTURED_OPERATOR.md`

### Report ruling

Classify each source class as structural gain, conditioning-only, oracle-only, or no gain. Do not use black-hole language yet.

---

## Task 4 — AART adapter and E3 physical validation

### Actions

1. install/import AART in the physics environment;
2. reproduce one official example without modifying backend code;
3. identify the exact arrays corresponding to lensing band, source radius, azimuth, and emission time;
4. document all conventions;
5. implement conversion into `RayMap`;
6. generate smoke maps for one geometry/order;
7. generate pilot maps at multiple resolutions;
8. implement stratified ray sampling and quadrature weights;
9. add `kgeo` or independent geodesic cross-check on selected rays;
10. freeze tolerances in `PILOT_FREEZE.json`;
11. generate the 12 core geometries, orders 0–2.

### Do not

- modify AART equations to force agreement;
- drop disagreeing rays without a preregistered validity rule;
- compare coordinate arrays before harmonizing conventions;
- treat adaptive-grid point count as uniform pixel area.

### Required report

`artifacts/reports/P1_E3_RAYMAP_VALIDATION.md`

Include convergence curves, cross-code residual histograms, rejected-ray counts, and cache hashes.

---

## Task 5 — Build physical historical operators

### Actions

- implement source basis evaluation on AART emission coordinates;
- build matrix-free order-specific operators;
- validate on a small dense export;
- run adjoint and mixing gates for every geometry;
- compute \(G_n\), cumulative \(G_N\), and restricted spectra;
- produce direct/resolved/unresolved results before any ML.

### Required report

`artifacts/reports/P1_PHYSICAL_OPERATOR_AUDIT.md`

The first page must state mechanically whether any physical geometry shows strict restricted-rank gain.

---

## Task 6 — E4 mechanism decomposition

### Actions

Implement FULL, DELAY-ONLY, SPATIAL-ONLY, NO-ROTATION, WEIGHT-EQUALIZED, DELAY-SHUFFLED, and DUPLICATE arms exactly as registered. Each arm must share source basis, row count, noise definition, and source normalization.

### Required report

`artifacts/reports/P1_E4_MECHANISM.md`

State which factor supplies each new mode. If the result depends on weight equalization, call it structurally visible but physically weak.

---

## Task 7 — E5 leakage and E6 geometry mismatch

### Actions

- implement `L_epsilon`;
- implement two-channel grouping and high-frequency proxy;
- run known and miscalibrated leakage;
- build true/assumed geometry operator pairs;
- compute principal angles and reconstruction bias;
- preserve paired seeds and sources.

### Required reports

- `P1_E5_ORDER_LEAKAGE.md`
- `P1_E6_GEOMETRY_MISMATCH.md`

Stop before reconstruction if the operator itself fails the registered robustness category.

---

## Task 8 — E7 linear/state-space reconstruction

### Actions

- implement TSVD, ridge, smoothness Tikhonov, TV/nonnegative, and state-space baselines;
- implement validation-based tuning;
- generate ID and OOD sources;
- run paired readout/noise experiments;
- generate exact null pairs and near-null pairs;
- compute uncertainty for Gaussian baselines;
- report per-time and per-mode errors.

### Required report

`artifacts/reports/P1_E7_LINEAR_RECONSTRUCTION.md`

The report must distinguish full-space, correct restricted-class, and misspecified-prior results.

---

## Task 9 — E8 learned manifold

### Actions

- train compact autoencoder with five registered seeds;
- log device, seed, epoch, train/validation losses, and checkpoint hash;
- select checkpoint using validation only;
- compute tangent Jacobians at held-out points;
- compute secant ratios for held-out pairs;
- compare with PCA and random-decoder controls;
- evaluate OOD source reconstruction by the autoencoder itself before inverse use.

### MPS rule

Training can use MPS. Tangent operators and reported eigenvalues must be recomputed on CPU float64 from frozen decoder Jacobians.

### Required report

`artifacts/reports/P1_E8_LATENT_IDENTIFIABILITY.md`

Do not proceed to inverse claims if the decoder excludes most OOD variation or has poor direct source reconstruction.

---

## Task 10 — E9 ML inverse and prior audit

### Actions

- implement latent MAP optimization;
- optionally implement amortized inverse and coordinate neural field;
- use identical data for all priors;
- run prior swaps, exact null pairs, near-null pairs, and OOD sources;
- compute likelihood/prior curvature and coverage;
- assign data-support labels by temporal region.

### Required report

`artifacts/reports/P1_E9_PRIOR_DOMINANCE.md`

Every impressive reconstruction figure must be paired with a data-support map and a prior-swap comparison.

---

## Task 11 — Optional E10 instrument stress

### Actions

- implement direct complex visibility sampling first;
- validate visibility adjoint;
- repeat identifiability audit;
- only then install/use `eht-imaging` or `ngehtsim` for realistic schedules;
- keep nonlinear closure quantities in a separate experiment because they leave the paper’s linear operator setting.

### Required report

`artifacts/reports/P1_E10_INSTRUMENT.md`

---

## 6. Exact command interface

Implement a CLI with commands of this form:

```bash
phrt provenance --out artifacts/manifests/environment.json
phrt reproduce-v01 --config configs/smoke.yaml
phrt build-raymaps --config configs/core.yaml --geometry a090_i050
phrt validate-raymaps --config configs/core.yaml
phrt run-identifiability --experiment E4 --config configs/core.yaml
phrt run-reconstruction --experiment E7 --config configs/core.yaml
phrt train-autoencoder --config configs/core.yaml --seed 0
phrt run-latent-audit --config configs/core.yaml
phrt build-figures --registry configs/paper1_experiment_registry_v0.2.yaml
phrt build-report --all
```

Every command must:

- refuse a dirty tree in registered mode unless `--allow-dirty` is explicit;
- compute a run ID from timestamp plus config hash;
- write logs and manifest;
- support `--resume`;
- avoid overwriting artifacts unless `--force` is explicit;
- return nonzero exit status on failed correctness gates.

---

## 7. Rank and singular-spectrum implementation

### Small dense/reference cases

Use `scipy.linalg.svd` in float64.

### Physical matrix-free cases

Preferred order:

1. if source dimension \(d\le 600\), accumulate \(G=A^\mathsf TC^{-1}A\) in batches and use `scipy.linalg.eigh`;
2. otherwise use `scipy.sparse.linalg.svds` or a randomized range finder on the `LinearOperator`;
3. validate smallest singular values against a reduced dense instance;
4. sort all returned singular values explicitly because sparse solvers do not guarantee order.

### Numerical-null basis

Use eigenvectors of \(G\) under the registered relative threshold. Verify every candidate by direct `A.matvec(v)` before labeling it null.

### Principal angles

Use orthonormal bases and `scipy.linalg.subspace_angles`. Report dimensions and treatment of unequal nullities.

---

## 8. ML implementation guidance

### Autoencoder

- keep model under roughly one million parameters;
- use deterministic source coefficient normalization learned on train only;
- save normalizer with checkpoint;
- use full precision float32 initially; mixed precision is optional, not required;
- log MPS fallbacks if enabled;
- run one CPU parity batch for every checkpoint;
- no pretrained external model.

### Latent inverse

For data \(y\), solve

\[
\min_z
\frac12\|C^{-1/2}(A G(z)-y)\|_2^2
+\lambda R(z).
\]

Use multiple starts registered before test evaluation. Record all starts, not only the best successful one. Compare MAP data residual with the true-source residual and the linear baseline.

### Coordinate neural field

Use a compact MLP over \((r,\sin\phi,\cos\phi,t)\), Fourier features, and nonnegative output. It is an optional per-instance baseline, not the primary source-manifold theorem test.

### Do not use diffusion until

- operator gates pass;
- linear baselines are complete;
- latent audit is complete;
- there is evidence that a stronger prior would answer a scientific question rather than merely improve image appearance.

---

## 9. Result tables

Every identifiability row must include at least:

```text
run_id
config_hash
geometry_id
spin
inclination_deg
source_class
readout
max_order
leakage_level
noise_model
source_dimension
data_dimension
numerical_rank
operational_rank
nullity
sigma_max
sigma_min_positive
restricted_sigma_min
kappa_positive
stable_rank
effective_rank
trace_information
min_gram_eigenvalue
runtime_seconds
peak_rss_mb
```

Every reconstruction row must add:

```text
source_seed
noise_seed
method
hyperparameter_rule
nrmse
nrmse_by_age_json
mode_correlation
spectral_error
data_residual
visible_component_error
null_component_error
coverage_90
prior_data_fraction
prior_swap_distance
status
```

---

## 10. Agent report format

At the end of every task, write:

```text
# TASK <ID> — <TITLE>

## Identity
- branch
- commit
- config hash
- environment hash
- hardware

## Mechanical gate result
PASS / FAIL / STOP

## Inputs
...

## Results
Exact tables, not adjectives.

## Diagnostics
...

## Deviations
Any deviation from protocol; otherwise NONE.

## Claim effect
What the result permits, demotes, or forbids.

## Artifacts
Paths and SHA-256.

## Next authorized step
Exactly one phase, or STOP.
```

Do not say “promising,” “strong,” or “breakthrough” without the registered quantitative comparison immediately beside it.

---

## 11. Final manuscript ingestion rule

The coding agent does not rewrite the paper opportunistically after each run. It produces canonical tables and a final ruling with one of:

- `PHYSICAL_STRUCTURAL_GAIN_SUPPORTED`;
- `PHYSICAL_CONDITIONING_GAIN_ONLY`;
- `ORACLE_ORDER_RESOLUTION_ONLY`;
- `PRIOR_DOMINATED_RECOVERY`;
- `NO_USEFUL_HISTORICAL_GAIN`;
- `INSTRUMENT_LIMITED_NEGATIVE_RESULT`.

Only after the final ruling is frozen should the manuscript be updated. Preserve all negative experiments and failed gates in supplementary artifacts.
