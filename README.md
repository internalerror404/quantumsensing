# Computational and ML Experiment: Quantum-Statistical Null-Channel Design

This package implements the experiment attached to the paper **Quantum Fisher Information for Lorentzian Light-Ray Tomography**. The experiment has two distinct purposes:

1. **Theorem verification.** Numerically verify the algebraic and analytic identities used in the paper without treating numerical agreement as a proof.
2. **Learned experimental design.** Test whether an amortized selector can choose finite null-ray networks that preserve rank and improve the worst visible metric direction under changing packet bandwidths, priors, and angular constraints.

The inverse solver is deliberately linear and transparent. The ML component chooses the measurement network; it does not replace the forward model or hide reconstruction errors inside a black-box decoder.

## 1. Forward model

The background is Minkowski spacetime with signature `(-,+,+,+)`. For a null ray `gamma` with tangent `k=(1,theta)`, the linearized delay functional is

```text
A_gamma(h) = 1/2 * integral_gamma h_mu_nu k^mu k^nu d lambda.
```

A transform-limited Gaussian one-photon temporal mode with RMS width `s` induces QFI weight

```text
q_gamma = 1 / s^2.
```

For a finite metric family `h_theta = sum_i theta_i e_i`, the sensitivity matrix is

```text
J[a,i] = A_{gamma_a}(e_i),
F = J^T diag(q) J.
```

All rank and design metrics are computed after declaring a parameter-space normalization or prior precision. Raw eigenvalues are not compared across arbitrary parameter units.

> **Project state (2026-08-18).** This README describes the original v0.2 design. The canonical
> live record is `STATUS.md`; the executed review is `REVIEW.md`. Deltas against the text below:
> Stage 0 now comprises **seven scientific/numerical checks plus one stop-on-failure process
> control** (gate 3 integrates the gauge direction on every ray; gate 5 computes Var(nu) from
> the packet; gate 6 uses a non-polynomial profile; gate 8 certifies the gauge quotient at
> d in {6,8,12,16}). The **§6 static redshift experiment is implemented and passing**
> (`experiments/run_static_redshift.sh`, report in `results/static_redshift_experiment.json`).
> The deterministic scaling study has run on the registered seeds (`run_scaling_study.sh`).
> The **DeepSets v0.2 selector is frozen as a dominated negative baseline** — it beats random
> and angular-spread but is strictly Pareto-dominated by greedy D-optimal selection (23x
> objective at 0.37 ms vs 16.6 ms); the large learned-selector campaign will not be run for
> this architecture, and stronger forward-only gates for any v0.3 selector are registered in
> `configs/experiment.yaml`.

## 2. Stage 0: pinned theorem gates

Run:

```bash
cd experiments
./run_task1.sh
```

The script stops rather than modifying tolerances if any gate fails.

1. **Conformal directions:** `abs(A_Gamma(phi eta)) < 1e-14`. This is pointwise null contraction and should be at machine precision.
2. **Compact-support gauge directions:** `abs(A_Gamma(2 d^s V)) < 1e-10`. This is a quadrature-limited endpoint cancellation.
3. **Kernel persistence:** gauge modes remain in the kernel after every ray addition.
4. **Rank identity:** `rank(F) = rank(J)`. The report states the SVD tolerance and uses the matched eigenvalue threshold for `F=B^T B`, including a floating-point roundoff floor.
5. **Packet-width scaling:** the log-log QFI slope versus `s` is `-2.00 +/- 0.01`.
6. **Endpoint failure mode:** when `V` is nonzero at an endpoint, `A_gamma(2 d^s V)=[g(V,k)]_boundary` holds to relative error `1e-8`.
7. **No silent patching:** any deviation from the paper's proved statements terminates the run and writes a failure report.

The current deterministic report is written to `results/task1_verification.json`.

## 3. Stage 1: deterministic finite-network design

Run:

```bash
cd experiments
./run_design_demo.sh
```

The benchmark uses a 12-dimensional gauge-quotiented family of localized metric modes:

- four Newtonian-like scalar modes;
- three gravitomagnetic `h_0i` modes;
- five spatial anisotropic or trace-free modes.

Candidate rays use Fibonacci-sphere directions and transverse offsets. Packet widths are heterogeneous. The benchmark compares:

- uniform random selection;
- angular farthest-point coverage;
- leverage-score selection;
- greedy D-optimal selection;
- relaxed E-optimal allocation, top-K rounding, and local swaps.

Reported metrics are:

- numerical rank;
- minimum eigenvalue of the normalized QFIM;
- log determinant;
- condition number;
- Bayesian linear-MAP reconstruction RMSE;
- design runtime.

The single provided run is an engineering sanity check, not a statistical paper result. The full paper evaluation must use the registered seed set and task distribution in `configs/experiment.yaml`.

## 4. Stage 2: amortized ML selector

The scientific ML question is:

> Can a learned task-to-design map choose a near-optimal finite null-ray network across changing metric families, QFI weights, priors, and blocked angular sectors, while preserving the exact gauge kernel?

### Inputs

For each candidate ray `i`, the selector receives:

- the gauge-quotient sensitivity vector `a_i`;
- `log(q_i)`, where `q_i=1/s_i^2` for the independent Gaussian probe;
- ray direction `theta_i`;
- transverse offset `z_i`;
- optionally, prior-whitened features and availability masks.

### Architecture

The starter implementation uses a DeepSets scorer:

1. shared MLP embedding for each candidate;
2. permutation-invariant global mean context;
3. shared score head;
4. soft K-allocation during training;
5. hard top-K distinct rays during evaluation.

A Set Transformer or graph policy is an allowed later ablation, not the default.

### Training task distribution

Each task randomizes:

- orthogonal mixing and log scaling of the physical basis;
- packet widths and QFI weights;
- blocked angular sectors up to 20%;
- ray-pool density;
- dimension `d` and budget ratio `K/d`;
- prior precision or parameter metric.

The full registration is in `configs/experiment.yaml`.

### Objectives

Train separate policies for:

- **D-optimality:** maximize `log det(F_tilde + epsilon I)`;
- **E-optimality:** maximize the minimum eigenvalue of the whitened QFIM;
- optional **A-optimality:** minimize `trace((F_tilde + Lambda)^-1)`.

Here `F_tilde=R^-1/2 F R^-1/2`, with `R` explicitly declared. The paper should not report a "visibility spectrum" without this normalization.

### Evaluation baselines

- random selection;
- angular spread;
- leverage scores;
- greedy D-optimal selection;
- relaxed E-optimal optimization plus rounding;
- per-instance trainable-logit optimizer as an oracle-time baseline.

### Primary ML gates

The following are preregistered; failure is a result, not a reason to change the target after the run:

1. At `K >= d`, full-rank success is at least 99% on held-out tasks whenever the candidate pool admits rank `d`.
2. Median minimum QFIM eigenvalue is at least 15% above uniform and angular-spread baselines, with a 95% bootstrap confidence interval excluding zero.
3. The policy is within 5% of the per-instance relaxed or greedy oracle on its registered design objective, or the gap is reported.
4. Amortized design inference is at least 20 times faster than the per-instance optimizer on the same hardware target.
5. No ablation produces gauge or conformal leakage.

### Reconstruction protocol

For selected rays `S`, simulate

```text
y = J_S theta + epsilon,
epsilon ~ N(0, Sigma_S).
```

Use the Bayesian linear-MAP estimator

```text
theta_hat = (J_S^T Sigma_S^-1 J_S + Lambda)^-1 J_S^T Sigma_S^-1 y.
```

A neural reconstructor is intentionally excluded from Paper 1. In a linear Gaussian model, it would confound the quality of measurement design with approximation capacity and could not improve on the correctly specified MAP estimator.

## 5. Required ablations

- no gauge quotient;
- uniform rather than QFI weights;
- D- versus E-optimal training;
- homogeneous versus heterogeneous packet width;
- independent versus correlated pure probes;
- delay-only versus delay plus static endpoint-redshift channels;
- candidate-ray density;
- `K/d` budget ratio;
- quadrature resolution and integration domain.

The no-gauge-quotient ablation should become singular. It is a negative control, not a baseline expected to work.

## 6. Static redshift rank-restoration experiment

Construct conformal modes `h_j=phi_j eta` that are invisible to every delay ray. Add static emitter-receiver pairs with linearized log-redshift rows

```text
R_lj = 1/2 * (phi_j(A_l) - phi_j(B_l)).
```

Verify:

- delay-only conformal rank is zero;
- combined rank equals `rank(R)`;
- adding independent endpoint pairs lifts conformal rank exactly as predicted;
- the result is unchanged under allowed static gauge transformations preserving the operational clock frame.

This is the numerical counterpart of the paper's clock-assisted conformal rank-restoration corollary.

## 7. PTA/LISA application gate

PTA and interferometric links are application-layer anchors, not evidence for the spacelike microlocal theorem. Their perturbations are on-shell, noncompact, lightlike-frequency objects with endpoint terms.

The staged gate is:

1. validate the deterministic one-way link response;
2. add pulsar and Earth terms explicitly;
3. sample an isotropic stochastic gravitational-wave ensemble;
4. recover the Hellings-Downs correlation with less than 1% RMS error.

Only after that gate should the paper claim a quantitative detector connection.

## 8. Commands

```bash
# Environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Theorem gates
./run_task1.sh

# Deterministic design sanity check
./run_design_demo.sh

# Short engineering-only selector check
./run_selector_sanity.sh

# Selector held-out evaluation vs deterministic baselines
./run_selector_eval.sh

# Static redshift rank restoration (section 6)
./run_static_redshift.sh

# Deterministic design scaling study (registered seeds)
./run_scaling_study.sh
```

The short selector run verifies code paths only. The paper result requires the registered multi-seed train/validation/test campaign, bootstrap intervals, and saved task manifests.
