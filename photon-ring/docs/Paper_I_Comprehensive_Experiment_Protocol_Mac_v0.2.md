# Paper I Comprehensive Experiment Protocol — Mac-Only Campaign v0.2

## Photon-Ring Retarded-Time Tomography I
### Null Spaces, Identifiability, and Stability of Historical Inversion from Near-Critical Null Geodesics

**Status:** preregistration and implementation specification.  
**Execution target:** one macOS machine; CPU-first numerical physics; optional Apple-Silicon MPS acceleration for compact neural models.  
**Evidence boundary:** theory plus controlled synthetic computation. No telescope detection, laboratory result, or recovery from a resolved real photon ring is claimed.

---

## 1. Scientific objective

The paper asks whether direct and higher-order black-hole lensing channels contain enough independent information to reconstruct a bounded exterior source history. The central operator is

\[
\mathcal A_N j=(\mathcal T_0j,\ldots,\mathcal T_Nj),
\]

with order-specific retarded transfer operators \(\mathcal T_n\). The corresponding unresolved observation is

\[
\mathcal U_Nj=\sum_{n=0}^{N}M_n\mathcal T_nj.
\]

The v0.1 manuscript already establishes the abstract laws that

\[
\ker \mathcal A_N=\bigcap_{n=0}^{N}\ker \mathcal T_n,
\]

and that the whitened information operator grows monotonically under separately measured orders. Its current numerical result is a deliberately small linear toy: one random seed, one hand-built source subspace, and oracle-tuned regularization. That experiment demonstrates the distinction among rank, restricted injectivity, and prior-driven reconstruction, but it is not yet a physical Kerr forecast or a publication-grade empirical campaign.

The v0.2 campaign must answer four claim-critical questions:

1. **Physical strictness:** Do actual Kerr order-resolved transfer maps remove restricted null directions that remain invisible in direct or unresolved data?
2. **Mechanism:** Is any gain caused by retarded-time diversity, source-plane spatial diversity, or their interaction?
3. **Robustness:** Does the gain survive partial order separation, finite resolution, noise, and modest geometry error?
4. **ML honesty:** When a learned model reconstructs a deep historical component, can we certify that the likelihood—not merely the prior—supports it?

The primary target is **controlled temporal-mode recovery with known observability labels**, not visual quality.

---

## 2. Claim hierarchy

Results must be classified using the following hierarchy. A lower category must not be described using language from a higher category.

### C0 — Forward-physics validation
The Kerr ray maps, delays, weights, and order labels are numerically converged and independently cross-checked.

### C1 — Structural identifiability gain
For a declared source class \(\mathcal C\),

\[
\ker(\mathcal A_N|_{\mathcal C})
\subsetneq
\ker(\mathcal A_0|_{\mathcal C})
\]

or, for a linear basis \(Q_{\mathcal C}\),

\[
\operatorname{rank}(A^{(N)}Q_{\mathcal C})
>
\operatorname{rank}(A_0Q_{\mathcal C}).
\]

### C2 — Conditioning gain without rank gain
The restricted rank is unchanged, but the restricted lower information bound rises materially:

\[
\sigma_{\min}^{+}(C^{-1/2}A^{(N)}Q_{\mathcal C})
>
\sigma_{\min}^{+}(C^{-1/2}A_0Q_{\mathcal C}).
\]

### C3 — Oracle-only gain
A benefit exists only with perfect order labels and disappears under modest registered order leakage or instrument projection.

### C4 — Prior-supported prediction
A reconstruction is accurate under a particular source prior, but likelihood curvature, null-pair tests, or prior swaps show that the relevant historical direction is not independently identified by the data.

### C5 — No useful gain
The physical higher orders are redundant or too ill-conditioned for the declared source class and observation model.

A negative C3, C4, or C5 result is scientifically valid. It must not be repaired by narrowing the test set after inspection.

---

## 3. Mac-only design principles

1. **Never form a giant movie-to-data matrix by default.** Implement all physical operators as `scipy.sparse.linalg.LinearOperator` objects with exact `matvec` and `rmatvec`.
2. **Use float64 on CPU for ray-map validation, Gram matrices, SVD/eigendecomposition, and correctness gates.** Use float32 on MPS only for neural training and inference.
3. **Cache geometry.** Ray maps are expensive but source-independent. Trace each geometry once, store it in HDF5, and reuse it across source classes and experiments.
4. **Use dimensionless units.** Express distances and times in \(M=GM/c^2\) and \(GM/c^3\) until a final illustrative conversion.
5. **Separate physics and ML environments.** AART and numerical physics must remain usable even if PyTorch or optional imaging packages change dependencies.
6. **Do not begin with diffusion models.** The paper lives or dies on operator audits. Compact linear, state-space, latent-decoder, and coordinate-MLP experiments are sufficient for the first publishable campaign.
7. **No Docker as the primary path.** Native arm64 environments preserve Apple-Silicon MPS access. Intel Macs remain supported through CPU execution.
8. **Every expensive sweep is resumable.** One geometry/order/source-class cell must be independently cached and replayable.
9. **All figures derive from canonical Parquet/CSV results.** Notebooks may explore but cannot be the sole source of a reported number.
10. **Do not treat pixels, rays, or source samples as independent replicates.** Geometry and source realization are the primary statistical units.

---

## 4. Core mathematical objects and software interfaces

For geometry \(g\), order \(n\), and source coefficient vector \(x\in\mathbb R^d\), define

\[
y_{g,n}=A_{g,n}x+\eta_{g,n}.
\]

The order-resolved operator is

\[
A^{R}_{g,N}=
\begin{bmatrix}
A_{g,0}\\
\vdots\\
A_{g,N}
\end{bmatrix}.
\]

A partially resolved readout is

\[
A^{L}_{g,N}=(L\otimes I)A^{R}_{g,N},
\]

where \(L\) is an order-leakage or pseudo-channel mixing matrix. An unresolved readout is the one-channel collapse

\[
A^{U}_{g,N}=\sum_{n=0}^{N}M_nA_{g,n}.
\]

Whitened operators are

\[
B=C^{-1/2}A,
\qquad
G=B^\mathsf TB.
\]

For a linear source class with orthonormal basis \(Q\), use

\[
B_{\mathcal C}=BQ.
\]

For a differentiable decoder \(G_\theta(z)\), use the local latent operator

\[
B_{\mathrm{lat}}(z)=B J_G(z).
\]

### Required software abstractions

- `RayMap`: screen coordinates, order label, source radius, source azimuth, retarded delay, redshift, transfer weight, pixel area, validity mask, and provenance.
- `SourceBasis`: evaluates basis functions and their adjoint accumulation at arbitrary \((r,\phi,t)\).
- `HistoricalOperator(LinearOperator)`: matrix-free forward and adjoint maps.
- `OrderMixer`: ideal, partial, and unresolved order readouts.
- `VisibilityOperator`: optional image-to-complex-visibility map.
- `NoiseModel`: row whitening, sampling, and log likelihood.
- `SourceGenerator`: controlled source histories with reproducible latent truth.
- `IdentifiabilityAudit`: ranks, singular spectra, null vectors, principal angles, tangent and secant diagnostics.
- `Reconstructor`: linear, state-space, latent, and neural-field estimators behind one interface.

---

## 5. Registered execution profiles

The core profile is designed to remain practical on a modest Mac because the source dimension is small and the physical operator is matrix-free.

| Profile | Valid rays/order | Observer times | Linear source dimension | Geometries | ML latent dim | Role |
|---|---:|---:|---:|---:|---:|---|
| `smoke` | 256 | 8 | 36 | 1 | 4 | unit tests only |
| `core` | 1,536 | 24 | 224 | 12 | 16 | registered main campaign |
| `stress` | 4,096 | 32 | 480 | 4 | 24 | scaling and resolution stress |
| `refine` | 8,192 | 24 | 224 | 2 | none | ray-map convergence only |

The coding agent may lower batch size automatically, but it must not silently lower source dimension, ray count, number of geometries, or test-set size in a registered run. Any profile change creates a new config hash.

### Core linear source basis

Use an equatorial thin-emitter basis with:

- 4 radial cubic B-spline modes;
- 7 real azimuthal Fourier modes \((m=0,1,2,3\), represented by one constant plus sine/cosine pairs);
- 8 temporal DCT modes.

Thus

\[
d=4\times7\times8=224.
\]

This basis is the primary exact restricted class because its singular spectrum can be computed and its modes have interpretable radial, angular, and temporal labels.

### Core Kerr geometry grid

Use the Cartesian grid

\[
a_\star\in\{0.0,0.5,0.9,0.98\},
\qquad
i\in\{20^\circ,50^\circ,75^\circ\}.
\]

All main results aggregate across these 12 geometries. Two separate non-grid geometries may be used for implementation pilots; they are excluded from the main analysis.

### Image orders

The registered physical campaign uses

\[
n\in\{0,1,2\},
\]

matching the order range directly supported by the standard AART lensing-band implementation. Claims about asymptotically large \(n\) belong to Paper II unless a separate validated extension is added.

---

## 6. Governance, preregistration, and data splits

### 6.1 Immutable registries

Before the main run, freeze:

- geometry grid;
- source families and parameter ranges;
- train/validation/test seeds;
- readout definitions;
- noise and leakage levels;
- numerical rank rules;
- primary metrics;
- baseline hyperparameter-selection rules;
- outcome categories;
- plotting scripts.

The frozen file is `configs/paper1_experiment_registry_v0.2.yaml`. Its SHA-256 hash must appear in every result manifest.

### 6.2 Pilot firewall

Only the two named pilot geometries may be used to:

- debug AART conventions;
- choose numerical tolerances after convergence evidence;
- set optimizer learning rates;
- select a compact neural architecture;
- freeze any effect-size threshold used for headline language.

No main-grid result may be inspected until `PILOT_FREEZE.json` is committed.

### 6.3 Source splits

Use source-seed namespaces rather than random slicing after generation.

- `train`: seeds 100000–109999;
- `validation`: seeds 200000–201999;
- `test_id`: seeds 300000–303999;
- `test_ood`: seeds 400000–403999;
- `null_pair`: seeds 500000–500999.

The exact number consumed depends on the profile, but namespaces must never overlap.

### 6.4 No oracle tuning in main results

The current toy experiment uses ground-truth-tuned regularization and labels that correctly as an oracle diagnostic. In the main campaign:

- ridge/Tikhonov parameters are selected on validation sources or by a registered discrepancy/GCV rule;
- neural early stopping uses validation likelihood and source loss only;
- test truths are used only for final scoring;
- an oracle curve may be included as an explicitly optimistic upper bound, never as the deployable result.

---

# 7. Detailed experiment program

## P1-E0 — Reproduce and independently audit the v0.1 toy experiment

### Purpose
Establish a known-good baseline and ensure the new repository reproduces the manuscript’s existing finite-dimensional results before any physical extension.

### Required actions

1. Run the supplied v0.1 generator unchanged.
2. Reimplement the toy operator independently using the new `HistoricalOperator` interface.
3. Reproduce the registered dimensions:
   - \(H=44\), \(K=6\), \(W=24\), \(M=2\), \(N_{\max}=5\), \(D=4\), \(\Gamma=0.6\);
   - 24-dimensional smooth restricted model;
   - seed 42.
4. Reproduce the three maximum-order rows:
   - resolved/identical;
   - resolved/diverse;
   - unresolved/diverse.
5. Reproduce the noise-free and noisy reconstruction diagnostics.
6. Demonstrate that the new deployable hyperparameter rule is worse than or equal to the oracle-tuned curve, as expected.

### Acceptance criteria

- all ranks match exactly;
- singular values and errors match within relative \(10^{-8}\) in float64;
- the original and independent operators agree on 20 random vectors within relative \(10^{-10}\);
- every output is regenerated from code rather than copied.

### Output

- `artifacts/e0_reproduction/e0_metrics.json`
- `artifacts/e0_reproduction/e0_comparison.parquet`
- `reports/P1_E0_REPRODUCTION.md`

Failure blocks all later phases.

---

## P1-E1 — Structured finite-dimensional factorial

### Purpose
Replace the single favorable random-projection example with a controlled factorial that separates delay diversity, spatial diversity, attenuation, and order collapse.

### Base construction

Use a structured discrete operator

\[
A_n=a_n P_n D_{\Delta_n},
\]

where \(D_{\Delta_n}\) samples a delayed source window and \(P_n\) is a spatial projection built from rotations, shear, and low-rank sampling rather than independent random matrices.

### Experimental factors

1. **Order label**: direct only, resolved, partial, unresolved.
2. **Delay structure**:
   - no delay diversity;
   - constant integer delay \(nD\);
   - perturbed delays \(nD+r_n\);
   - pixel-dependent delay spread.
3. **Spatial structure**:
   - identical across orders;
   - rotation only;
   - rotation plus shear;
   - independent matched-rank control.
4. **Attenuation**:
   - equalized row energy;
   - \(a_n=e^{-\Gamma n}\) with \(\Gamma\in\{0.3,0.6,0.9\}\).
5. **Source class**:
   - full coefficient space;
   - smooth separable basis;
   - localized temporal atoms;
   - orbit-constrained tangent basis.

Use at least 20 registered random seeds for any remaining random component.

### Primary questions

- Can delays alone remove source-class null directions?
- Does spatial rotation/shear remove directions that delay-only channels miss?
- Does unresolved summation create strict rank loss or only conditioning loss?
- Does attenuation convert a structural gain into an operationally useless near-null direction?

### Primary metrics

- numerical and operational rank;
- restricted nullity;
- full singular spectrum;
- \(\sigma_{\min}\), \(\kappa^+\), stable rank, effective rank;
- number and identity of modes first crossing the operational threshold at each order;
- Gram increment spectrum \(\Delta G_n=A_n^\mathsf TC_n^{-1}A_n\).

### Negative controls

- Gaussian random operator with identical dimensions and row norms;
- shuffled delay labels;
- duplicated order block;
- zero-amplitude higher order;
- full-rank leakage matrix versus rank-one collapse.

### Output

- `tables/e1_identifiability_factorial.parquet`
- `tables/e1_mode_onset.parquet`
- `figures/e1_spectrum_grid.*`
- `figures/e1_mechanism_ablation.*`
- `reports/P1_E1_STRUCTURED_FACTORIAL.md`

This experiment establishes the interpretation framework but cannot by itself support a black-hole-specific claim.

---

## P1-E2 — Null-space and near-null mode atlas

### Purpose
Make the abstract null space physically interpretable rather than reporting rank alone.

### Procedure

For each E1 arm and later each physical Kerr geometry:

1. compute exact numerical null vectors when present;
2. compute the 20 smallest nonzero right singular vectors;
3. reshape each vector into radial × azimuthal × temporal basis coefficients;
4. assign:
   - dominant temporal frequency;
   - temporal center of mass / historical age;
   - dominant azimuthal harmonic;
   - radial support;
   - source-class membership;
5. inject \(x_\pm=x_0\pm\alpha v\) for null and near-null vectors;
6. verify the predicted data indistinguishability or noise-level ambiguity;
7. track when each mode becomes visible as \(N\) increases.

### Principal-angle analysis

For visible and null subspaces \(V_N\), compute principal angles between:

- \(N\) and \(N+1\);
- resolved and unresolved arms;
- true and perturbed geometries;
- image-domain and visibility-domain instruments.

### Required figures

- atlas of representative exact null modes;
- atlas of weakest visible modes;
- mode-onset heatmap indexed by temporal and azimuthal frequency;
- principal-angle stability versus geometry perturbation;
- source perturbation/data residual pairs.

### Interpretation rule

A new order that increases rank only by revealing extremely oscillatory or extremely old modes with vanishing \(\sigma_i\) must be reported as structural but unstable—not as usable history recovery.

---

## P1-E3 — Physical Kerr ray-map generation and validation

### Purpose
Replace hand-built projections with actual order-resolved Kerr transfer maps.

### Physics backend

Use AART as the primary semi-analytic ray tracer for an equatorial, optically thin source. Cache separately for each geometry and order:

- screen coordinates \((\alpha,\beta)\);
- lensing-band/order label \(n\);
- source radius \(r_s\);
- source azimuth \(\phi_s\);
- coordinate emission time or delay \(\Delta t\);
- redshift factor;
- transfer/magnification weight used by the declared source model;
- pixel quadrature area and validity mask.

Use `kgeo` or an independent geodesic implementation to cross-check a stratified sample of rays after coordinate and sign conventions are harmonized.

### Ray sampling

For each order, stratify valid rays by:

- screen azimuth;
- distance from the critical curve / lensing-band boundary;
- source radius;
- delay quantile.

Do not take the first \(K\) rows of an adaptive grid. Sampling weights must preserve quadrature area.

### Convergence study

For the two `refine` geometries:

1. generate coarse, core, and refined ray maps;
2. compare ray-wise delay, source coordinates, redshift, and weighted operator predictions;
3. compare restricted singular spectra on the fixed 224-dimensional basis;
4. freeze core resolution only if headline metrics change by less than the registered tolerance.

### Validation gates

- all retained rays have finite coordinates and positive declared intensity weights;
- ray maps are deterministic under replay;
- direct and higher-order maps cover the intended lensing bands;
- sampled cross-tracer rays agree within the convention-aligned tolerance frozen after pilot;
- the operator response converges under ray-map refinement;
- no main result uses an invalid/masked ray.

### Output

- `raymaps/a{spin}_i{inc}_n{order}_{profile}.h5`
- `tables/e3_raymap_convergence.parquet`
- `tables/e3_cross_tracer.parquet`
- `reports/P1_E3_RAYMAP_VALIDATION.md`

A failed physical validation blocks all Kerr claims, but E1 can remain as a mathematical computation.

---

## P1-E4 — Kerr mechanism decomposition: delay versus spatial diversity

### Purpose
Determine why physical higher-order maps add information.

### Arms

For each geometry, build:

1. **FULL:** physical \(r_s,\phi_s,\Delta t,w\) for every order.
2. **DELAY-ONLY COUNTERFACTUAL:** retain physical order delays and order amplitudes but force all orders to use a matched common source-plane spatial sampling map.
3. **SPATIAL-ONLY COUNTERFACTUAL:** retain physical source-plane maps and weights but collapse order delays to a common reference.
4. **NO-ROTATION ALIGNMENT:** undo the estimated inter-order azimuthal rotation before evaluating the source basis.
5. **WEIGHT-EQUALIZED:** normalize order blocks to equal whitened operator norm, isolating geometry from attenuation.
6. **DELAY-SHUFFLED:** permute delays within each order while preserving their marginal distribution.
7. **DUPLICATE CONTROL:** replace every higher order by a scaled duplicate of order zero.

Counterfactual arms are mechanism probes and must never be described as physical observation forecasts.

### Primary endpoint

For each source class, decompose the resolved gain into:

- strict rank gain;
- increase in restricted \(\sigma_{\min}\);
- added trace information;
- changed null-vector morphology.

### Key inference

- gain in FULL and DELAY-ONLY implies temporal diversity is sufficient;
- gain in FULL and SPATIAL-ONLY implies spatial remapping is sufficient;
- gain only in FULL implies a genuine interaction;
- gain only after weight equalization implies the physical channel is structurally useful but practically suppressed.

### Output

- `tables/e4_mechanism_decomposition.parquet`
- `figures/e4_rank_venn_or_upset.*`
- `figures/e4_sigma_min_ablation.*`
- `reports/P1_E4_MECHANISM.md`

---

## P1-E5 — Partial order resolution and leakage phase diagram

### Purpose
Bridge the ideal resolved theorem and realistic observations in which order labels are inferred imperfectly.

### Leakage model

For three orders, define

\[
L_\epsilon=(1-\epsilon)I+\epsilon\frac{\mathbf1\mathbf1^\mathsf T}{3},
\qquad
\epsilon\in\{0,0.02,0.05,0.10,0.20,0.40,0.70,1.0\}.
\]

At \(\epsilon=0\), the orders are perfectly resolved. At \(\epsilon=1\), all pseudo-channels are identical and the order axis is rank one.

Also test:

- two-channel grouping: \(n=0\) versus \(n\ge1\);
- direct image plus high-spatial-frequency proxy;
- known leakage matrix;
- miscalibrated leakage matrix with registered 1%, 5%, and 10% relative perturbations;
- order-dependent missingness.

### Metrics

- restricted rank and \(\sigma_{\min}\);
- condition number of the order mixer itself;
- reconstruction error;
- calibration bias;
- number of modes whose data support falls below threshold;
- prior-swap sensitivity.

### Outcome labels

- `ROBUST_RESOLUTION_GAIN`: benefit persists through at least the registered modest-leakage level;
- `FRAGILE_RESOLUTION_GAIN`: benefit collapses under modest leakage;
- `ORACLE_ONLY_GAIN`: benefit exists only at \(\epsilon=0\);
- `NO_RESOLUTION_GAIN`: no structural or conditioning benefit.

The threshold defining “modest leakage” must be frozen after pilot and before main-grid inspection.

---

## P1-E6 — Geometry mismatch and source–geometry degeneracy

### Purpose
Test whether a small error in black-hole geometry rotates the visible/null subspaces enough to mimic source history.

### True-versus-assumed operator pairs

Generate data with \(A(a_\star,i)\) and invert with:

- spin offsets \(\Delta a_\star\in\{\pm0.02,\pm0.05\}\);
- inclination offsets \(\Delta i\in\{\pm1^\circ,\pm3^\circ,\pm5^\circ\}\);
- delay-scale errors \(\{\pm0.5\%,\pm1\%,\pm2\%,\pm5\%\}\);
- source-plane position-angle errors \(\{\pm1^\circ,\pm3^\circ,\pm5^\circ\}\);
- transfer-weight exponent errors where applicable.

### Measurements

- operator prediction error;
- principal angles between visible and null subspaces;
- reconstruction bias by historical age;
- residual goodness of fit;
- inferred source change aligned with geometry derivatives;
- local joint Fisher matrix for source coefficients plus geometry parameters.

### Optional joint refinement

Implement an alternating or joint local optimizer over source coefficients and a small geometry parameter vector. This is a stress test, not required for the first paper claim.

### Kill condition

If the order-resolved gain disappears under geometry errors materially smaller than the expected calibration regime, the paper must describe the method as geometry-oracle theory rather than a robust inversion proposal.

---

## P1-E7 — Linear and state-space reconstruction campaign

### Purpose
Translate singular-spectrum conclusions into deployable reconstruction performance without ML ambiguity.

### Source families

1. **L1 smooth separable:** random coefficients in the registered 224-dimensional basis with controlled spectral decay.
2. **L2 orbiting hotspot:** one compact source with randomized radius, phase, width, angular speed, rise, and decay.
3. **L3 stochastic field:** low-rank Gaussian random field with randomized spatial and temporal correlation scales.
4. **L4 OOD:** two hotspots, spiral-like perturbation, abrupt flare, and variability spectra outside the training range.

### Readout arms

- direct only;
- ideal resolved \(n=0,1,2\);
- registered partially resolved;
- unresolved sum;
- resolved duplicate-order negative control;
- oracle full source samples as an optimistic ceiling.

### Noise levels

Use whitened leading-order effective SNR values

\[
\{\infty,300,100,30,10\},
\]

with independent registered noise seeds. Add one heteroscedastic arm in which weak orders have lower effective SNR.

### Baselines

1. truncated SVD;
2. zeroth-order ridge;
3. temporal second-difference Tikhonov;
4. nonnegative temporal-TV or spatial-TV solver;
5. Gaussian-process / linear state-space smoother;
6. source-class oracle projection as an explicitly optimistic ceiling.

### Hyperparameter selection

- validation grid or registered discrepancy principle;
- no per-test ground-truth tuning;
- all baseline budgets matched by number of data evaluations where relevant.

### Primary metrics

- coefficient and movie NRMSE;
- per-retarded-time NRMSE;
- temporal-mode recovery correlation;
- source power-spectrum error;
- hotspot trajectory/radius/phase error for L2;
- whitened data residual;
- recovered-versus-true null/visible component error;
- coverage for methods that report intervals;
- compute time and peak memory.

### Exact null-pair test

For every rank-deficient arm, construct

\[
x_\pm=x_0\pm\alpha v,
\qquad v\in\ker A.
\]

Confirm identical clean data, run every reconstructor, and report its unavoidable ambiguity. A method may select one member due to its prior, but that is not counted as data recovery.

### Near-null discrimination test

Use the smallest visible singular vectors to construct pairs whose Mahalanobis data distance spans \(0.25,0.5,1,2,4\). Measure pair-discrimination accuracy and compare with the theoretical Gaussian limit.

---

## P1-E8 — Learned source manifold and tangent/secant identifiability

### Purpose
Test the manuscript’s proposition that a generative prior is locally identifiable only when its tangent directions avoid the forward null space.

### Compact Mac-friendly model

Train an autoencoder over 224-dimensional physical basis coefficients.

Recommended starting architecture:

- encoder: 224 → 256 → 128 → 48 → latent 16;
- decoder: latent 16 → 48 → 128 → 256 → 224;
- GELU activations;
- no stochastic VAE term in the first registered model;
- source nonnegativity enforced only when the basis representation makes that meaningful;
- training on a mixture of L1–L3 source families;
- OOD L4 reserved entirely for testing.

Architecture changes after pilot require a new protocol version.

### Tangent audit

At each of at least 256 held-out latent points:

1. compute decoder Jacobian \(J_G(z)\);
2. compute \(BJ_G(z)\);
3. report tangent rank, \(\lambda_{\min}\), condition number, and weakest latent direction;
4. repeat across readout arms and geometries.

### Secant audit

Sample at least 20,000 held-out latent pairs and evaluate

\[
r_{ij}=
\frac{\|B(G(z_i)-G(z_j))\|_2}
{\|G(z_i)-G(z_j)\|_2}.
\]

Report lower-tail quantiles \(q\in\{0.1\%,1\%,5\%\}\), median, and minimum observed value. These are diagnostics, not proofs of the global infimum.

### Controls

- random decoder with matched output covariance;
- PCA basis with the same latent dimension;
- decoder trained only on hotspots;
- decoder trained on the mixed source family;
- order-label shuffle;
- source-seed leakage audit.

### Interpretation

A decoder can make \(BJ_G(z)\) full rank by excluding physical alternatives. That is conditional identifiability on the learned class, not evidence that the full historical source is observed. The paper must report both full-space and model-restricted results.

---

## P1-E9 — Learned reconstruction and prior-dominance audit

### Purpose
Determine when ML improves stability and when it merely chooses a plausible member of a data-equivalence class.

### ML estimators

1. latent optimization through the frozen decoder;
2. amortized latent regressor from data to \(z\);
3. compact coordinate neural field optimized per instance;
4. optional small plug-and-play denoiser after all simpler baselines pass.

A diffusion model is optional and cannot become the only ML result.

### Prior-swap protocol

For identical data, reconstruct using:

- temporal smoothness prior;
- Gaussian-process/state-space prior;
- learned decoder prior;
- intentionally mismatched hotspot-only prior.

Compute pairwise reconstruction disagreement, especially projected onto exact and near-null subspaces.

### Likelihood-versus-prior information

For linear Gaussian models, report exact

\[
\Sigma_{\rm post}=(K^{-1}+A^\mathsf TC^{-1}A)^{-1}.
\]

For latent models, use a registered Laplace approximation around the MAP:

\[
H_{\rm post}\approx
J_G^\mathsf TA^\mathsf TC^{-1}AJ_G+H_{\rm prior}.
\]

Report:

- likelihood curvature;
- prior curvature;
- posterior variance reduction attributable to the likelihood;
- per-time-bin data fraction;
- 90% interval coverage;
- prior-swap displacement;
- null-pair behavior.

### OOD requirement

The main test set must include source morphologies and variability spectra absent from training. A visually plausible but OOD-biased reconstruction is a failure for a general-history claim.

### Required claim labels

Every reconstructed temporal region receives one of:

- `DATA_IDENTIFIED`;
- `DATA_CONDITIONED_BUT_WEAK`;
- `PRIOR_DOMINATED`;
- `OUTSIDE_CAUSAL_OR_OBSERVATIONAL_FOOTPRINT`.

---

## P1-E10 — Instrument projection and VLBI stress test

### Purpose
Test whether image-domain identifiability survives a plausible interferometric measurement map.

### Execution order

Do not begin E10 until E0–E9 core gates pass.

### Measurement levels

1. ideal order-resolved image samples;
2. beam-convolved image movie;
3. complex visibilities at registered \((u,v,t)\) samples with known gains;
4. total light curve plus high-spatial-frequency visibility proxy;
5. sparse ground-like array;
6. optional ground-plus-space array;
7. optional nonlinear closure quantities, clearly separated from the linear theorem.

### Implementation

Use a direct nonuniform DFT or `finufft` for the core small problem. `eht-imaging` or `ngehtsim` may be added later for realistic schedules and noise, but installation friction must not block the mathematical campaign.

### Metrics

Repeat the complete identifiability and reconstruction audit after applying the instrument operator. The correct comparison is

\[
A_{\rm instrument}=F_{\rm instrument}A_{\rm image},
\]

not visual comparison of reconstructed images alone.

### Decisive negative outcome

If the restricted lower bound is essentially zero for all plausible instrument arms, the paper should conclude that the temporal archive is a mathematical property of order-resolved rays but not recoverable under the tested observation architecture.

---

# 8. Numerical rank, operational rank, and stability conventions

## 8.1 Numerical rank

For dense small matrices use

\[
\sigma_i>
\max(m,d)\,\epsilon_{64}\,\sigma_1.
\]

Also report sensitivity to thresholds \(10^{-8},10^{-10},10^{-12}\) relative to \(\sigma_1\). Do not hide threshold dependence.

## 8.2 Operational rank

Define a mode as operationally visible only if its expected unit-amplitude Mahalanobis response exceeds the registered detection threshold. Report operational rank separately from algebraic rank.

## 8.3 Condition number

For rank-deficient operators report

\[
\kappa^+=\sigma_{\max}/\sigma_{\min}^{+}
\]

on the identifiable support. Never report an ordinary finite condition number after silently dropping null modes.

## 8.4 Effective dimensions

Report:

- stable rank \(\|B\|_F^2/\|B\|_2^2\);
- spectral-entropy effective rank;
- exact nullity;
- operational nullity.

## 8.5 Source normalization

All comparisons across orders and geometries require a declared source norm and noise whitening. Row normalization that removes physical attenuation must be labeled as an ablation, not used in the primary physical result.

---

# 9. Correctness gates

These are mechanical requirements. A failed gate stops the dependent phase.

| Gate | Requirement | Tolerance / rule |
|---|---|---|
| G0 Environment | native environment, package manifest, hardware report | exact artifact present |
| G1 v0.1 reproduction | registered ranks/metrics reproduced | rank exact; floats rel. \(<10^{-8}\) |
| G2 Dense/operator parity | dense small matrix equals `LinearOperator` | rel. \(<10^{-10}\) |
| G3 Adjoint | \(\langle Ax,y\rangle=\langle x,A^*y\rangle\) | rel. \(<10^{-8}\) float64 |
| G4 Order collapse | mixer(resolved) equals unresolved | rel. \(<10^{-10}\) |
| G5 Kernel injection | numerical null vector changes source but not data | normalized residual \(<10^{-8}\) |
| G6 Information monotonicity | \(G_{N+1}-G_N\succeq0\) | min eig \(\ge-10^{-10}\max(1,\|G\|_2)\) |
| G7 Grid convergence | primary metrics stable under refinement | threshold frozen after pilot |
| G8 Cross-tracer | stratified ray checks agree after convention alignment | threshold frozen after pilot |
| G9 Source split | no train/validation/test seed or artifact overlap | exact hash audit |
| G10 No test tuning | hyperparameters selected without test truth | provenance audit |
| G11 CPU/MPS parity | frozen neural inference agrees | rel. \(<10^{-4}\) |
| G12 Coverage | uncertainty claim calibrated | 90% nominal within registered band |
| G13 Replay | deterministic stages reproduce hashes | exact; stochastic summaries within MC tolerance |

`correctness_gates.json` must contain inputs, measured value, threshold, pass/fail, and evidence path for every gate.

---

# 10. Statistical analysis

1. **Primary aggregation unit:** geometry. Source realizations and noise draws are nested within geometry.
2. **Intervals:** geometry-level bootstrap confidence intervals, with source-level bootstrap nested when feasible.
3. **Paired comparisons:** resolved versus unresolved must use identical geometry, source, noise, and sampling seeds.
4. **Headline effects:** median and lower-quartile gains across geometries; never the best geometry alone.
5. **Mode multiplicity:** when testing many singular modes, report false-discovery-controlled mode-level findings or treat the atlas descriptively.
6. **ML seeds:** at least five independent training seeds for any headline neural result; select no single favorable seed.
7. **Failure rate:** report optimizer failures and excluded runs. Exclusions require a preregistered mechanical reason.
8. **Resolution uncertainty:** include core-versus-refined variation as a numerical systematic.
9. **Geometry uncertainty:** treat E6 mismatch as a systematic, not only another random seed.
10. **No pseudo-replication:** thousands of rays do not turn one geometry into thousands of independent physical demonstrations.

---

# 11. Primary result tables and figures

## Required tables

1. `physical_identifiability_by_geometry`
   - rank, nullity, \(\sigma_{\min}\), \(\kappa^+\), stable rank for every readout/order/source class.
2. `mechanism_decomposition`
   - FULL, delay-only, spatial-only, alignment, equalized, shuffled, duplicate.
3. `leakage_phase_diagram`
   - all leakage levels and calibration errors.
4. `geometry_mismatch`
   - subspace angles, bias, residual, recovered source error.
5. `linear_reconstruction`
   - per method/readout/noise/source family.
6. `latent_identifiability`
   - tangent rank and lower eigenvalue distributions.
7. `prior_dominance`
   - data fraction, coverage, prior-swap displacement, null-pair results.
8. `runtime_memory`
   - platform, wall time, peak RSS, cache size, and profile.

## Required figures

1. physical singular-spectrum panels across the 12 Kerr geometries;
2. restricted rank and \(\sigma_{\min}\) versus retained order;
3. null/near-null mode atlas;
4. delay-versus-spatial mechanism ablation;
5. order-leakage phase diagram;
6. geometry-mismatch principal-angle plot;
7. per-age reconstruction error for resolved/partial/unresolved;
8. tangent/secant identifiability distributions;
9. prior-swap and data-fraction map;
10. one concise end-to-end schematic.

---

# 12. Canonical artifacts

Every run must emit:

```text
artifacts/
  manifests/<run_id>.json
  gates/correctness_gates.json
  raymaps/*.h5
  tables/e0_reproduction.parquet
  tables/e1_identifiability_factorial.parquet
  tables/e2_mode_atlas.parquet
  tables/e3_raymap_convergence.parquet
  tables/e4_mechanism_decomposition.parquet
  tables/e5_leakage.parquet
  tables/e6_geometry_mismatch.parquet
  tables/e7_reconstruction.parquet
  tables/e8_latent_identifiability.parquet
  tables/e9_prior_audit.parquet
  tables/e10_instrument.parquet
  figures/
  reports/
  logs/
```

Each manifest includes:

- git commit and dirty-tree status;
- config path and SHA-256;
- environment lock hash;
- hardware and macOS information;
- random seeds;
- input artifact hashes;
- output artifact hashes;
- start/end timestamps;
- peak memory;
- gate status.

---

# 13. Claim firewall

The coding agent must not write or imply any of the following unless the specified evidence exists.

| Claim | Minimum evidence |
|---|---|
| Higher Kerr orders shrink the historical null space | strict restricted-rank gain on physical AART maps across the registered main geometry set |
| Delays are sufficient | FULL and DELAY-ONLY counterfactual both show registered gain |
| Spatial diversity is necessary | FULL gains while DELAY-ONLY does not, with SPATIAL-ONLY/interaction analysis |
| Partial order resolution is viable | benefit persists under registered leakage/instrument arm |
| ML recovers history | data-supported mode recovery, calibrated coverage, OOD tests, and prior-swap audit |
| ML lifts a null space | prohibited; any apparent result triggers leakage/hidden-restriction investigation |
| Real black-hole history is recoverable | prohibited in this synthetic paper |
| Current EHT resolves the required order channels | prohibited unless supported by separate observational evidence |
| All past moments are stored or recoverable | prohibited |

The safe central claim, if supported, is:

> Order-resolved near-critical Kerr channels can enlarge and stabilize the identifiable portion of a declared bounded exterior source-history class; the gain is limited by physical attenuation, order mixing, instrument projection, and geometry uncertainty, and learned priors cannot convert likelihood-null directions into measured history.

---

# 14. Stop and pivot rules

### S1 — Ray-map failure
If AART maps do not converge or cannot be independently checked, stop physical claims and repair the backend.

### S2 — No physical strictness
If resolved physical maps never improve restricted rank or conditioning beyond direct/unresolved controls, retain the theory and publish a negative feasibility result only if the null-space audit is itself novel and robust.

### S3 — Random-control-only effect
If gains appear in E1 random/structured abstractions but not in physical Kerr maps, remove any black-hole feasibility claim.

### S4 — Oracle-only fragility
If gains vanish under modest order leakage, position the result as a requirement for future order-resolving instruments, not an immediately deployable reconstruction.

### S5 — Geometry fragility
If small geometry mismatch dominates source recovery, the next paper must be joint geometry–history estimation; do not hide the dependency.

### S6 — Prior domination
If ML performance changes materially under plausible priors in weak modes, label those modes prior-predicted and remove “recovered” language.

### S7 — OOD collapse
If learned models succeed only on their training morphology, restrict the claim to that source family or stop the general-history claim.

### S8 — Instrument annihilation
If realistic instrument projection restores a large null space, report that the archive is present in ideal ray channels but not recoverable under the tested instrument.

---

# 15. Minimum viable publishable campaign

A first strong paper does **not** require a laboratory, real EHT data, GRMHD-scale training, or a diffusion model. The minimum defensible campaign is:

1. E0 reproduction and all operator gates;
2. E1 structured factorial and E2 null atlas;
3. E3 AART maps for 12 Kerr geometries, orders 0–2;
4. E4 delay/spatial mechanism decomposition;
5. E5 ideal/partial/unresolved order comparison;
6. E6 modest geometry mismatch;
7. E7 linear/state-space reconstruction with exact null-pair controls;
8. E8 compact learned-manifold tangent/secant audit;
9. E9 prior-swap and OOD reconstruction audit;
10. one modest complex-visibility stress arm or a clearly stated image-domain scope.

The most important empirical result is not the prettiest reconstructed movie. It is a robust map of **which historical modes become observable, which merely become less ill-conditioned, and which remain prior choices**.

---

# 16. Execution order

```text
P0  governance + environment lock
P1  E0 toy reproduction
P2  operator library + correctness gates
P3  E1 structured factorial + E2 mode atlas
P4  E3 AART ray maps + cross-check + convergence
P5  E4 mechanism decomposition
P6  E5 leakage + E6 geometry mismatch
P7  E7 linear/state-space reconstruction
P8  E8 learned manifold audit
P9  E9 ML reconstruction + prior-dominance audit
P10 E10 instrument stress (only after P0–P9 pass)
P11 manuscript ingestion and claim ruling
```

No phase may be skipped because a later neural result looks promising.
