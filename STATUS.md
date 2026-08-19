# Experiment inventory — done vs remaining

Audited against the experiment spec v0.2 (§2–§7) and `configs/experiment.yaml`. Status assigned
by reading **and running** the package; see `REVIEW.md` for the measurements behind each verdict.

Four states are used, because "the code runs and the gate passes" is not the same as "the
experiment was done":

| | meaning |
|---|---|
| **DONE** | implemented, and the output is evidence for the claim |
| **HOLLOW** | code exists and passes, but does not test what it claims |
| **FAILS** | fails when actually performed |
| **ABSENT** | no code |

---

## Headline

Counts as of the fix commit; the parenthesised figures are where this stood at first review.

| block | done | hollow | fails | absent |
|---|---:|---:|---:|---:|
| Stage 0 — 7 numerical checks + 1 process control | **8** (3) | **0** (3) | **0** (1) | 0 |
| Stage 1 — deterministic design | **5** (2) | 0 | 0 | 0, with 1 partial (2) |
| Stage 2 — amortized selector (10) | **6** (3) | **1** (2) | **1** (1) | **2** (4) |
| Required ablations (8 + 1 convergence sweep) | **7** (0) | 0 | 0 | 1 deferred |
| §6 static redshift (5) | **5** (0) | 0 | 0 | **0** (5) |
| §7 PTA / Hellings–Downs (4) | 0 | 0 | 0 | 4 |

Stage 0 now tests its claims — precisely: **seven scientific/numerical checks and one
stop-on-failure process control (gate 7) pass**; calling all eight "theorem gates" would
overstate what gate 7 measures. Stage 1 produces exact rather than Monte-Carlo metrics. Stage 2
has a working policy and a minimal harness that computes gates 2, 3 and 4 — gate 3 fails, which
is a reported result rather than a defect. What remains is the deterministic full-metric
campaign, the seven remaining scientific ablations and the numerical-convergence sweep, and
the separate §7 detector application. §6 and the joint delay+redshift ablation (D+R-1) are
done and verified; the registered scaling campaign has run on the registered seeds.

---

## Stage 0 — pinned theorem gates (spec §2)

| # | gate | status | note |
|---|---|---|---|
| 1 | conformal `\|A(φη)\|<1e-14` | **DONE** | now a *relative* residual on `h_{μν}=φη_{μν}` assembled and contracted index by index: **2.6e-17**. Was a factored-out scalar, i.e. a test of `normalize()` |
| 2 | compact gauge `\|A(2d^sV)\|<1e-10` | **DONE** | 4.07e-14, and the report now carries the order-convergence table (6.0e-8 @128 → 4.1e-14 @1024) so the tolerance is justified, not asserted |
| 3 | kernel persistence over ray additions | **DONE** | integrates the gauge direction on **all 288 rays** at the order used to build J: **1.2e-12**. Was an analytic zero substituted for the integral; performed properly at the old order it gave 3.3e-07 |
| 4 | `rank(F)=rank(J)` | **DONE** | unchanged — matched `τ_B²` threshold + roundoff floor. Careful work |
| 5 | log-log slope `= -2.00±0.01` | **DONE** | `Var(ν̂)` now computed from `ψ_s` by quadrature and checked against `1/(4s²)` (**6.7e-16**) before the slope is fitted. Was `a²/s²` fitted to itself |
| 6 | endpoint formula, rel. err ≤1e-8 | **DONE** | non-polynomial profile `0.35e^{0.4 sin 1.7t}+0.18 cos 0.9t`, so quadrature error is real: **1.5e-14**. Was degree-2, integrated exactly, error `0.0` |
| 7 | no silent patching | **DONE** (mechanism) | `SystemExit` on failure, report written first. Still a hardcoded `"pass": true` — a policy statement, not a measurement |
| 8 | gauge-quotient independence | **DONE** (new) | certified at **every registered dimension**: margins 1.9e6 (d=6), 7.7e5 (d=8), 7.5e5 (d=12), 6.9e5 (d=16) against a sampled gauge floor of 8.6e-9. Makes each `d` the scaling study uses tested rather than asserted |

Suite runtime 0.9 s → 46.6 s, from raising the package quadrature order to 1024 and from gate 8's
80 sampled gauge columns. Worth it: at the old order the gauge cancellation was 3.3e-07.

## Stage 1 — deterministic finite-network design (spec §3)

| item | status | note |
|---|---|---|
| 5 baselines (random, angular, leverage, greedy-D, relaxed-E) | **DONE** | relaxed-E is now the convex allocation + multi-start batched swap refinement, within 0.4–1.0% of a 40×-restart search (was 7% below it); greedy-D uses the rank-1 determinant update, 263× faster, identical subset |
| 6 demo metrics (rank, λ_min, logdet, cond, MAP RMSE, runtime) | **DONE** | RMSE is now the closed form `sqrt(tr((F+Λ)⁻¹)/d)` — exact, and the unpaired-Monte-Carlo problem (noise at 10% of the design gaps) is gone |
| registered scaling campaign — runtime, rank, memory, blocking | **DONE** | full surface × `seed_set` in `results/scaling_study.json`; see the resolution section below |
| registered full-metric campaign — RMSE, worst-direction, coverage | **DONE** | 64 cells × 5 registered seeds at order 1024, seed-wise records in `results/full_metric_campaign.json`; compact table in `paper/tables/campaign_table.md`; see the campaign section below |
| 3 further config metrics | **PARTIAL** | `worst_direction_error` and `full_rank` now reported; `credible_interval_coverage` still absent |

The single instance is exactly what the spec says it is — an engineering sanity check. It is not
a paper result and the spec does not claim otherwise.

## Stage 2 — amortized ML selector (spec §4)

| item | status | note |
|---|---|---|
| DeepSets architecture | **DONE** | embedding → mean context → score head |
| soft-K train / hard top-K eval | **DONE** | straight-through hard top-K: the forward pass now evaluates the design that gets measured. Was an 8.41-of-16 budget mismatch and a 1000× drop on rounding |
| candidate inputs (`a_i`, `log q_i`, `θ_i`, `z_i`) | **DONE** | 5 of 5 — availability mask now passed as a feature *and* applied to the logits |
| D and E objectives | **DONE** | both implemented; the E surrogate now uses a scale-relative `τ=0.02λ_max` (a fixed 0.03 agreed with the hard min to 1.3e-13) |
| A-optimality | **ABSENT** | marked optional in the spec |
| trained policy that works | **DONE** | 0.120 vs random 0.0185 on 100 held-out tasks. Was 10× *worse* than random |
| task distribution | **DONE** | 7 of 8 axes: `d∈{6,8,10,12}`, `K/d∈{1,1.25,1.5,2}`, pool density, angular **sector** blocking, mixing, scaling, widths. Yields 12/12 distinct D-optimal subsets at 26% overlap. Prior `R` still frozen. `d=16` is supported by the physical family and certified modulo gauge (gate 8), but was not part of the frozen DeepSets v0.2 training distribution |
| 20 000 / 2 000 / 5 000 task campaign | **ABSENT** | 1200 steps × batch 8 = 9 600 draws; evaluation is 100 tasks at a held-out seed. No registered-scale splits |
| 6 evaluation baselines vs the policy | **PARTIAL** | five now run against the policy in `evaluate_selector.py`; the per-instance logit oracle still does not exist |
| 5 preregistered ML gates | **PARTIAL** | **3 of 5 computed** — gate 2 PASS (3.57× random, 1.92× angular, CIs excluding 1.0), gate 3 FAIL at 0.026× (gap reported, as the spec permits), gate 4 PASS at **53.8× end-to-end** (16.56 ms policy vs 891.69 ms oracle; an earlier 494×/1.8 ms figure timed only the model forward pass and is superseded). Gates 1 and 5 need the ablation sweep |
| MAP reconstruction on selector output | **ABSENT** | implemented for Stage 1 only |

Gate 3's failure is the substantive open question: the policy sits ~40× below the relaxed-E
oracle and ~18× below a 0.6 ms greedy heuristic. Closing it needs a better selection mechanism,
not a longer run.

## Required ablations — 7 of 8 complete, 1 deferred

8 registered scientific ablations in `experiment.yaml`, plus one numerical-convergence sweep
tracked separately (quadrature convergence validates implementation, not physics). The audit
taxonomy distinguishes ABSENT (no code) from READY/UNRUN (mechanism exists, sweep not run):

| status | ablations |
|---|---|
| **DONE** | delay-only vs delay+redshift (D+R-1); **QFI vs uniform weights × homogeneous vs heterogeneous packets** (W-1, run as the registered 2×2 with cross-evaluation and total-information control, `results/weight_packet_ablation.json`); **no-gauge-quotient negative control** (NQ-1, `results/no_quotient_control.json`, also in CI); **D vs E** (cross-objective table from the representative computations, `paper/tables/d_vs_e_cross.md`) |
| **DONE (analysis of existing records)** | candidate-pool density and K/d — reduced from the 320 registered campaign records by `experiments/src/analyze_density_budget.py` → `paper/tables/density_budget_analysis.md`, `results/density_budget_analysis.json`. No new numerics. The per-pool-whitening caveat is handled explicitly: cross-M comparisons use raw-coordinate posteriors (fixed prior) plus the declared rescaling M·λ_min; whitened comparisons are used as-is only along the K/d axis at fixed (M, d), where R is identical |
| **CONSOLIDATED** | quadrature/integration stability — one supplement, `paper/tables/convergence_supplement.md`, drawn programmatically from the canonical JSONs |
| **DEFERRED past Paper 1** | independent vs correlated probes (off-diagonal W — foundation of the quantum companion paper) |

**W-1 headline** (3 representative cells × 5 registered seeds, (1/M)Σq = 1 in every arm,
greedy-D designer, every design cross-scored under the true objective): QFI-aware selection
beats weight-blind selection by median **2.36× / 2.32× / 7.18×** on λ_min at
(M,d)=(256,12)/(1024,12)/(4096,16); selected-set overlap is only **28% / 11% / 4%** — the
packet weight changes *which rays are chosen*, not just the score; and heterogeneous
resources under QFI-aware selection beat information-matched homogeneous resources by
**1.94× / 2.08× / 2.82×**. This substantiates the claim that packet bandwidth enters the
Lorentzian normal operator as a meaningful statistical weight. No success criterion was
preregistered; the numbers are the result.

**NQ-1 headline** (72-direction pool, 12 physical + 2 conformal + 2 potential-gauge columns,
order 1024): unquotiented nullity is exactly 4 at every ray budget in {16,32,64,128,288};
scale separation is explicit — exact conformal zeros 1.2e-18, quadrature potential-gauge
floor 2.4e-10, smallest physical singular value 6.9e-2 — with the rank threshold pinned
between the floors so quadrature noise is never counted as recovered gauge information.

**Density/budget headline** (reduction of the 320 campaign records, medians over the
registered seeds): rank is never the constraint anywhere on the registered surface — full
rank in all 320 records, including under pinned 20%-sector blocking — so pool density and
ray budget act purely on conditioning and precision. Enlarging the pool 16× (M 256 → 4096
at K/d = 1.5) buys ~2× pool-rescaled λ_min at d ≤ 12 and **4.03×** at d = 16, at only
1.7–3.2× selection-latency cost (still ≤ 5 ms median); the density benefit grows with
dimension, consistent with W-1. Doubling the ray budget (K/d 1 → 2 at M = 1024) buys
2.4–3.2× whitened λ_min and 1.6–2× conditioning; marginal per-step gains stay ≥ 1.25× up
to K/d = 2 at d ≥ 8, while at d = 6 they saturate at K/d = 1.5 (last step 1.01×).

Runnable today via existing flags, just never swept:

| ablation | how |
|---|---|
| D vs E objective | `--objective` |
| candidate-ray density | now a per-task random variable in `generate_tasks` |
| K/d budget ratio | now a per-task random variable; `--k` pins it |
| quadrature resolution / domain | `DEFAULT_ORDER`, `lam_extent=`; gate 2 already reports a convergence table |

Need new code:

| ablation | missing piece |
|---|---|
| no gauge quotient (negative control) | a non-quotiented mode family to contrast against |
| uniform vs QFI weights | trivial, but `q` is threaded through everywhere |
| homogeneous vs heterogeneous packet width | width sampler is hardcoded log-uniform |
| independent vs correlated pure probes | **no off-diagonal `W` support anywhere** — `light_ray.py` and both experiment scripts assume `W=diag(q)`. This is the largest of the five |
| delay-only vs delay + static redshift | **DONE** — Experiment D+R-1 (`run_joint_ablation.sh`, `results/joint_delay_redshift.json`); see the D+R-1 section below |

## §6 static redshift / conformal rank restoration — DONE

Implemented via an externally authored, independently verified patch (base blobs and SHA-256
checksums matched; order-512 and order-1024 reruns on this machine reproduced the shipped
reports **byte-identically**). `experiments/run_static_redshift.sh`, canonical report at order
1024 in `results/static_redshift_experiment.json`; also added to CI. Problem: 4 localized
stationary conformal modes, 5 static clocks, 10 candidate links, 144 delay rays.

| check | result |
|---|---|
| conformal modes `h_j = φ_j η` as a family | 4 stationary modes, componentwise tensor assembly |
| endpoint matrix `R_lj = ½(φ_j(A_l) − φ_j(B_l))` | two independent code paths agree exactly; graph form `R = ½ B Φ` exact; link reversal antisymmetric |
| delay-only conformal rank = 0 | max \|A\| = 3.4e-18 over 144 rays, rank 0 |
| combined rank = rank(R) | redshift rank 4 = combined rank 4 |
| rank lifts exactly as predicted | greedy-selected 4 links (O0→O2, O1→O2, O1→O3, O3→O4) give rank 4; 3 links give rank 3 — the row-count lower bound is realized |
| invariance under allowed static gauge | stationary V, V₀=0, V=0 at every clock; assembled tensor norm 8.6e-2 at endpoints (not a zero perturbation), h₀₀=h₀ᵢ=0, redshift response 0 |

Beyond the five registered checks: an independent **nonlinear lapse** validation
(exact ζ(ε) from N=√(1+εφ); centered-difference slope 2.00017 → the linear formula is
derived, not self-compared), and order-512/1024 stability (identical reports up to the
delay-noise floor; identical selected links).

One caveat recorded honestly: the redshift-block gauge invariance is structural — R reads
only h₀₀, which vanishes identically for any stationary V with V₀=0. The endpoint-fixing
makes the tested perturbation visibly nonzero at the clocks, which is what the check
demonstrates. Integration-side verification (this session): the same gauge modes also vanish
on the **delay** rays — interior-support cancellation converges 1.4e-8 (order 1024, the
narrow 0.30-radius bumps) → 2.0e-11 (2048) → 4.4e-15 (4096) → 5.8e-17 (8192) — so the
**combined** delay+redshift channel is gauge invariant for the allowed class, not only the
redshift block. The delay+redshift ablation should integrate these modes at order ≥ 2048 or
widen the bumps.

## Experiment D+R-1: joint delay–redshift rank restoration — DONE

The delay-vs-delay+redshift ablation, run as the mixed-channel experiment rather than a
concatenation smoke test. Joint stationary family `h = Σθᵢeᵢ + Σαⱼφⱼη` with d_p ∈ {6,8,12,16}
stationary physical modes (all h₀ᵢ=0, inside the static ansatz; the scalar modes have
nonzero R_p, so the rank law is tested in its nontrivial form), the four §6 conformal
modes, 144 candidate rays, 10 candidate links. Order 1024; 7.9 s. **All ten locked gates
pass**, including stop-on-patch. Observed ranks, every d_p:

| arm | expected | observed |
|---|---|---|
| A: K=d_p+4 delay rays | d_p | d_p ✓ |
| B: + 3 greedy links | d_p+3 | d_p+3 ✓ |
| C: + 4 greedy links | d_p+4 | d_p+4 ✓ |
| D: + 4 MORE delay rays (negative control) | d_p, nullity 4 | d_p, nullity 4 ✓ |
| E: + all 10 links | d_p+4 | d_p+4 ✓ |
| F: E + constant conformal mode | d_p+4; kernel on that mode | ✓, kernel weight > 0.999 |

The rank law `rank(J_joint) = d_p + rank(R_c)` held in every registered case. Arm D is the
load-bearing negative control: four additional delay rays (highest-sensitivity-norm
heuristic — the choice is irrelevant, since every delay row has identically zero conformal
columns) leave the conformal nullity at exactly 4, and the machine-readable full-bank check
confirms it for the **entire 144-ray delay bank** (rank d_p, nullity 4). Four clock links
lift it to 0 — more data through the null channel does nothing; a new observable does
everything. Whitened λ_min moves from
~1e-35 (arms A, D) to 3e-2 (arm C). Gate 9 confirms appending clock rows never shrinks the
physical block's spectrum (max degradation ≤ 0); gate 7's kernel vector localizes on the
constant conformal mode to >0.999.

The supplementary §6 gauge check is now a **machine-readable gate**
(`combined_static_gauge_invariance`: delay residuals 1.4e-8 / 2.0e-11 / 4.4e-15 at orders
1024/2048/4096, selected order 2048 against 1e-10; redshift response exactly 0), so it no
longer lives only in narrative history.

Statistical layer, declared not hidden, in **both coordinate conventions**: raw mode
coordinates (transparent audit; conformal std 1.0000 → 0.9995 under unit noise — honest and
deliberately weak, and the physical block sits at ~0.997 too: the whole synthetic instance is
low-information relative to its N(0,I) prior, not just the clocks) and full-bank-whitened
coordinates (rescaling-invariant; conformal rmse 0.861 at ρ=1). The clock-precision curve is
a **log-spaced sweep** in ρ = σ_D/σ_R (25 points, 0.1–100, whitened coordinates), with the
declared points ρ ∈ {1, 10, 50} marked: u_conf = 0.861 / 0.401 / 0.312 (4 links, d_p=12).
Rank restoration is exact at every ρ; posterior contraction is the separate precision
statement, and the report keeps the two claims apart (`paper_claim` uses rank language only;
`statistical_qualification` carries the caveat). Coverage is calibrated (~0.95) in every arm.
The central three-panel figure (visibility spectrum / rank restoration / precision
conversion) is generated by `experiments/src/make_joint_figure.py` →
`paper/figures/joint_rank_restoration.{pdf,png}`. The static ansatz is now **enforced at
runtime**: `static_tensor_redshift_matrix` rejects any mode with |h₀ᵢ| > 1e-14 at a clock
endpoint rather than silently producing rows the endpoint formula does not cover.

## Registered full-metric campaign — DONE

`run_full_metric_campaign.sh` → `results/full_metric_campaign.json` (seed-wise records) and
the compact paper table `paper/tables/campaign_table.md`. Order 1024 throughout; J built
once per pool size and reused across dimensions, budgets, and seeds (the seed changes q,
not the geometry). 64 cells × 5 registered seeds = 320 records, 104 s total.

Headlines: **100% full-rank in all 320 records**; whitened λ_min spans 0.083–4.27 across the
surface; median blocking retention 0.989 under the pinned 20% sector (per-cell medians
0.63–1.00, weakest at M=512, d=16); coverage calibrated at 0.945–0.948 on the representative
cells (a calibration check, not a design discriminator — the model is correctly specified);
greedy selection 0.7–7 ms per cell against J-builds of 1.4–26 s, confirming the cost
structure. Posterior blocks reported in both declared conventions (whitened RMSE 0.26–0.45;
raw RMSE ~0.90–0.94, the low-information regime already noted for D+R-1). The relaxed-E
reference on the three registered representative cells is 2.0–3.7× greedy-D on λ_min at
1.1–41 s per instance — the expected E-vs-D objective gap, priced but not deployed.

## Manuscript note: promote the block-rank law to a proposition

D+R-1's rank law should enter the paper as a stated proposition, not an empirical
observation: for J = [[D_p, 0], [R_p, R_c]] with D_p of full column rank,
ker J = {0} ⊕ ker R_c, hence rank J = d_p + rank R_c. One-paragraph proof: if J(x,y)ᵀ = 0
then D_p x = 0, full column rank gives x = 0, and the remaining equation is R_c y = 0. The
experiment then *verifies the hypotheses and demonstrates the law* in four nontrivial finite
families with R_p ≠ 0 — which is the correct division of labor between theorem and
computation. The paper's restrained quantum implication follows: for any probe QFI W_ρ ⪰ 0,
ker J ⊆ ker(Jᵀ W_ρ J), so quantum resources optimize extraction within a forward map; only
new observables enlarge the identifiable subspace.

## §7 PTA / LISA application gate — ABSENT

All four staged steps absent: one-way link response, pulsar + Earth terms, isotropic stochastic
GW ensemble, Hellings–Downs to <1% RMS. No code (`grep -i "hellings\|pulsar\|PTA\|LISA"` → nothing).

By far the heaviest item in the program and the only one that is not laptop-scale. Nothing in
Stages 0–2 depends on it, and the paper already frames it as an application anchor rather than
evidence for the microlocal theorem — so it is the cleanest thing to defer or split off.

## Reproducibility bundle (paper Appendix C) — 5 of 6

Present: theorem-verification script, design benchmark, DeepSets sanity implementation, YAML
registration, JSON reports. Not in what was shared: the LaTeX manuscript and bibliography.

---

## Resolution of the ML question (post-hoc, v0.2 gates unchanged)

The deterministic **scaling study has now run on the full registered surface with the
registered seeds** — the first use of `seed_set` anywhere in the package
(`results/scaling_study.json`, `run_scaling_study.sh`):

- greedy D-optimal worst case: **~6 ms** at M=4096 (run-to-run range 3.8–6.0 ms on
  this machine); median 0.14–2.8 ms across all 64 cells; peak traced memory **586 KiB**;
- **100% full-rank success on every cell and every registered seed**, including
  under a pinned worst-case 20% angular-sector block — median λ_min retains 97%
  of its unblocked value (worst cell 42%, still full rank);
- relaxed-E reference at K/d=1.5: 0.3 s (M=256) to 42.5 s (M=4096, d=16);
- **order stability**: the runtime sweep builds J at order 512; on the boundary
  cells (M=256/d=6, M=1024/d=12, M=4096/d=16) the greedy-selected subsets are
  **identical** between orders 512 and 1024 on every registered seed, with
  λ_min shifts ≤ 2.9e-8. Final paper tables must still be built at order 1024;
- **normalization declared**: reported eigenvalues are of the whitened
  visibility operator, R = diag(‖J_col‖²), J̃ = JR^(-1/2). Rank and
  blocked/unblocked ratios are normalization-independent; absolute
  cross-family λ_min values are not physical QFI in the raw parameterization;
- **cost structure**: at M=4096 the J-build is ~13 s while selection is
  milliseconds. The DeepSets policy consumes features derived from J, so it
  never touched the dominant cost — it amortized the step that was already
  cheap. This, not latency alone, is why ML has no operational role here.

The post-hoc practical-utility analysis (`selector_evaluation.json`) shows the v0.2 policy is
**strictly Pareto-dominated**: greedy-D is 23.3× better on the objective (CI95 [13.5, 43.7])
at 0.37 ms vs the policy's 16.6 ms. Preregistered outcomes stand: gate 2 PASS, gate 3 FAIL
(gap reported), gate 4 PASS.

**Selector disposition:** DeepSets v0.2 is frozen; the 20 000/2 000/5 000-task campaign will
**not** be run for this architecture — a post-hoc practical-futility decision (recorded here,
gates unchanged): a larger campaign would only add precision around a structurally clear
conclusion.

**Consequence for Paper 1:** deterministic Fisher-aware selection is cheap everywhere the
registration reaches, so amortization is not currently justified by latency. The one-shot
DeepSets selector is frozen as an honest negative baseline; Paper 1 should present it as a
diagnostic ("one-shot amortized selection was tested and dominated by deterministic
Fisher-aware optimization"), resting on the theorems, the eight gates, the registered
deterministic experiments, static-redshift rank restoration, and one-shot discrimination.
Stronger gates for any future v0.3 architecture (Pareto non-domination, quality-at-latency
≥ 0.95, staged 0.80/0.95 oracle targets) are registered forward-only in
`configs/experiment.yaml`.

## Critical path

Done: gates (3), selector repair (2), task distribution (4), performance (6), the scaling
study on registered seeds, the post-hoc Pareto analysis, a CI portability gate
(`.github/workflows/experiments.yml`, macos-14 + ubuntu × Python 3.11/3.12, uploading the
JSON reports), and **§6 static redshift** (externally authored patch, independently
verified and integrated). What remains:

1. ~~Delay-only vs delay+redshift ablation~~ — **done** (Experiment D+R-1, all ten gates,
   every d_p). The paper's central experimental figure now exists.
2. ~~Registered deterministic full-metric campaign~~ — **done** (64 cells × 5 seeds at
   order 1024; `paper/tables/campaign_table.md` is the paper's main quantitative table).
3. ~~Deterministic ablations~~ — the minimal submission package is **complete**: W-1
   (weights × packets, cross-evaluated), NQ-1 (no-quotient control), the D-vs-E
   cross-objective table, and the consolidated convergence supplement. Candidate density
   and K/d are analyzed from the existing campaign records
   (`paper/tables/density_budget_analysis.md`). Off-diagonal W remains deliberately
   deferred to the quantum companion paper.
4. **Manuscript work is now the bottleneck** (v1.0 checked against the canonical JSONs —
   the cited numbers match, and it wisely says "millisecond scale" rather than pinning a
   jittery worst-case): theorem-by-theorem proof audit, quantum-metrology review of the
   QFI statements, novelty matrix, citation verification, venue positioning. Go/no-go
   standard: stop experimenting unless a run exposes gauge leakage, rank instability under
   certified quadrature, extreme weight sensitivity, failure outside a hand-picked family,
   or raw-vs-normalized inconsistency. None observed.
4. **v0.3 learned selector** — only if an activation condition in the YAML becomes true;
   sequential Fisher-aware scoring (a_i^T F_t^{-1} a_i, alignment with v_min(F_t)) + swap
   refinement, against the forward-registered practical gates.
5. **§7 PTA** — separate effort, or a separate paper.
