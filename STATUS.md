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
| Stage 1 — deterministic design | **4** (2) | 0 | 0 | 0, with 2 partial (2) |
| Stage 2 — amortized selector (10) | **6** (3) | **1** (2) | **1** (1) | **2** (4) |
| Required ablations (9) | 0 | 0 | 0 | 9 |
| §6 static redshift (5) | **5** (0) | 0 | 0 | **0** (5) |
| §7 PTA / Hellings–Downs (4) | 0 | 0 | 0 | 4 |

Stage 0 now tests its claims — precisely: **seven scientific/numerical checks and one
stop-on-failure process control (gate 7) pass**; calling all eight "theorem gates" would
overstate what gate 7 measures. Stage 1 produces exact rather than Monte-Carlo metrics. Stage 2
has a working policy and a minimal harness that computes gates 2, 3 and 4 — gate 3 fails, which
is a reported result rather than a defect. What remains is the **full** harness (gates 1 and 5,
What remains is the deterministic full-metric campaign, the eight scientific ablations and
numerical-convergence sweep, and the separate §7 detector application. §6 is done and
independently verified; the registered scaling campaign has run on the registered seeds.

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
| registered full-metric campaign — RMSE, worst-direction, coverage | **PARTIAL / REMAINING** | metrics implemented in `design_experiment.py` (except coverage) but not yet swept over the registered seeds; this is the paper's main quantitative table |
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

## Required ablations — 0 run (8 registered scientific ablations in `experiment.yaml`, plus one numerical-convergence sweep the spec text lists; quadrature convergence validates implementation, not physics, so it is counted separately)

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
| delay-only vs delay + static redshift | **unblocked** — §6 forward blocks exist; the ablation run itself remains |

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

1. **Delay-only vs delay+redshift ablation** — the first registered run that joins the two
   forward blocks; the §6 caveat above sets its quadrature requirements.
2. **Registered deterministic full-metric campaign** — RMSE, worst-direction error,
   credible-interval coverage, condition number, blocked/unblocked retention, build and
   selection time, at order 1024, over `seed_set`. The paper's main quantitative table.
3. **Deterministic ablations** — no-gauge-quotient negative control, uniform vs QFI weights,
   homogeneous vs heterogeneous widths, delay vs delay+redshift (after §6), candidate
   density, K/d, D vs E. The learned-policy ablation campaign is no longer needed for
   Paper 1. Off-diagonal W (correlated probes) is deliberately deferred past Paper 1 — it
   is the foundation of the quantum companion paper, not a blocker here.
4. **v0.3 learned selector** — only if an activation condition in the YAML becomes true;
   sequential Fisher-aware scoring (a_i^T F_t^{-1} a_i, alignment with v_min(F_t)) + swap
   refinement, against the forward-registered practical gates.
5. **§7 PTA** — separate effort, or a separate paper.
