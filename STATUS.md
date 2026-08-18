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

| block | done | hollow | fails | absent |
|---|---:|---:|---:|---:|
| Stage 0 — theorem gates (7) | 3 | 3 | 1 | 0 |
| Stage 1 — deterministic design | 2 | 0 | 0 | 2 |
| Stage 2 — amortized selector (10) | 3 | 2 | 1 | 4 |
| Required ablations (9) | 0 | 0 | 0 | 9 |
| §6 static redshift (5) | 0 | 0 | 0 | 5 |
| §7 PTA / Hellings–Downs (4) | 0 | 0 | 0 | 4 |

Stages 0–2 have working scaffolding end to end. The **evidence** they were built to produce
mostly does not exist yet, and spec §6 and §7 have not been started.

---

## Stage 0 — pinned theorem gates (spec §2)

| # | gate | status | note |
|---|---|---|---|
| 1 | conformal `\|A(φη)\|<1e-14` | **HOLLOW** | `k@ETA@k` factors out as a scalar; reduces to testing `normalize()`. Passes at 8e-19 |
| 2 | compact gauge `\|A(2d^sV)\|<1e-10` | **DONE** (narrow) | real quadrature test, passes at 4.07e-14 — but on **one ray** at order 1024 |
| 3 | kernel persistence over ray additions | **FAILS** | substitutes the analytic endpoint formula, which is exactly `0.0` at all 576 endpoints. Performed properly: **3.3e-07 vs a 1e-10 tolerance** at the order `mode_matrix` uses |
| 4 | `rank(F)=rank(J)` | **DONE** | matched `τ_B²` threshold + explicit roundoff floor. Careful work |
| 5 | log-log slope `= -2.00±0.01` | **HOLLOW** | computes `a²/s²`, then fits the slope of `a²/s²`. Never builds a packet or computes `Var(ν̂)`. Result: −2.0000000000000004 |
| 6 | endpoint formula, rel. err ≤1e-8 | **HOLLOW** | integrand is degree 2, so Gauss–Legendre is exact. Measured relative error: **0.0**. Cannot fail |
| 7 | no silent patching | **DONE** (mechanism) | `SystemExit` on failure, report written first. Reported as a hardcoded `"pass": true`, which is a policy statement, not a measurement |

**Not in the spec but should be:** the "12-dimensional gauge-quotiented family" claim is nowhere
tested. I tested it and it **holds** (σ_min(J)=7.4e-3 vs a gauge floor of 2.5e-8). Promote to gate 8.

## Stage 1 — deterministic finite-network design (spec §3)

| item | status | note |
|---|---|---|
| 5 baselines (random, angular, leverage, greedy-D, relaxed-E) | **DONE** | all five implemented and run |
| 6 demo metrics (rank, λ_min, logdet, cond, MAP RMSE, runtime) | **DONE** | all six reported |
| registered multi-seed campaign | **ABSENT** | one run at seed `20260818`, which is **not in** `seed_set: [2026, 3407, 9181, 17041, 27183]`. No seed in the registered set has ever been run |
| 3 further config metrics | **ABSENT** | `worst_direction_error`, `credible_interval_coverage`, `full_rank_success_rate` |

The single instance is exactly what the spec says it is — an engineering sanity check. It is not
a paper result and the spec does not claim otherwise.

## Stage 2 — amortized ML selector (spec §4)

| item | status | note |
|---|---|---|
| DeepSets architecture | **DONE** | embedding → mean context → score head |
| soft-K train / hard top-K eval | **HOLLOW** | both exist, but the top-16 hold only 8.41 of the K=16 soft budget — the two regimes optimize different designs |
| candidate inputs (`a_i`, `log q_i`, `θ_i`, `z_i`) | **DONE** | 4 of 5; availability masks claimed in the README are not passed |
| D and E objectives | **DONE** | both implemented, selectable by flag |
| A-optimality | **ABSENT** | marked optional in the spec |
| trained policy that works | **FAILS** | **10× worse than random** on its own objective. Fixable — see REVIEW.md §B1, 12.8 s CPU to 1.37× angular_spread |
| task distribution | **HOLLOW** | 3 of 8 registered axes randomized. The primary axis (orthogonal mixing) leaves the D-optimal ranking **exactly** invariant; blocking is i.i.d. per ray, not angular sectors; `d`, `M`, `K/d`, prior `R` are all frozen |
| 20 000 / 2 000 / 5 000 task campaign | **ABSENT** | 250 steps × batch 8 = 2 000 draws, no splits |
| 6 evaluation baselines vs the policy | **ABSENT** | five exist in `design_experiment.py` but are **never run against the selector**; the per-instance logit oracle does not exist at all |
| 5 preregistered ML gates | **ABSENT** | **0 of 5 computed by any code in the package** |
| MAP reconstruction on selector output | **ABSENT** | implemented for Stage 1 only |

The missing evaluation harness is why the selector's failure is invisible in its own report.

## Required ablations (spec §5) — 0 of 9 run

Runnable today via existing flags, just never swept:

| ablation | how |
|---|---|
| D vs E objective | `--objective` |
| candidate-ray density | `candidate_rays(direction_count, offsets_per_direction)` |
| K/d budget ratio | `--k` |
| quadrature resolution / domain | `order=`, `lam_extent=` |

Need new code:

| ablation | missing piece |
|---|---|
| no gauge quotient (negative control) | a non-quotiented mode family to contrast against |
| uniform vs QFI weights | trivial, but `q` is threaded through everywhere |
| homogeneous vs heterogeneous packet width | width sampler is hardcoded log-uniform |
| independent vs correlated pure probes | **no off-diagonal `W` support anywhere** — `light_ray.py` and both experiment scripts assume `W=diag(q)`. This is the largest of the five |
| delay-only vs delay + static redshift | blocked on §6 below |

## §6 static redshift / conformal rank restoration — ABSENT

The spec lists five checks; none is implemented. `light_ray.py` has **no redshift functional at
all** — grep for `redshift`, `emitter`, `receiver`, `zeta` returns nothing.

| check | status |
|---|---|
| conformal modes `h_j = φ_j η` | partial — `conformal_contraction` exists, but not as a family |
| endpoint matrix `R_lj = ½(φ_j(A_l) − φ_j(B_l))` | **ABSENT** |
| delay-only conformal rank = 0 | **ABSENT** as a rank statement |
| combined rank = rank(R) | **ABSENT** |
| invariance under allowed static gauge transformations | **ABSENT** |

Smallest genuinely new experiment in the program, and the only place clock resources enter.
Corollary 5.2 is pure finite-dimensional linear algebra once `R` exists — this is a day of work,
not a research project, and it unblocks the delay+redshift ablation.

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

## Critical path

The registered campaign cannot run meaningfully until the harness exists, and the harness is
what turns a passing script into a result:

1. **Evaluation harness** — splits, the 6 baselines, bootstrap CIs, the 5 ML gates. Unblocks all
   of Stage 2 and all 9 ablations. Nothing about the ML claim is measurable without it.
2. **Fix the selector** (REVIEW.md §B1) — verified, ~13 s CPU.
3. **Fix gates 3, 5, 6, and 1** so Stage 0 tests its claims; add the gauge-independence gate.
4. **Fix the task distribution** — sector blocking, and randomize `d`, `M`, `K/d`.
5. **§6 redshift** — ~a day, unblocks the delay+redshift ablation.
6. **Performance** before the campaign, not after: rank-1 greedy (263× measured), convex
   relaxed-E, analytic RMSE. Without these the campaign is ~55 h single-core *per configuration*.
7. **§7 PTA** — separate effort, or a separate paper.
