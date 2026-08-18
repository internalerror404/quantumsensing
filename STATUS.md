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
| Stage 0 — theorem gates (8, was 7) | **8** (3) | **0** (3) | **0** (1) | 0 |
| Stage 1 — deterministic design | **3** (2) | 0 | 0 | **1** (2) |
| Stage 2 — amortized selector (10) | **6** (3) | **1** (2) | **1** (1) | **2** (4) |
| Required ablations (9) | 0 | 0 | 0 | 9 |
| §6 static redshift (5) | 0 | 0 | 0 | 5 |
| §7 PTA / Hellings–Downs (4) | 0 | 0 | 0 | 4 |

Stage 0 now tests its claims. Stage 1 produces exact rather than Monte-Carlo metrics. Stage 2
has a working policy and a minimal harness that computes gates 2, 3 and 4 — gate 3 fails, which
is a reported result rather than a defect. What remains is the **full** harness (gates 1 and 5,
the nine ablations), the registered multi-seed campaign, and spec §6 and §7, neither started.

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
| 8 | gauge-quotient independence | **DONE** (new) | σ_min(J)=6.4e-3 against a sampled gauge floor of 8.6e-9, **margin 7.5e5**. Makes `d=12` tested rather than asserted, and gives the no-gauge-quotient negative control something to be a control against |

Suite runtime 0.9 s → 46.6 s, from raising the package quadrature order to 1024 and from gate 8's
80 sampled gauge columns. Worth it: at the old order the gauge cancellation was 3.3e-07.

## Stage 1 — deterministic finite-network design (spec §3)

| item | status | note |
|---|---|---|
| 5 baselines (random, angular, leverage, greedy-D, relaxed-E) | **DONE** | relaxed-E is now the convex allocation + multi-start batched swap refinement, within 0.4–1.0% of a 40×-restart search (was 7% below it); greedy-D uses the rank-1 determinant update, 263× faster, identical subset |
| 6 demo metrics (rank, λ_min, logdet, cond, MAP RMSE, runtime) | **DONE** | RMSE is now the closed form `sqrt(tr((F+Λ)⁻¹)/d)` — exact, and the unpaired-Monte-Carlo problem (noise at 10% of the design gaps) is gone |
| registered multi-seed campaign | **ABSENT** | still one run at seed `20260818`, which is **not in** `seed_set: [2026, 3407, 9181, 17041, 27183]`. No registered seed has been run |
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
| task distribution | **DONE** | 7 of 8 axes: `d∈{6,8,10,12}`, `K/d∈{1,1.25,1.5,2}`, pool density, angular **sector** blocking, mixing, scaling, widths. Yields 12/12 distinct D-optimal subsets at 26% overlap. Prior `R` still frozen; `d=16` needs more modes than the family has |
| 20 000 / 2 000 / 5 000 task campaign | **ABSENT** | 1200 steps × batch 8 = 9 600 draws; evaluation is 100 tasks at a held-out seed. No registered-scale splits |
| 6 evaluation baselines vs the policy | **PARTIAL** | five now run against the policy in `evaluate_selector.py`; the per-instance logit oracle still does not exist |
| 5 preregistered ML gates | **PARTIAL** | **3 of 5 computed** — gate 2 PASS (3.57× random, 1.92× angular, CIs excluding 1.0), gate 3 FAIL at 0.026× (gap reported, as the spec permits), gate 4 PASS at 494×. Gates 1 and 5 need the ablation sweep |
| MAP reconstruction on selector output | **ABSENT** | implemented for Stage 1 only |

Gate 3's failure is the substantive open question: the policy sits ~40× below the relaxed-E
oracle and ~18× below a 0.6 ms greedy heuristic. Closing it needs a better selection mechanism,
not a longer run.

## Required ablations (spec §5) — 0 of 9 run

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

Done: gates (3), selector (2), task distribution (4), and the performance work (6) — rank-1
greedy, convex relaxed-E, analytic RMSE. What remains:

1. **Finish the harness** — gates 1 and 5, the per-instance logit oracle, credible-interval
   coverage, and the nine ablation sweeps. Three of five gates are computed; the sweep is what
   the other two need.
2. **Close or characterise gate 3.** The policy is 40× below the oracle. Candidates: a
   sequential/autoregressive selection head instead of one-shot top-K, a differentiable
   determinantal or greedy-unrolled layer, or per-instance logit fine-tuning warm-started from
   the policy. This is the open research question, not a bug.
3. **Run the registered campaign** on `seed_set` — no registered seed has ever been run. Now
   affordable: relaxed-E dropped 7.9 s → 3.5 s and greedy-D is 263× faster.
4. **§6 static redshift** — ~a day, and it unblocks the delay+redshift ablation.
5. **§7 PTA** — separate effort, or a separate paper.
