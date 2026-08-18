# Review — QFI Lorentzian Light-Ray Tomography experiment package (v0.2)

Reviewed: paper v0.2, experiment spec v0.2, `experiments/src/*.py`, `configs/experiment.yaml`,
`requirements.txt`, the three `run_*.sh` scripts.

Everything shared was reconstructed into the layout the scripts assume and **executed**. All
numbers below are measured on this machine (Linux, NumPy 2.4.6, SciPy 1.17.1, Torch 2.13, CPU),
not inferred from reading. Reproduction scripts are in `review/`.

---

## STATUS: the three blocking findings are fixed

Fixed on this branch and re-verified by running the package. The diagnoses below are kept as the
evidence trail; measurements marked "was" are the pre-fix numbers.

| finding | before | after |
|---|---|---|
| **B1** selector 10x worse than random | 0.0020 vs random 0.0205 | **0.120 vs random 0.0185** — gate 2 passes at 3.57x (random) and 1.92x (angular), both CIs excluding 1.0 |
| **B3** gate 3 vacuous, fails when honest | analytic zero substituted; real value 3.3e-07 vs 1e-10 | **integrates all 288 rays at order 1024: 1.2e-12** |
| **M4/M5** gates 1, 5, 6 cannot fail | slope -2.0000000000000004; gate 6 error exactly 0.0 | **gate 5 computes Var(nu) from the packet (matches 1/(4s^2) to 6.7e-16); gate 6 uses a non-polynomial V, error 1.5e-14; gate 1 is now a relative residual on the assembled tensor** |
| **M1** task mixing leaves D-optimal ranking invariant | logdet shift spread 4e-12; ~100% subset overlap | **12/12 distinct D-optimal subsets, 26% mean pairwise overlap** |
| **B4** relaxed-E "oracle" below a naive search | -7.0% vs 40x-restart, 7.9 s | **-0.4% to -1.0% vs 40x-restart, 3.5 s** |
| **new** gate 8, gauge-quotient independence | untested | **margin 7.5e5, passes** |

Also fixed in passing, because they blocked the Mac run or the above: the `torch>=2.2` /
NumPy 2 pin conflict, `python` to `${PYTHON:-python3}` in all four run scripts,
`matplotlib.use("Agg")`, angular *sector* blocking replacing i.i.d. thinning, availability
masking of logits, float64 in the training loss, rank-1 greedy D (263x, identical subset),
analytic posterior RMSE replacing 500-trial Monte Carlo, and `worst_direction_error` /
`full_rank` added to the Stage 1 metrics.

A minimal evaluation harness was added (`experiments/run_selector_eval.sh`) because B1 is not
verifiable without one. On 100 held-out tasks at a seed distinct from training:

```
                  median lambda_min
   learned              0.1202
   random               0.0185
   angular_spread       0.0509
   leverage             0.5286
   greedy_D             2.1746
   relaxed_E            3.7157

   gate 2 vs random           3.57x  CI95 [2.28, 5.85]   PASS
   gate 2 vs angular_spread   1.92x  CI95 [1.09, 3.26]   PASS
   gate 3 vs oracle           0.026x CI95 [0.015, 0.041] FAIL - gap reported, as the spec permits
   gate 4 speedup              494x  (1.80 ms vs 889 ms) PASS
   held-out full-rank rate     100%
```

**Gate 3 still fails, and that is the honest result.** The policy is ~40x below the relaxed-E
oracle and ~18x below greedy D-optimal, which runs in 0.6 ms. The spec anticipates this
("or the gap is reported"). It is a real finding about DeepSets-with-top-K on this problem, not
a bug: closing it needs a better selection mechanism, not a longer run. Note also that gate 2
passes while the policy loses to a 0.6 ms greedy heuristic by 18x -- the gate only requires
beating the two weakest baselines, which is the discrimination problem in M7.

Whole suite, end to end: **2.7 minutes.**

```
run_task1.sh            46.6 s     8/8 gates
run_design_demo.sh       4.1 s     (was 8.5 s)
run_selector_sanity.sh  16.4 s     1200 steps
run_selector_eval.sh    92.1 s     100 held-out tasks
make_figures.py          1.7 s
```

**Update after the scaling study:** the deterministic scaling study has since run over the full
registered surface on the registered seeds: greedy-D worst case **3.8 ms** at M=4096, d=16,
K=32, 100% full-rank on all 64 cells. The post-hoc Pareto analysis shows the policy strictly
dominated by greedy-D (23.3× better, 45× faster). Preregistered v0.2 outcomes unchanged;
consequence and forward-only v0.3 gates in `STATUS.md` and `configs/experiment.yaml`.

Still open, in priority order: spec section 6 (static redshift), the ablation sweeps and gates
1 and 5, the full deterministic campaign metrics, and spec section 7 (PTA). See `STATUS.md`.

---

## Bottom line

The physics and the linear algebra are sound — I checked the derivations and they hold, and the
12-mode family really is independent modulo gauge. Stage 0 and Stage 1 run clean in seconds.

But **three of the seven theorem gates cannot fail**, one of them because it substitutes an
analytic zero for the numerical integral it claims to test — and when that integral is actually
computed it misses its own tolerance by ~3.5 orders of magnitude. And **the learned selector,
the actual scientific contribution of Stage 2, currently performs 10× worse than random
selection** on its own registered objective. None of the five preregistered ML gates are
computed by any code in the package.

None of this is fatal. The failures are concentrated in training mechanics and gate
construction, not in the theory. I fixed the selector in ~13 s of CPU time (below) as proof.

---

## 1. What is correct

Worth stating plainly, because it is most of the package.

**Derivations checked by hand, all correct:** Lemma 2.1 (eikonal linearization, including the
inverse-metric sign), Prop 3.1 and Cor 3.2 (`Var(ν̂)=1/(4s²)` ⟹ `F=(1/s²)(Au)(Av)`),
Prop 4.1 (both the conformal and potential kernels, and the endpoint term), eqs (24)–(25)
(`N²=1-εh₀₀` ⟹ `R_AB h = ½(h₀₀(B)-h₀₀(A))`, and `R_AB(φη)=½(φ(A)-φ(B))`), the timelike-invisibility
argument (`|ξ₀|>|ξ'|` ⟹ empty constraint set), Theorem 7.1, and Prop 8.1 including the
small-Δτ expansion.

**The 12-mode family is genuinely gauge-quotiented.** This is the load-bearing claim of Stage 1
and it is nowhere verified in the code, so I verified it: built an 80-column numerical gauge
block (60 sampled `2d^sV` modes + 20 sampled `φη` modes) on the same ray pool.

```
gauge block max|G| = 2.5e-08      (pure quadrature noise, as it should be)
sigma(J):  min 7.40e-03   max 2.02e-01   cond 27.4
```

Six orders of margin between the smallest physical singular value and the gauge floor. The
claim holds. **Add this as a gate** — right now `d=12` is asserted, not tested.

**Gate 2 does real work** and passes with a 4000× margin (4.07e-14 against 1e-10).

**The Bayesian setup is correctly specified**: `θ~N(0,I)` prior matches `Λ=I` in the MAP gain,
and `Σ_S=diag(s²)` matches `noise_std=1/√q`. The estimator is the right one.

**Stage 1 reproduces Fig. 2 qualitatively** — λ_min spans 9e-4 (random) to 2.45 (relaxed_E).

---

## 2. Blocking issues

### B1 — The learned selector is 10× worse than random

Ran `run_selector_sanity.sh` as shipped, then evaluated the deterministic baselines on the
*exact same held-out task* (replaying the RNG stream):

| design | λ_min | vs learned |
|---|---:|---:|
| **learned selector (E)** | **0.002037** | 1.0× |
| random | 0.020483 | 10.1× |
| angular_spread | 0.137610 | 67.6× |
| leverage | 0.431021 | 211.6× |
| greedy_D | 0.952001 | 467.4× |
| relaxed_E | 2.517062 | 1235.9× |

The report says `rank: 12, min_eigenvalue: 0.00204` and looks healthy in isolation. It is a
catastrophic failure, and nothing in the pipeline surfaces it, because `train_selector.py`
never runs a baseline.

Four compounding causes, each measured:

1. **Soft/hard mismatch.** At the trained logits the top-16 rays hold only **8.41 of the K=16**
   soft photon budget — 47% of the mass being optimized lives outside the set that gets
   evaluated. Soft λ_min ≈ 2.23, hard top-K λ_min = 0.002. A 1000× gap between the training
   objective and the evaluation metric.
2. **The softmin is not smooth.** `tau = 0.03` is fixed, but eigenvalues span 2.2–28.7. So
   `-τ·logsumexp(-λ/τ)` equals the hard min to **1.3e-13** — gradients flow through a single
   eigenvector, reintroducing exactly the E-optimal degeneracy the surrogate exists to avoid.
   `design_experiment.py` gets this right (`tau = 0.02*λ_max`); the trainer does not. Same
   surrogate, two implementations, only one correct.
3. **Temperature never anneals.** `max(0.25, 1.5*0.999^step)` is 1.168 at step 250 and still
   0.334 at the default 1500. The 0.25 floor needs ~5000 steps.
4. **Training in float32** for λ_min of a 12×12 Gram. Free to fix; these matrices are tiny.

**Verified fix.** Five changes — straight-through hard top-K (train forward = eval forward),
`tau = 0.02*λ_max`, anneal to 0.05, float64, quadrature order 512 — 1200 steps, **12.8 s CPU**:

```
median lambda_min over 30 held-out tasks
   learned     0.1154      (was 0.0020)
   angular     0.0841
   leverage    0.2162
   greedy_D    1.1125

learned / angular_spread = 1.37x     (learned_gate_2 needs >= 1.15x)   PASSES
original code            = 0.015x                                       FAILS
```

See `review/selector_fix_probe.py`. Note the policy still sits **10× below greedy_D**, which
takes 0.6 ms — so `learned_gate_3` (within 5% of the oracle) still fails badly. That is a real
result worth reporting, not a bug.

### B2 — No evaluation harness exists

Zero of `learned_gate_1..5` are computed anywhere in the package. `train_selector.py` trains and
reports **one** held-out task with no baseline, no split, no bootstrap, no timing comparison, no
ablation. The config preregisters five gates and nine ablations against code that evaluates none
of them. This is the largest missing component by volume, and it is what let B1 go unnoticed.

### B3 — Gate 3 is vacuous, and the honest version fails

`task1_verify.py:110-116` does not integrate the gauge contraction. It evaluates the *analytic
endpoint formula* `[V·k φ]_∂γ` at λ=±2.5, which is outside the bump support, so
`product_bump` returns **exactly 0.0 at all 576 endpoints**. `gauge_col` is an identically-zero
array; gate 3 then checks that zero is small. The committed report shows it:
`J_singular_values[13] = 0.000000e+00`.

Computing what gate 3 claims to compute — `A_γ(2d^sV)` on all 288 rays:

```
order= 128   max|A| = 1.93e-05
order= 256   max|A| = 3.31e-07     <-- the order mode_matrix() actually uses
order= 512   max|A| = 3.29e-09
order=1024   max|A| = 1.21e-12     <-- first order that clears 1e-10
```

**At the quadrature order used to build J, the gauge cancellation is 3.3e-07 against a 1e-10
gate — off by 3500×.** Gate 2 passes only because it tests one favourable ray at order 1024.

This propagates: `mode_matrix` at its default `order=256` carries 3.1e-06 relative error, and
`train_selector.py` uses `order=192` (1.5e-05). Not enough to threaten rank (σ_min is 7.4e-03)
but inconsistent with 1e-10 gate language, and gratuitous — order 512 costs ~2.4 s for M=1024.

**Fix:** integrate the real thing over the whole pool, set the default order to 512+, and report
the achieved value and observed convergence order rather than pass/fail.

### B4 — The E-optimal "oracle" is not an oracle

`relaxed_e_design` parameterizes the allocation as `k·softmax(x)`. The true relaxed E-optimal
problem — maximize λ_min over `{w≥0, Σw=k, w≤1}` — is **convex**; the softmax reparameterization
makes it non-convex in `x`, so L-BFGS-B from `x0=0` finds a local optimum. Compared against a
plain 40-restart local search on exact λ_min:

```
relaxed_E          lambda_min = 1.8271   ( 7.9 s)
40x restart swap   lambda_min = 1.9650   (49.4 s)
-> relaxed_E is 7.0% BELOW a naive multi-restart search
```

`learned_gate_3` measures the policy to **within 5%** of this. The reference point is itself 7%
suboptimal, so the gate's tolerance is smaller than its reference's error.

Two additional defects in the same function: `tau = max(vals[-1]*0.02, 1e-8)` is recomputed from
the current iterate on every objective evaluation, so the objective L-BFGS-B is minimizing
**changes as the optimizer moves** — the curvature estimates and line search are operating on a
non-stationary function; and `result.success` is never checked.

**Fix:** solve the actual convex problem (SDP via cvxpy/CLARABEL, or Frank–Wolfe on exact λ_min
with projection onto the capped simplex). d=12 — this is cheap and removes the ambiguity.

---

## 3. Methodological issues

**M1 — The task distribution is degenerate for D-optimality.** `generate_tasks` randomizes an
orthogonal mixing + log scaling: `J → J·G`. But `logdet(GᵀF_S G) = logdet(F_S) + 2logdet(G)`, and
the second term is *constant across subsets S*. Measured over 200 random subsets:

```
logdet shift spread = 3.96e-12       (2*logdet G = 2.12)
```

The D-optimal ranking is **exactly invariant** under the primary task randomization. The
20,000 training tasks present essentially one D-optimal target. (Greedy still moves 0–2 of 16
picks, purely from the untransformed `1e-8` ridge.) E-optimality is genuinely randomized, since
λ_min is not invariant — so the D-policy results will look strong and demonstrate nothing about
amortization.

**M2 — Blocking is i.i.d. per ray, not angular sectors.** `blocked = rng.random(m) < U(0,0.20)`
removes a random 0–20% of rays independently. The paper and config specify *angular sectors* —
contiguous solid-angle exclusions (Sun avoidance, occultation, horizon). Random thinning leaves
angular coverage essentially intact and almost never threatens rank; sector blocking removes a
whole direction band and genuinely threatens polarization identifiability. This makes
`learned_gate_1` (99% full-rank) nearly free, and it is the physically interesting case.

**M3 — The config's task distribution is mostly unimplemented.** `experiment.yaml` registers
`physical_dimension: [6,8,12,16]`, `ray_counts: [256,512,1024,4096]`,
`transverse_offsets_per_direction: [1,4,8]`, `K_over_d: [1.0,1.25,1.5,2.0]`. `generate_tasks`
randomizes **none** of them: d=12, M=384, K from `--k`. `learned_gate_1` tests generalization
across d and K/d that was never trained.

**M4 — Gates 5 and 6 are tautologies.**
- Gate 5 computes `fisher = a²/s²` and then fits the log-log slope of `a²/s²`. Result:
  **-2.0000000000000004**. It never constructs a wave packet or computes `Var(ν̂)`. It verifies
  that `polyfit` can fit a line to a perfect line. A real gate 5 would build
  `ψ_s(t)=(2πs²)^{-1/4}exp(-t²/4s²)`, compute `Var(ν̂)` numerically, and check it equals
  `1/(4s²)` — that would actually test Corollary 3.2.
- Gate 6's integrand `f'(t)=0.2+0.2t-0.09t²` is degree 2, which Gauss–Legendre integrates
  **exactly**. Measured relative error: **0.0**, against a 1e-8 threshold. The gate cannot fail.
  Use a non-polynomial `V` so quadrature error is real, and route it through the same tensor-
  assembly path as gate 2 rather than a separately hand-differentiated function (`f` and `fp`
  are currently hardcoded independently in `light_ray.py:190-198` — edit one, forget the other).

**M5 — Gate 1 is a test of `normalize()`.** `conformal_contraction` factors `k@ETA@k` out as a
scalar, so the gate reduces to `|−1+|ϑ|²| · ∫φ` ≈ 8e-19. No cancellation along the ray is
exercised. Assembling `h_μν = φη_μν` as an actual tensor field and contracting componentwise
would test something. Also: the 1e-14 threshold is *absolute*, so it silently tracks the
amplitude of φ — make it relative to `½∫|φ|(|k⁰|²+|ϑ|²)dλ`.

**M6 — The whitening R is pool-dependent.** `design_experiment.py:168` sets R to the column norms
of the *candidate* Jacobian. That is a legitimate declared R (and the report says so), but it
changes when the pool changes — which is exactly what the `candidate_pool_density` and blocking
ablations do. Eigenvalues are therefore not commensurable across the ablations that most need
comparing. Fix R to something pool-independent (a Sobolev/energy norm on mode coefficients, or a
fixed physical normalization of the basis) and declare it once.

**M7 — `learned_gate_2` does not discriminate.** It requires beating only random and
angular_spread — the two weakest baselines. My fixed policy passes at 1.37× while being 10×
worse than greedy_D, which runs in 0.6 ms. A policy can clear the preregistered improvement gate
and still be far worse than a trivial heuristic. Since it is preregistered, don't move it — but
add greedy_D and leverage as *reported* comparators so the gate can't be read as "the policy
wins". Also specify the statistic exactly: median of the per-task **ratio** (bootstrapped over
tasks, on a log scale, λ_min being multiplicative), not the ratio of medians.

**M8 — Posterior RMSE is noisy and unpaired.** 500 Monte Carlo trials, drawn from an RNG that
`random_design` has already advanced, so each design is scored on different noise draws. Measured
spread across 30 independent streams: **sd = 0.0047**, which is **10% of the design gaps** being
reported (leverage 0.4285 vs relaxed_E 0.3817 → gap 0.0468). There is a closed form:
`E‖θ̂-θ‖² = tr((F+Λ)⁻¹)`. Analytic 0.3160 vs MC mean 0.3158 — exact agreement, zero noise, ~1000×
faster, and the fairness problem disappears. Use it.

**M9 — Two different numerical-rank conventions.** `task1_verify.consistent_weighted_ranks` uses
`τ_B²` with a roundoff floor (correct and carefully done). `train_selector.rounded_metrics` uses
`matrix_rank(f, tol=1e-9·λ_max)`. `learned_gate_1` is a rank gate — pick one convention and share it.

**M10 — Not implemented at all:** the static redshift / conformal rank-restoration experiment
(spec §6, paper §9.4) and the PTA/Hellings–Downs gate (spec §7, paper §9.5). No redshift
functional, no `R` matrix, nothing. Also promised in `metrics:` but never computed:
`worst_direction_error`, `credible_interval_coverage`, `full_rank_success_rate`,
`rank_of_J_and_F`.

**M11 — Latent, not currently biting:** nothing constrains hard top-K to *available* rays; the
selector could spend slots on blocked rays. Measured 0/800 over 50 tasks — the `log(q)` feature
is doing the job. Still worth masking logits to `-inf`, since the README claims availability
masks are an input and they are not.

**M12 — `soft_design_loss` computes `eigvalsh` unconditionally** (line 64) even on the D branch
where it is unused, and the D branch uses `torch.logdet`, which returns NaN rather than raising
on non-PD input. Prefer `slogdet` or a Cholesky.

---

## 4. Paper-side notes

- **`q` is overloaded by a factor of 4.** Continuum: eq (30)/(32) with `A=L₂/2` gives
  `q = Var(ν̂) = 1/(4s²)`. Finite-network: eq (38) uses `W = diag(s⁻²) = 4Var`. Both are
  internally correct — the §6 sentence about the factors cancelling is right — but the same
  symbol denotes two things differing by 4. The code uses `1/s²` throughout. Rename one.
- **`A_γ` is not affine-reparametrization invariant.** Under `k→k/a, λ→aλ` it scales as `1/a`.
  The `k⁰=1` normalization is therefore a physical choice tied to the receiver's clock, not a
  convention — and it is what makes the achromaticity remark (2.2) true. State it where (6) is
  defined; the code depends on it silently.
- **Table 1 gate 6 writes `[g(V,k)]`**, implying V is a vector, while eq (19) writes `[V_μk^μ]`
  with `V ∈ C_c^∞(Ω;T*M)`, a covector. The code does the covector pairing correctly
  (`covector @ k`, no index raising) — but if anyone "fixes" it to match Table 1 by raising with
  η, the time component flips sign. Make Table 1 match (19).
- **Two support conventions coexist.** Prop 4.1 needs compact interior support; Cor 5.2 needs
  `φ_j ≠ 0` at the endpoints. These are consistent — conformal invisibility is pointwise and
  needs no support hypothesis — but §9.4 will implement two families under different
  conventions, and the code should not apply one support check to both.
- **Fig. 2 vs RMSE tell different stories.** relaxed_E wins λ_min (2.45 vs 0.85) but *loses*
  RMSE to greedy_D (0.382 vs 0.309). That is correct behaviour — RMSE tracks A-optimality, which
  D-optimality approximates better than E — but the figure caption ("design changes the worst
  visible direction") should not be read as "E-optimal designs reconstruct better".

---

## 5. Running this on a Mac

All three scripts run end to end. Total wall time here: **< 15 s**. Stage 0 and Stage 1 are
comfortably laptop-scale; the *registered campaign* is not (see below).

```
run_task1.sh           0.9 s     7/7 gates pass
run_design_demo.sh     8.5 s     relaxed_E accounts for 7.2 s
run_selector_sanity.sh 5.4 s
make_figures.py        ~3 s      3 figures, PDF + PNG
```

**Setup** (Apple Silicon; use `uv` or Homebrew Python, not the system 3.9):

```bash
brew install uv
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
```

**Fix the torch pin first.** `requirements.txt` pairs `numpy>=2.0` with `torch>=2.2`, but torch
2.2 was built against NumPy 1.x and fails at import under NumPy 2. NumPy 2 support landed in
torch 2.4. Change to `torch>=2.4`.

**The `run_*.sh` scripts call `python`, not `python3`.** macOS has no `python` shim. With
`set -euo pipefail` they die with `python: command not found` unless a venv is active. Use
`"${PYTHON:-python3}"`.

**Stay on CPU — do not port this to MPS.** MPS has no float64, and the theorem gates plus every
λ_min/condition-number metric need it. The workload is 12×12 eigendecompositions; CPU is the
correct target, and the code already does the right thing by never calling `.to("mps")`. Leave it.

**macOS-specific parallelization trap.** When you scale to the multi-seed campaign you will reach
for `multiprocessing`. macOS defaults to `spawn` (Linux uses `fork`), which requires picklable
work items — and `design_experiment.py`'s `designers` dict holds **lambdas**, which do not
pickle. This works on Linux and fails only on your Mac. Use module-level functions, or
`joblib`/`concurrent.futures` with named callables. Also set `OMP_NUM_THREADS=1` per worker;
otherwise each of N workers spawns N BLAS threads and the campaign gets slower with more cores.

**Add `matplotlib.use("Agg")`** at the top of `make_figures.py` — the default macOS backend wants
the main thread and a window server.

**BLAS variation is not a risk here.** Accelerate vs OpenBLAS differ in last-bit rounding, but the
gate margins are enormous (gate 1: 8e-19 vs 1e-14; gate 2: 4e-14 vs 1e-10). Gates 1–6 will pass
identically on your Mac. The one number that will differ is gate 2's exact value.

### The registered campaign will not fit on a laptop as written

Measured scaling of the dominant cost:

```
M= 256   mode_matrix 0.56 s | greedy_D 0.044 s | relaxed_E  4.4 s
M=1024   mode_matrix 2.44 s | greedy_D 0.152 s | relaxed_E 39.3 s
```

`relaxed_E` at M=1024 is **39 s per task**. Against `test_tasks: 5000` that is **~55 hours
single-core for one configuration** — and the config lists 4 dimensions × 4 pool sizes × 3 offset
counts × 4 K/d ratios. As written this is months, not nights.

Three changes make it tractable:

1. **Rewrite `greedy_d_design` with the rank-1 update** `logdet(C+q·aaᵀ) = logdet C + log(1+q·aᵀC⁻¹a)`.
   Measured: **0.156 s → 0.0006 s, 263× faster, byte-identical subset.** (`review/` has the
   implementation.)
2. **Replace the swap-loop `relaxed_E` with the convex SDP** (B4). This fixes the oracle-quality
   problem *and* is far faster than 39 s.
3. **Use the analytic RMSE** (M8). Removes 500 Monte Carlo trials per design per task.

With those, plus `joblib` across your performance cores, a full configuration should land in
minutes. Budget the PTA/Hellings–Downs gate separately — it is the only genuinely heavy piece in
the program, and nothing in Stages 0–2 depends on it. Defer it.

---

## 6. Suggested order of work

1. Fix the `torch` pin and `python`→`python3` (5 minutes, unblocks the Mac run).
2. Fix gate 3 to integrate the real thing; raise default quadrature order to 512 (B3).
3. Make gates 1, 5, 6 non-tautological (M4, M5). Add the gauge-independence check from §1 as
   gate 8 — it is the one structural claim currently untested and it passes.
4. Build the evaluation harness: splits, baselines, bootstrap, the five gates (B2). Nothing about
   Stage 2 is measurable until this exists.
5. Fix the selector: straight-through top-K, scale-relative τ, real annealing, float64 (B1).
6. Fix the task distribution: sector blocking, and actually randomize d / M / K/d (M2, M3).
7. Replace `relaxed_E` with the convex solve (B4) and `greedy_D` with the rank-1 update.
8. Then the redshift experiment (M10). PTA last, or as a separate paper.

Steps 1–3 are a day. Steps 4–6 are where the scientific result actually lives.
