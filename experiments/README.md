# Experiments — findings log

This file is the record. It carries the measured numbers, the corrections, and
the traps. Read it before re-deriving anything.

Run with `uv run python experiments/NN_name.py`; add `--refresh` to recompute,
`--plot-only` to redraw from cache without computing anything. Experiment 01 is
float64 throughout, takes `--quick` for a smoke-sized sweep, and caches per
cell under `data/01/` — an interrupted sweep loses at most the cell it was in,
and every cache key carries its sampling budget and solver settings, so numbers
from different settings cannot collide. Experiment 02 trains in float32 and
caches the whole run under `data/02/`, keyed by its hyperparameters; its cached
file is also what `01 --real` reads.

---

## 01 — where the two approximations can be trusted

**Setup.** N=64 sites, one head, D swept over {3, 8, 16, 32, 64, 128, 256, 512},
beta over {0.5, 1, 2, 4}. Couplings are synthetic and fixed across the whole
sweep: `softmax(randn(N, N))`, row-stochastic and asymmetric — the structural
properties softmax attention has, without a trained model in the loop. Drive is
on the sphere of radius R. Sampling: 5 independent replicates x 64 chains x 400
steps after 150 burn-in.

### The two approximations get different references

They are separately controlled and must not be measured against a common
reference:

| approximation | reference | why |
|---|---|---|
| mean field | the sampled stochastic chain | it is the thing being approximated |
| large D | the **exact** mean-field expression | deterministic on both sides |

Monte Carlo cannot resolve the large-D error at the top of the D range — it has
already fallen below the sampling noise floor there. Measuring it against the
exact mean-field expression instead costs nothing and has no noise at all.

### Correction: norm-type summaries are biased upward by sampling noise

`||m||` and `||C_delayed||_F` square their argument, so Monte Carlo noise adds in
quadrature and inflates them. Measured at the **full** sampling budget:

| cell | naive `||C||_F / D` | bias-corrected | bias |
|---|---|---|---|
| beta=0.5, D=3 | 0.0274 | 0.0215 | **+27.9%** |
| beta=4, D=64 | 0.3137 | 0.3137 | +0.0% |

Uncorrected, the weak-signal cell shows a mean-field discrepancy that does not
exist. The fix is to evaluate the quadratic form across two *independent*
replicates: `<C_r, C_s>` for `r != s` has uncorrelated noise on its two legs, so
its expectation is the true squared norm with no noise term. (`pair_mean` in the
script is this estimator, once, for every corrected quantity.)

**`sigma_hk` needs no correction** and is left as the plain replicate mean:
it is weighted by `(J - J^T)^2`, whose diagonal vanishes identically, so no
site's covariance estimate is ever multiplied by itself. Measured bias +0.00% at
every point checked. Do not "fix" it to match the others.

This is why `num_repeats` has a floor of 3: the correction needs independent
pairs, and three replicates give three of them to average over rather than one.

### Correction: the noise floor needs the Bessel correction

Deviations are taken from the sample mean, so they carry one degree of freedom
less. Without the `R-1`, the floor comes out low by exactly `sqrt(R/(R-1))` and a
pure-noise measurement looks like a real error — the tell was `MF err / floor =
1.41` in *every* row of a 2-replicate run.

### Quantities are plotted dimensionless

Otherwise the D axis shows nothing but the convention `R^2 = D/2 - 1`.
Magnetizations are bounded by R, so `||m||/R` is the fraction of full
polarization; the other two are built from `Tr(Sigma_i Sigma_j)`, a sum of D
terms, so they carry one factor of D. Divided out, all three plateau in D and
the departure at small D is visible rather than buried under a power law.

### beta is not the stability knob — drive strength is

The Lipschitz bound `rho = beta/2` is the gain of the response at `h = 0`, and
the response is concave, so a site sitting in a strong field is far less
responsive than the bound allows. Local contraction at the fixed point, measured
by power iteration on `df/dm` (D=64):

| drive/R | beta=0.5 | beta=1 | beta=2 | beta=4 |
|---|---|---|---|---|
| 1.00 | 0.212 | 0.370 | 0.546 | **0.669** |
| 0.50 | 0.221 | 0.424 | 0.724 | 0.882 |
| 0.20 | 0.224 | 0.445 | 0.847 | **1.565** |
| 0.05 | 0.225 | 0.450 | 0.895 | 1.781 |
| 0.00 | 0.225 | 0.450 | 0.900 | 1.800 |

The bound is tight *only* as the drive vanishes, where the column lands on
`beta/2` exactly. At full drive the iteration converges to machine precision
even at beta=6, where the bound reads 3.0. Raising beta alone never crosses the
boundary; turning the drive down does, and at beta=4 the crossing sits between
0.5R and 0.2R.

(These numbers are from the `0.9 * softmax` couplings used while designing the
experiment; the script itself uses plain `softmax`, so its asymptotes are
`beta/2` rather than `0.45 beta`.)

### Why sigma_hk is far more accurate than the correlations it is built from

Measured: at beta=16 the delayed correlations are **59% wrong** while the
entropy production is **0.9% wrong**, across four coupling draws. This mirrors
Aguilera et al. on binary asymmetric kinetic Ising, where it is visible but not
commented on.

The exact relation is `sigma = beta sum_ij (J_ij - J_ji) C^del_ij`, which follows
from the forward/backward log-ratio of the transition kernel (normalizers and
the drive term both average to zero in the steady state). `sigma_hk` is what you
get by substituting the mean-field `C^del_ij = beta J_ij C*_ij` and
antisymmetrizing, legitimate because `C*` is symmetric.

**The first explanation was wrong.** "The mean-field error is mostly symmetric,
and sigma only reads the antisymmetric part" is true but is *not* the main
mechanism. The antisymmetric error normalized by the antisymmetric **signal**
plateaus at 0.18, not at 0.005 — if it were aligned with `J - J^T`, sigma would
be 18% wrong.

The actual mechanism is **alignment**, and it decomposes exactly:

```
dsigma/sigma = (||dC_anti|| / ||C_anti||) * cos(dC_anti, dJ) / cos(C_anti, dJ)
```

| beta | \|\|dC\|\|/\|\|C\|\| | dsigma/sigma | \|\|dC_anti\|\|/\|\|C_anti\|\| | cos(error, dJ) | cos(signal, dJ) |
|---|---|---|---|---|---|
| 0.25 | 0.63 | 0.017 | 0.72 | -0.001 | 0.68 |
| 1 | 0.24 | 0.007 | 0.30 | +0.021 | 0.95 |
| 4 | 0.45 | 0.003 | 0.18 | -0.004 | 0.98 |
| 16 | 0.59 | 0.009 | 0.18 | -0.049 | 0.98 |

- The **signal** is aligned with `J - J^T` almost by construction:
  `antisym(beta J C*)_ij = (beta/2)(J_ij - J_ji) C*_ij` is that very matrix
  modulated by a symmetric weight. Measured `cos -> 0.98`.
- The **error** has no systematic component along it. Measured `|cos| < 0.05`,
  against `1/sqrt(N(N-1)/2) = 0.022` for a directionless error at N=64.

Checks out numerically: `0.18 * 0.05 / 0.98 = 0.009` against a measured 0.0089.

**So the protection is a projection onto one direction out of N(N-1)/2, where
the signal lives by construction and the error does not.** It should therefore
*strengthen with N* like `1/sqrt(N)`. The `--only scaling` sweep measures this
over N in {16, 32, 64, 128, 256}; at smoke budget the error's `|cos|` tracks
the directionless reference `sqrt(2/N(N-1))` across the whole range (0.085 at
N=16 down to below 1e-3 at N=256) while the signal alignment stays at 0.84-0.99,
and `dsigma/sigma` falls with N as predicted. Full-budget confirmation pending.

Downstream consequences:

- **sigma_hk is licensed well outside the Plefka regime**, which is what makes it
  usable as a live diagnostic during training.
- **But it is a bad validator of its own approximation.** "sigma_hk matches
  sampling" does not imply "mean field is working" — it is one number projected
  onto the one direction the error avoids. Anyone monitoring sigma_hk to check
  the approximation's health is reading the least informative possible statistic.
- **The mismatch proxy has no such protection.** It is a KL between single-site
  laws and is sensitive to exactly the magnetization error that is visible. The
  two diagnostics have different trust profiles; reporting both is doing more
  work than it looks like.
- **Accuracy of the value does not transfer to the gradient.** If sigma_hk enters
  an objective, `d sigma_hk / d theta` is protected only if the error stays
  misaligned as theta moves. Unverified, and the failure would be silent.

**Open risk.** The couplings here are `softmax(randn)` — structureless. Real
attention is not: if the mean-field error and the attention asymmetry come to
share structure (both dominated by positional locality, say), the alignment
could stop being zero. The bridge for testing this is wired: experiment 02
caches its trained probe couplings and drives, and
`01 --real experiments/data/02/<run>.pt` reruns the alignment measurement on
head 0 of every layer. Run it on a fully trained model before this goes in the
post — the only run so far is a 30-step wiring check, which says nothing.

### Correction: the mean-field error was never measured, only the noise floor

`||m_exact - m_pooled|| / ||m_pooled||` is biased upward by sampling noise in
exactly the way the norm summaries are. In the smoke sweep it agreed with the
Monte Carlo floor to a few percent at **every** beta, so the entire mean-field
column was a picture of the sampling budget rather than of the approximation.
The apparent "error improves as beta rises" was the floor falling: `||m||/R`
grows 0.24 -> 0.95 with beta while fluctuations get suppressed, so relative
noise drops on both counts.

Same fix as everywhere else. With `m_r = m_true + n_r`, the product
`<ref - m_r, ref - m_s>` for `r != s` has independent noise on its two legs, so
the cross terms vanish in expectation and what is left is the squared error with
no floor added. It can come out negative when the true error is far below the
floor, which is honest; clamp only at the last step.

Noise-corrected magnetization error, D=64, full budget:

| beta | drive | \|\|m\|\|/R | naive | floor | **corrected** |
|---|---|---|---|---|---|
| 0.25 | 1.0R | 0.120 | 0.0230 | 0.0231 | **0.0000** |
| 1 | 1.0R | 0.410 | 0.0081 | 0.0064 | 0.0049 |
| 4 | 1.0R | 0.792 | 0.0091 | 0.0023 | **0.0088** |
| 16 | 1.0R | 0.947 | 0.0035 | 0.0011 | 0.0033 |
| 0.25 | 0.2R | 0.024 | 0.1163 | 0.1153 | 0.0150 |
| 1 | 0.2R | 0.098 | 0.0309 | 0.0294 | 0.0093 |
| 4 | 0.2R | 0.351 | 1.0766 | 0.0104 | **1.0766** |
| 16 | 0.2R | 0.936 | 0.0243 | 0.0048 | 0.0238 |

**The corrected error is non-monotonic in beta, peaking near beta ~ 4.** Both
intuitions are right in different regimes: it grows because Plefka is a
beta-expansion, then falls again because `||m||/R -> 0.95` and everything pins
to the drive, leaving nothing to fluctuate. At beta=0.25 it is exactly zero
within resolution, as it must be.

### The right knobs are `u = beta ||x|| / R` and `beta`, and D is not one of them

**Derivation, and how it relates to the post.** The post gives the response and
the stiffness as

    m = beta h / (1 + gamma_h),    gamma_h = sqrt(1 + beta^2 ||h||^2 / R^2)
    R^2 = D/2 - 1,                 kappa_h = beta R ||h||

which is `vmf.response_large_d` and `vmf.gamma` verbatim. Where that comes from
in turn: the exact response is `m = R A_D(kappa) h_hat`, and
`mean_resultant_large_d` is `A_D(kappa) ~ (kappa/R^2) / (1 + sqrt(1 +
(kappa/R^2)^2))`. Substituting, and using `kappa/R^2 = beta ||h|| / R`,

    m = R (beta||h||/R) / (1 + sqrt(1 + beta^2||h||^2/R^2)) h_hat = beta h / (1 + gamma)

so `response_large_d` follows from `mean_resultant_large_d`, with
`gamma = sqrt(1 + (kappa/R^2)^2)`.

Now non-dimensionalize. Put `mu = m/R` and `xi = x/R`, so `h = R(xi + J mu)` and
`||h||^2/R^2 = ||xi + J mu||^2` — R cancels *inside* gamma. Then
`mu = beta(xi + J mu)/(1 + gamma)`, and setting `v = beta(xi + J mu)` gives

    mu = v / (1 + sqrt(1 + ||v||^2)),      v = beta xi + beta J mu

`R` cancels and **D disappears entirely**. This is not new dynamics — it is the
post's own equation with the length unit divided out — but it makes visible that
`R` appears *only* as the unit of length, so `D` is not a parameter of the
limiting problem at all. With `||xi|| = a` (drive in multiples of R), the drive
enters only through `u = a beta`, so the whole large-D problem is two scalars:
`u` (pinning) and `beta` (coupling).

`u` is `kappa` measured in units of the Bessel order `R^2`. The codebase already
named it: `bessel_ratio`'s docstring says convergence is set by `u = x/order`,
"for us `x = kappa = beta R ||h||` and `order = R^2`, so `u = beta ||h|| / R`".

Verified — fixed point in units of R, at fixed `(a, beta)`:

| a | beta | u | D=8 | D=32 | D=128 | D=512 |
|---|---|---|---|---|---|---|
| 1.0 | 1 | 1.0 | 0.4184 | 0.4188 | 0.4177 | 0.4179 |
| 1.0 | 4 | 4.0 | 0.7929 | 0.7976 | 0.7963 | 0.7963 |
| 0.2 | 4 | 0.8 | 0.3579 | 0.3599 | 0.3578 | 0.3596 |
| 0.05 | 8 | 0.4 | 0.2190 | **0.8671** | 0.2215 | **0.6297** |

The last row does not collapse, and that is the point: it is multistable, so
different drive realizations fall into different basins. **The collapse holds
exactly where the fixed point is unique, and the scatter is the order parameter.**

The single-site saturation law `||m||/R = u/(1+sqrt(1+u^2))` predicts the
measured fixed points directly: `u=0.8 -> 0.3508` (measured 0.3513),
`u=4 -> 0.7808` (0.792), `u=16 -> 0.9395` (0.947).

Consequence for experiment design: **sweeping beta at fixed full drive walks the
diagonal `u = beta`**, moving pinning and coupling together. It cannot separate
them. The plane to sweep is `(u, beta)`, and the interesting corner -- large
beta, small `u` -- is the one the current sweep never visits.

### Multistability, and the fourth quadrant

At beta=4, drive=0.2R, D=64, solving from six different starts:

| start | residual | \|\|m\|\|/R |
|---|---|---|
| zeros | 2.1e-14 | 0.3513 |
| random 1-4 | ~1e-14 | 0.3513 |
| random 5 | 8.0e-15 | **0.7110** |

Two genuine fixed points, 33.9 apart, both converged to 1e-14. Local
contraction at the zeros branch is **rho = 1.702**, so that branch is
*unstable*, and the sampled chain sits at `||m||/R = 0.6328` -- near the other
one. The 108% "mean-field error" in the table above is therefore **branch
selection, not approximation failure**.

This is the concrete existence proof for the fixed-point-with-carried-state
quadrant of the design table. Round 1 concluded the unique fixed point was
probably a toy-scale artifact and noted that while it holds, that row of the
table is one cell -- at `K -> inf` the initialization is provably inert and
`to_v` gets no gradient. Here the initialization is **not** inert: it selects
the branch. The quadrant is real, and it lives at large beta and small `u`.

**New gotcha: Anderson solves the root-finding problem, so it converges to
repelling fixed points as happily as attracting ones.** A clean residual is not
evidence that the point is physical. Beyond the contraction boundary the map
goes multistable and any reported "mean-field error" silently becomes a
statement about which basin the solver fell into. Report the local rho and a
branch check beside every fixed point, not just the residual.

### Relation to the post's stability condition

The post states `rho_t = beta J_t / 2 < 1` as a **sufficient** condition for a
unique fixed point, `rho_t = beta/2` for row-stochastic softmax attention, and
that "when `rho_t >= 1`, convergence becomes uncertain and multiple branches may
coexist". The multistability above is a direct empirical confirmation of that
sentence. Two refinements:

- **`beta/2` is evaluated at `h = 0`, so it is loose wherever the drive pins.**
  At beta=4, full drive, the bound reads 2.0 and the measured contraction is
  0.669; at beta=16 it is still only ~0.80. The post says *sufficient*; the
  measurement says how much slack the drive buys.
- **The boundary is a curve in `(u, beta)`, not a value of beta.** `beta = 2` is
  its `u -> 0` asymptote. The finite-D exact map gives `beta_c(D) = 2D/(D-2)`
  (6.0 at D=3, 2.008 at D=512), so the post's large-D statement and the code's
  `contraction_factor` agree in the limit and differ at small D.

### The figure this experiment should produce

The three original figures (fidelity, errors, contraction) each showed one slice
of what is really a two-parameter problem, and the fidelity sweep walked the
diagonal `u = beta` without saying so. They are replaced by a phase diagram, one
per quantity:

- axes **`u` (pinning) by `beta` (coupling)** at fixed D, one panel per
  D in {3, 16, 64};
- **top row: colour = noise-corrected mean-field error**, i.e. the *exact*
  mean-field fixed point measured against the sampled chain. Large-D plays no
  part in it;
- **bottom row = large-D versus exact mean field** on the same axes, which is
  deterministic and therefore free, drawn on a finer grid, and shrinks left to
  right as D grows. (It began as an inset; at 38% it covered the low-beta,
  high-u corner, precisely the region where the approximation works and a
  reader most needs to see);
- **overlaid**: the measured `rho = 1` contour, the post's `beta = 2` asymptote,
  and cells flagged where a multi-start solve finds more than one branch.

That carries the three results at once — where the approximation can be
trusted, where the map stops contracting, and where the fourth quadrant lives —
with the post's sufficient condition drawn on top.

### Figures

- `01_phase_magnetization`, `01_phase_delayed`, `01_phase_entropy` — the phase
  diagrams above, one per quantity: sampled mean-field error on top, large-D
  error below, `rho = 1` contour, `beta_c`, and per-cell caveat flags.
- `01_entropy_robustness` — why sigma_hk outlives the correlations it is built
  from: relative errors on the left, the alignment mechanism on the right.
- `01_alignment_scaling` — the projection protection against N, with the
  `sqrt(2/N(N-1))` directionless reference.

(An earlier design — `01_fidelity` / `01_errors` / `01_contraction`, curves
against D at fixed full drive — was replaced by the phase diagrams: it walked
the diagonal `u = beta` without saying so.)

---

## 02 — is a stack of these things an ordinary trainable network?

Wired and smoke-tested; **no full training run recorded yet**, so this section
carries mechanics only. The script trains a finite-K=1, amortized-init,
causal stack on character-level next-token prediction, prints the control room
(field decomposition, per-layer gradients, embedding geometry) every
`--log-every` steps, and writes `02_training`, `02_signal`, and
`02_field_balance`. The whole run — history, n-gram baselines, and the trained
probe couplings for `01 --real` — is cached under `data/02/<run-tag>.pt`;
`--plot-only` with the same flags redraws without training.

---

## Traps that cost real time

- **Never mix exact and `_large_d` expressions inside one formula.** They differ
  at O(1) in places that look harmless. Each cell computes its fixed point and
  all three quantities in one consistent flavour, never a hybrid.
- **`R^2 = D/2 - 1`**, so D must be > 2. D=3 is the smallest usable, is fine for
  visualizing exact mean-field dynamics, and is useless for large-D diagnostics.
- **The relevant D is `dim_head`, not `dim`.** Each head is an independent spin
  system.
- **`scipy.special.ive` underflows where our `bessel_ratio` does not.** At D=512,
  x=0.01 both Bessels flush to zero and scipy's ratio is `0/0`; the backward
  recurrence returns the correct `x/D`. Do not "fix" our version to match scipy.
- **`sigma_hk` does not decay.** It is the housekeeping cost of holding a
  nonequilibrium steady state and stays positive at the fixed point for
  asymmetric J. Only the mismatch relaxes. Symmetric J sends it to exactly zero.
- **A converged residual does not mean a physical fixed point.** Anderson solves
  the root-finding problem and finds repelling points too. Always report the
  local contraction beside the residual.
- **And the converse: rho without a residual is meaningless.** Measured at a
  point the solver never reached, the power iteration returns `rho ~ 1.000` for
  *every* beta, because the iterate is wandering rather than sitting anywhere.
  That drew a spurious closed `rho = 1` contour across the top of the phase
  plane, which looked like real physics -- an instability bounded above by
  self-pinning. It was not. At D=64, u=0.125 the honest picture is:

  | beta | rho | residual | converged |
  |---|---|---|---|
  | 1 - 6 | 0.483 -> 2.894 | ~1e-15 | yes |
  | 8 | 3.853 | 1.1e-01 | **no** |
  | 12 - 32 | 0.999 -> 1.000 | ~1e-03 | **no** |

  Where the solve converges, rho is monotone in beta and tracks `beta/2` closely
  (2.894 against 3.0 at beta=6). Mask rho wherever the residual misses tolerance,
  or the contour is drawn through points that are not fixed points. The right-hand
  side of the contour *is* real: at beta=16, u >= 1.26 the solve converges and
  rho = 0.97-0.99, so strong pinning genuinely restores contraction.
- **A string replacement that silently does not match leaves the old behaviour
  in place.** The mask above was "added" once without taking effect; the tell was
  a residual column of exactly `0.0e+00` everywhere. Check the field you just
  wrote actually contains what you think.
- **Every Monte Carlo comparison needs the cross-replicate correction.** Norms,
  errors against a reference, anything quadratic in the samples. Three separate
  quantities in this experiment were pure sampling noise until it was applied,
  and each one looked like a physical result.
- **`beta` and drive strength are not independent knobs.** The large-D problem
  depends only on `u = beta ||x||/R` and `beta`. Sweeping beta at fixed drive
  moves both at once along `u = beta`.
- **Never paint an unresolved cell at the bottom of a colour scale.** A cell
  whose error falls below its own noise floor has not been measured. Rendered as
  the lightest colour it reads as "1e-3", when the floor underneath it may be
  1e-1 -- two orders of magnitude of precision the sampling never had. Mask them
  to a neutral fill with its own legend entry, and carry a floor *per quantity*,
  since the magnetization floor does not describe the other two. At smoke budget
  this turns most of the phase plane grey, which is the correct picture: the
  only cells with resolved error are the ones past the contraction boundary,
  where the comparison is ill-posed for other reasons anyway.
