"""Experiment 01 -- where the two approximations can be trusted.

The forward pass rests on two stacked approximations, and they are controlled by
different knobs, so they are measured against different references:

- **mean field** replaces the fluctuating spins of the neighbours by their
  averages.  Its reference is the stochastic chain itself, sampled.  Plefka is
  an expansion in beta, so this is the one that should degrade as beta grows.
- **large D** drops the Bessel functions for closed forms.  Its reference is the
  exact mean-field expression, evaluated in the same place.  This one is
  deterministic on both sides, which matters: Monte Carlo cannot resolve it at
  the top of the D range, where the large-D error has already fallen below the
  sampling noise floor.

Mixing those references would produce a number that is a controlled limit of
neither, which is the same trap as mixing exact and large-D expressions inside
one formula.

Once lengths are measured in units of R the large-D problem has exactly two
parameters -- ``u = beta ||x|| / R`` for pinning and ``beta`` for coupling -- and
D is not one of them.  So the sweep is a phase plane in ``(u, beta)`` with D as
the panel axis, rather than a set of curves against D.  Sweeping beta at fixed
drive, as an earlier version did, walks the diagonal ``u = beta`` and cannot
separate the two.

``01_phase_{quantity}`` -- one per quantity, two rows:

- **top**, the mean-field error: the exact mean-field fixed point against the
  sampled chain, with the Monte Carlo noise divided out.  Large D plays no part.
- **bottom**, the large-D error against that same exact mean field.  Both sides
  are deterministic, so it costs nothing and is drawn on a finer grid.

Overlaid on both: the measured ``rho = 1`` contour, the post's sufficient
condition ``beta_c`` as its ``u -> 0`` asymptote, and per-cell flags for
multistability, non-convergence, and ergodicity breaking -- past the contraction
boundary a reported "error" can be branch selection or an average over
degenerate ordered states rather than an approximation failure.

``01_entropy_robustness`` -- why sigma survives where the correlations it is
built from do not.  ``01_alignment_scaling`` -- the protection is a projection
onto one direction out of N(N-1)/2, so it should strengthen like 1/sqrt(N);
this measures that.

Couplings are synthetic: row-stochastic and asymmetric, the structural
properties softmax attention has, without a trained model in the loop.  Real
couplings from a trained experiment-02 run are read back with
``--real experiments/data/02/<run>.pt``, which reruns the robustness
measurement on them -- the open risk is that trained attention and the
mean-field error come to share structure, which would break the alignment
protection that structureless couplings enjoy.

Run with ``uv run python experiments/01_approximation_fidelity.py``.  ``--quick``
takes a smoke-sized sweep, ``--refresh`` recomputes cached cells, and
``--plot-only`` redraws from cache without computing anything, which is how the
figures should be tuned -- a full redraw is nine seconds against six minutes.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy
import torch

import common
from neqnn import fixed_point as fp, mean_field as mf, proxies, stochastic, vmf

SITES = 64

# The phase plane.  The large-D problem has exactly two parameters once lengths
# are measured in units of R -- u = beta ||x||/R for pinning and beta for
# coupling -- and D is not one of them, so D becomes the panel axis rather than
# a swept variable.  Three values are enough to show the large-D error collapsing
# left to right: it runs about 1.5 at D=3, 0.09 at D=16, 0.02 at D=64.
PHASE_DIMS = [3, 16, 64]
U_VALUES = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
# Resolution of the deterministic overlay grid (contraction, branches, large-D
# error).  It needs no sampling, so it can be far finer than the sampled cells
# and gives a smooth rho = 1 contour.
FINE = 13
# Log-symmetric about beta_c = 2: offsets of -3 .. +3 powers of two.  Sampling
# symmetrically around the stability scale is the honest way to ask whether the
# approximations fail on one side of it.
BETAS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]

# The m = 0 fixed point loses stability where the gain of the response at h = 0
# exceeds one.  For row-stochastic J that is beta (D/2-1)/D = 1, so
# beta_c = 2D/(D-2): 6.0 at D=3, 2.07 at D=64, 2.008 at D=512, and 2 in the
# limit.  This is *not* a critical point of the driven system -- the drive is an
# external field and smears the transition, which is what figure 3 shows -- but
# it is the natural scale to read beta against.
critical_beta = lambda dim: 2 * dim / (dim - 2)

# Enough samples to put the noise floor below the mean-field error at beta >= 1,
# which is the comparison that has to be resolved.  At beta = 0.5 the mean-field
# error falls under the floor, and that is itself the finding.
SAMPLING = dict(num_repeats=5, num_chains=64, num_steps=400, burn_in=150)
# Three is the floor, not a taste: the bias correction needs an independent
# pair, and three replicates give three pairs to average over rather than one.
QUICK = dict(num_repeats=3, num_chains=16, num_steps=60, burn_in=30)

QUANTITIES = ("magnetization", "delayed", "entropy")

# Solver settings behind every deterministic cell.  They are folded into the
# cache keys below: numbers computed under different settings must not collide
# on disk, which a hand-bumped version suffix only papers over.
SOLVE = dict(max_iter=80, tol=1e-12)
BRANCH = dict(max_iter=200, tol=1e-13, residual_tol=1e-8)
POWER_ITERATIONS = 60

# The sites axis of the alignment-scaling sweep.  The protection is a projection
# onto one direction out of N(N-1)/2, so |cos(error, dJ)| for a directionless
# error should fall like 1/sqrt of that count.
SCALING_SITES = [16, 32, 64, 128, 256]


def solve_tag(starts: int) -> str:
    """Deterministic-solver provenance, for cache keys."""
    return (
        f"s{starts}_fp{SOLVE['max_iter']}-{SOLVE['tol']:g}"
        f"_br{BRANCH['max_iter']}-{BRANCH['tol']:g}_pi{POWER_ITERATIONS}"
    )


def problem(dim: int, seed: int = 0, sites: int = SITES):
    """A fixed drive on the sphere and a fixed row-stochastic asymmetric coupling.

    ``couplings`` is drawn from the same seed at every D so the D axis moves one
    thing only.  It cannot depend on D anyway -- it is N x N -- but seeding it
    separately makes that explicit.
    """
    torch.manual_seed(seed)
    couplings = torch.softmax(torch.randn(sites, sites), -1)
    torch.manual_seed(seed + 1)
    drive = stochastic.random_state((sites,), dim)
    return drive, couplings


def summarize(tensors: dict) -> dict:
    """Collapse each quantity to one scalar for consoles and axes.

    Kept beside the tensors rather than replacing them: errors are always formed
    on the full tensors, since a summary can agree by cancellation where the
    underlying fields do not.

    Each is made dimensionless, because otherwise the D axis shows nothing but
    the convention R^2 = D/2 - 1.  Magnetizations are bounded by R, so ||m||/R is
    the fraction of full polarization; the other two are built from
    Tr(Sigma_i Sigma_j), which is a sum of D terms and so carries one factor of D.
    Divided out, all three plateau in D and the departure at small D is visible
    rather than buried under a power law.
    """
    dim = tensors["magnetization"].shape[-1]
    return {
        "magnetization": float(tensors["magnetization"].norm(dim=-1).mean())
        / vmf.radius(dim),
        "delayed": float(tensors["delayed"].norm()) / dim,
        "entropy": float(tensors["entropy"]) / dim,
    }


def pair_mean(
    values: torch.Tensor, indices: list[int] | None = None, *, dim=None
) -> torch.Tensor:
    """Mean over unordered pairs r != s of ``(values[r] * values[s]).sum(dim)``.

    The estimator behind every noise-corrected number in this experiment.  A
    squared quantity formed on one replicate absorbs its own Monte Carlo noise
    in quadrature and is biased upward; formed *across* two independent
    replicates the noise on the two legs is uncorrelated, the cross terms vanish
    in expectation, and what is left is the true squared quantity with no noise
    term.  ``dim=None`` contracts everything to a scalar; ``dim=-1`` keeps the
    site axis.
    """
    if indices is None:
        indices = list(range(len(values)))
    pairs = list(itertools.combinations(indices, 2))
    products = [
        (values[r] * values[s]).sum() if dim is None else (values[r] * values[s]).sum(dim)
        for r, s in pairs
    ]
    return torch.stack(products).mean(0)


def cross_summary(replicates: dict, indices: list[int]) -> dict:
    """The same summaries, but with the Monte Carlo noise bias taken out.

    Two of the three are norms, and a norm squares its argument, so sampling
    noise biases them *upward* -- badly wherever the signal is weak.  Measured
    at full budget: +27.9% at beta=0.5, D=3, which is large enough to
    manufacture a mean-field discrepancy that is not there.  ``pair_mean`` is
    the fix.

    ``sigma_hk`` needs no such treatment and is left as the plain replicate mean:
    it is weighted by ``(J - J^T)^2``, whose diagonal vanishes identically, so no
    site's covariance estimate is ever multiplied by itself.  Measured bias
    +0.00% at every point checked, which is why it is the one left alone.
    """
    dim = replicates["magnetization"].shape[-1]
    squared = pair_mean(replicates["magnetization"], indices, dim=-1)
    delayed = pair_mean(replicates["delayed"], indices)
    return {
        "magnetization": float(squared.clamp_min(0).sqrt().mean()) / vmf.radius(dim),
        "delayed": float(delayed.clamp_min(0).sqrt()) / dim,
        "entropy": float(replicates["entropy"][indices].mean()) / dim,
    }


def relative_floor(replicates: torch.Tensor, pooled: torch.Tensor) -> float:
    """Standard error of the pooled estimate, relative -- the resolution limit.

    The ``R-1`` is not cosmetic: deviations are taken from the sample mean, so
    they carry one degree of freedom less, and without it the floor comes out low
    by ``sqrt(R/(R-1))`` and a pure-noise measurement looks like a real error.
    """
    repeats = len(replicates)
    scale = pooled.norm()
    if float(scale) <= 0:
        return float("inf")
    # reshape, not flatten(1): the entropy replicates are scalars, so the stack
    # is one-dimensional and there is no axis 1 to flatten from.
    spread = (
        (replicates - pooled)
        .reshape(repeats, -1)
        .norm(dim=-1)
        .pow(2)
        .sum()
        .div(repeats - 1)
        .sqrt()
    )
    return float(spread / repeats**0.5 / scale)


def corrected_error(reference: torch.Tensor, replicates: torch.Tensor) -> float:
    """Relative error of ``reference`` against the *true* mean of ``replicates``.

    The naive ``||reference - pooled|| / ||pooled||`` is biased upward by the
    sampling noise in exactly the way the norm summaries were, and by enough to
    swamp the thing being measured: at every beta in the smoke sweep it agreed
    with the Monte Carlo floor to a few percent, so the whole mean-field column
    was a picture of the noise rather than of the approximation.

    Same fix as everywhere else.  Writing ``m_r = m_true + n_r``, the product
    ``<reference - m_r, reference - m_s>`` for r != s carries independent noise on
    its two legs, so the cross terms vanish in expectation and what is left is
    the squared error with no noise floor added.  It can come out slightly
    negative when the true error is far below the floor, which is honest and is
    clamped only at the last step.

    Returns NaN when the denominator is not positive.  That is not a numerical
    guard but a statement: it means the sampled quantity averages to nothing, so
    there is no scale to measure a relative error against.  It happens in the
    ergodicity-broken corner, where every chain magnetizes but in its own
    direction and the pooled average cancels.  Silently clamping it would report
    a huge error where the right answer is "this comparison is not defined here".
    """
    numerator = pair_mean(reference - replicates)
    denominator = pair_mean(replicates)
    if float(denominator) <= 0:
        return float("nan")
    return float((numerator.clamp_min(0) / denominator).sqrt())


def mean_field_cell(drive, couplings, beta, *, large_d: bool):
    """Fixed point and the three quantities read off it, in one consistent flavour."""
    step = mf.step_large_d if large_d else mf.step
    traces = mf.covariance_traces_large_d if large_d else mf.covariance_traces
    step_fn = lambda m: step(m, drive, couplings, beta)

    solve = fp.anderson(step_fn, torch.zeros_like(drive), **SOLVE)
    field = mf.effective_field(solve.solution, drive, couplings)
    # At the fixed point the two consecutive fields coincide, which is exactly
    # the steady-state form of the delayed correlation.
    covariance_traces = traces(field, field, beta)
    delayed = beta * couplings * covariance_traces
    tensors = {
        "magnetization": solve.solution,
        "delayed": delayed,
        # Identical to housekeeping_entropy_production(couplings, traces, beta),
        # but routed through the exact relation so both sides of the comparison
        # apply the same functional and only the delayed correlations differ.
        "entropy": proxies.entropy_production(couplings, delayed, beta),
    }
    return {
        "tensors": tensors,
        "values": summarize(tensors),
        "residual": solve.residual,
    }


def sampled_cell(drive, couplings, beta, *, references, num_repeats, **sampling):
    """Independent Monte Carlo replicates, so every quantity gets an honest error bar.

    Repeating the whole estimate rather than keeping a chain axis inside it is
    what keeps the memory flat: the per-site covariances are N x D x D and only
    one replicate's worth is ever alive.  The spread across replicates is then
    the error bar for all three quantities uniformly, including the two that are
    nonlinear in the samples and have no closed-form standard error.
    """
    if num_repeats < 3:
        raise ValueError(
            f"num_repeats must be at least 3 so the cross-replicate correction "
            f"averages over more than one pair, got {num_repeats}"
        )
    magnetizations, delayed, entropy, chain_magnitudes = [], [], [], []
    for repeat in range(num_repeats):
        torch.manual_seed(1000 + repeat)
        estimates = stochastic.estimate(drive, couplings, beta, **sampling)
        magnetizations.append(estimates.magnetizations)
        chain_magnitudes.append(estimates.chain_magnetizations)
        delayed.append(estimates.delayed_correlations)
        # The exact relation on sampled delayed correlations, not the mean-field
        # formula fed sampled covariances.  The latter is nearly tautological --
        # it shares the functional with the thing it is meant to test -- and is
        # measurably worse besides: 50.12 against a truth of 47.90 at beta=4,
        # where this route gives 47.85.
        entropy.append(
            proxies.entropy_production(couplings, estimates.delayed_correlations, beta)
        )
    replicates = {
        "magnetization": torch.stack(magnetizations),
        "delayed": torch.stack(delayed),
        "entropy": torch.stack(entropy),
    }
    values = cross_summary(replicates, list(range(num_repeats)))

    pooled = {name: values_.mean(0) for name, values_ in replicates.items()}
    # Errors against the mean-field predictions are formed here, where the
    # replicate stack is still alive, so they can be noise-corrected the same
    # way the summaries are.  Caching only the pooled tensors would leave the
    # naive, floor-contaminated comparison as the only one available later.
    mean_field_error = {
        name: corrected_error(references["tensors"][name], replicates[name])
        for name in QUANTITIES
    }
    pooled_magnitude = pooled["magnetization"].norm(dim=-1).mean()
    return {
        "tensors": pooled,
        "mean_field_error": mean_field_error,
        # Two numbers, not one.  The ratio alone cannot tell ergodicity breaking
        # from noise: where the magnetization is tiny, both legs are noise and the
        # ratio is large for no physical reason.  Breaking needs the chains to
        # actually be magnetized *and* to disagree about the direction.
        "ergodicity": float(
            torch.stack(chain_magnitudes).mean(0).mean()
            / pooled_magnitude.clamp_min(torch.finfo(pooled_magnitude.dtype).tiny)
        ),
        "chain_saturation": float(torch.stack(chain_magnitudes).mean(0).mean())
        / vmf.radius(drive.shape[-1]),
        "values": values,
        # Noise floor: the standard error of the pooled estimate, relative --
        # the resolution limit of every dashed curve in figure 2.  The R-1 is not
        # cosmetic: deviations are taken from the sample mean, so they carry one
        # degree of freedom less, and without it the floor comes out low by
        # sqrt(R/(R-1)) and a pure-noise measurement looks like a real error.
        "floor": relative_floor(replicates["magnetization"], pooled["magnetization"]),
        # One per quantity, not one for all three.  A cell whose error sits below
        # its own floor has not been measured, and rendering it at the bottom of
        # the colour scale claims a precision the sampling does not have.
        "floors": {
            name: relative_floor(replicates[name], pooled[name]) for name in QUANTITIES
        },
    }


def sampling_tag(sampling: dict) -> str:
    """Fold the sampling budget into the cache key.

    Without this a ``--quick`` cell and a full-budget cell collide on disk, and
    the next full run silently reuses smoke-test numbers -- the figures would
    redraw, the console would print, and nothing would look wrong.  Provenance
    has to be part of the key, not a thing to remember.
    """
    return "r{num_repeats}c{num_chains}s{num_steps}b{burn_in}".format(**sampling)


def alignment_cell(drive, couplings, beta, sampling, *, seed_base: int) -> dict:
    """Why sigma survives a regime where the delayed correlations do not.

    ``sigma = beta sum (J_ij - J_ji) C_ij`` weights ``C`` by an antisymmetric
    matrix, so it reads only the antisymmetric part of the delayed correlations.
    Two separable things could therefore protect it, and reporting one without
    the other confounds them:

    - the mean-field error could be mostly *symmetric*, and so invisible;
    - the antisymmetric remainder could be poorly *aligned* with ``J - J^T``.

    So both are measured, and the antisymmetric error is normalized by the
    antisymmetric signal rather than by the whole matrix -- otherwise the
    fraction has no scale.

    Takes the drive and couplings rather than drawing them, so the same
    measurement runs on synthetic ``softmax(randn)`` draws and on trained
    couplings exported from experiment 02.
    """
    sites = couplings.shape[-1]
    asymmetry = couplings - couplings.T
    predicted = mean_field_cell(drive, couplings, beta, large_d=False)

    sampled_delayed = []
    for repeat in range(sampling["num_repeats"]):
        torch.manual_seed(seed_base + repeat)
        estimates = stochastic.estimate(
            drive,
            couplings,
            beta,
            num_chains=sampling["num_chains"],
            num_steps=sampling["num_steps"],
            burn_in=sampling["burn_in"],
        )
        sampled_delayed.append(estimates.delayed_correlations)
    sampled = torch.stack(sampled_delayed).mean(0)

    antisymmetric = lambda t: 0.5 * (t - t.transpose(-1, -2))
    gap = predicted["tensors"]["delayed"] - sampled
    gap_symmetric = 0.5 * (gap + gap.T)
    gap_antisymmetric = antisymmetric(gap)

    sigma_predicted = float(predicted["tensors"]["entropy"])
    sigma_sampled = float(proxies.entropy_production(couplings, sampled, beta))
    return {
        "delayed_error": float(gap.norm() / sampled.norm()),
        "entropy_error": abs(sigma_predicted - sigma_sampled) / abs(sigma_sampled),
        "symmetric": float(gap_symmetric.norm()),
        "antisymmetric": float(gap_antisymmetric.norm()),
        "antisymmetric_fraction": float(gap_antisymmetric.norm() / gap.norm()),
        # The one that carries scale: antisymmetric error against antisymmetric
        # signal.  If sigma is protected purely by symmetry this tracks the
        # entropy error; if it is protected by misalignment it sits well above it.
        "antisymmetric_relative": float(
            gap_antisymmetric.norm() / antisymmetric(sampled).norm()
        ),
        "alignment": float(
            (gap_antisymmetric * asymmetry).sum()
            / gap_antisymmetric.norm()
            / asymmetry.norm()
        ),
        # The other half of the story.  The *signal* is aligned with J - J^T
        # almost by construction: antisym(beta J C*)_ij = (beta/2) (J_ij - J_ji)
        # C*_ij is that very matrix modulated by a symmetric weight.  So sigma
        # keeps nearly all of the signal while the error, which has no systematic
        # component along that one direction, is projected away.
        "signal_alignment": float(
            (antisymmetric(sampled) * asymmetry).sum()
            / antisymmetric(sampled).norm()
            / asymmetry.norm()
        ),
        # What |cos| an error with no preferred direction would show: sigma reads
        # one direction out of the N(N-1)/2 antisymmetric ones, so the protection
        # is a 1/sqrt(N) effect and should strengthen with system size.
        "random_alignment": (2.0 / (sites * (sites - 1))) ** 0.5,
        "sigma_ratio": sigma_predicted / sigma_sampled,
    }


def robustness_cell(dim, beta, seed, sampling, *, sites: int = SITES) -> dict:
    """One synthetic coupling draw through the alignment measurement."""
    drive, couplings = problem(dim, seed=seed, sites=sites)
    return alignment_cell(
        drive, couplings, beta, sampling, seed_base=2000 + 17 * seed
    )


ROBUSTNESS_COLUMNS = {
    "dC/C": 8,
    "dsigma/sigma": 12,
    "anti/Canti": 11,
    "cos(err,dJ)": 12,
    "cos(sig,dJ)": 12,
    "random": 8,
}


def robustness_row(console, label: str, cells: list[dict]) -> None:
    mean = lambda key: sum(cell[key] for cell in cells) / len(cells)
    console.row(
        label,
        mean("delayed_error"),
        mean("entropy_error"),
        mean("antisymmetric_relative"),
        mean("alignment"),
        mean("signal_alignment"),
        cells[0]["random_alignment"],
    )


def robustness_sweep(sampling: dict, refresh: bool, *, dim: int = 64) -> dict:
    """The same measurement over several coupling draws -- one draw is not a result."""
    seeds = [0, 10, 20, 30]
    console = common.Console({"beta": 5, **ROBUSTNESS_COLUMNS})
    console.rule(f"entropy robustness -- D={dim}, N={SITES}, {len(seeds)} coupling draws")
    console.header()

    results = {}
    for beta in BETAS:
        per_seed = [
            common.cached(
                f"01/robust_d{dim}_n{SITES}_beta{beta}_seed{seed}_{sampling_tag(sampling)}",
                lambda beta=beta, seed=seed: robustness_cell(dim, beta, seed, sampling),
                refresh=refresh,
            )
            for seed in seeds
        ]
        results[beta] = per_seed
        robustness_row(console, f"{beta:g}", per_seed)
    print(f"\nrobustness sweep complete in {console.elapsed()}")
    return results


def scaling_sweep(
    sampling: dict, refresh: bool, *, dim: int = 64, beta: float = 4.0
) -> dict:
    """Does the alignment protection strengthen like 1/sqrt(N)?

    The error's |cos| against ``J - J^T`` should track the directionless value
    ``sqrt(2/(N(N-1)))`` if it has no systematic component, while the signal's
    stays pinned near one.  Divergence of the measured curve from the reference
    as N grows would mean the error is acquiring structure along the asymmetry
    -- exactly the failure mode the real-couplings check watches for.
    """
    seeds = [0, 10, 20]
    console = common.Console({"N": 5, **ROBUSTNESS_COLUMNS})
    console.rule(
        f"alignment scaling -- D={dim}, beta={beta:g}, {len(seeds)} coupling draws"
    )
    console.header()

    results = {}
    for sites in SCALING_SITES:
        per_seed = [
            common.cached(
                f"01/robust_d{dim}_n{sites}_beta{beta}_seed{seed}_{sampling_tag(sampling)}",
                lambda sites=sites, seed=seed: robustness_cell(
                    dim, beta, seed, sampling, sites=sites
                ),
                refresh=refresh,
            )
            for seed in seeds
        ]
        results[sites] = per_seed
        robustness_row(console, f"{sites}", per_seed)
    print(f"\nscaling sweep complete in {console.elapsed()}")
    return results


def real_couplings_sweep(path: Path, sampling: dict, refresh: bool) -> dict:
    """The alignment measurement on couplings a trained model actually produced.

    The synthetic sweeps use ``softmax(randn)`` -- structureless.  Trained
    attention is not, and if the mean-field error and the attention asymmetry
    come to share structure (both dominated by positional locality, say), the
    alignment protecting sigma_hk could stop being zero.  This reads the probe
    couplings and drives exported by an experiment-02 run and reruns the
    measurement on head 0 of every layer.
    """
    payload = torch.load(path, weights_only=False)
    couplings, drives = payload["couplings"], payload["drives"]
    beta = payload["beta"]
    depth = couplings.shape[0]

    console = common.Console({"layer": 5, **ROBUSTNESS_COLUMNS})
    console.rule(
        f"entropy robustness on trained couplings -- {path.stem}, "
        f"D={drives.shape[-1]}, N={couplings.shape[-1]}, beta={beta:g}, head 0"
    )
    console.header()

    results = {}
    for layer in range(depth):
        cell = common.cached(
            f"01/real_{path.stem}_L{layer}h0_{sampling_tag(sampling)}",
            lambda layer=layer: alignment_cell(
                drives[layer, 0].double(),
                couplings[layer, 0].double(),
                beta,
                sampling,
                seed_base=3000 + 31 * layer,
            ),
            refresh=refresh,
        )
        results[layer] = cell
        robustness_row(console, f"{layer}", [cell])
    print(f"\nreal-couplings sweep complete in {console.elapsed()}")
    return results



#
# The phase plane.  Everything above measures one point at a time; this walks the
# two parameters the large-D problem actually has.
#


def phase_problem(dim: int, u: float, beta: float, seed: int = 0):
    """Drive of norm ``a R`` with ``a = u / beta``, so the cell sits at ``(u, beta)``.

    ``u = beta ||x|| / R`` is the pinning parameter and ``beta`` the coupling one.
    Sweeping beta at fixed drive walks the diagonal ``u = beta`` and cannot
    separate them, which is what the earlier version of this experiment did.
    """
    torch.manual_seed(seed)
    couplings = torch.softmax(torch.randn(SITES, SITES), -1)
    torch.manual_seed(seed + 1)
    drive = (u / beta) * stochastic.random_state((SITES,), dim)
    return drive, couplings


def local_contraction(step_fn, solved: torch.Tensor) -> float:
    """``fixed_point.local_contraction`` at this experiment's iteration budget."""
    return fp.local_contraction(step_fn, solved, iterations=POWER_ITERATIONS)


def count_branches(step_fn, template: torch.Tensor, *, starts: int = 6) -> int:
    """Distinct converged fixed points from the m=0 start plus random ones.

    Past the contraction boundary the map goes multistable and the m=0
    convention silently picks a branch, which is how a 108% "mean-field error"
    turned out to be branch selection.  The solving and the counting live in
    ``fixed_point.distinct_fixed_points``; what is local here is the start
    convention, seeded so every cell probes from the same initializations.
    """

    def start_points():
        yield torch.zeros_like(template)
        for index in range(1, starts):
            torch.manual_seed(500 + index)
            yield 0.5 * stochastic.random_state(
                template.shape[:-1], template.shape[-1]
            )

    return len(fp.distinct_fixed_points(step_fn, start_points(), **BRANCH))


def deterministic_grid(dim: int, us, betas, refresh: bool, *, starts: int = 6) -> dict:
    """Everything on the plane that needs no sampling, and so can be drawn finely.

    The large-D error, the contraction, and the branch count are all closed-form
    or solver-only.  They cost nothing next to the Monte Carlo, so they get a
    finer grid than the sampled panels and give a smooth rho = 1 contour.
    """

    def compute() -> dict:
        errors = {name: torch.zeros(len(betas), len(us)) for name in QUANTITIES}
        contraction = torch.zeros(len(betas), len(us))
        branches = torch.zeros(len(betas), len(us))
        residual = torch.zeros(len(betas), len(us))
        for row, beta in enumerate(betas):
            for column, u in enumerate(us):
                drive, couplings = phase_problem(dim, u, beta)
                exact = mean_field_cell(drive, couplings, beta, large_d=False)
                large_d = mean_field_cell(drive, couplings, beta, large_d=True)
                for name in QUANTITIES:
                    errors[name][row, column] = common.relative(
                        large_d["tensors"][name], exact["tensors"][name]
                    )
                # Contraction and branch count come from the *exact* map, because
                # that is the surface the sampled panels measure.  Reading them
                # off the large-D map instead would draw a contour belonging to a
                # different system on top of this one -- the same hybrid mistake
                # as mixing the two inside one formula.
                step_fn = lambda m: mf.step(m, drive, couplings, beta)
                residual[row, column] = exact["residual"]
                # A contraction measured at a point that is not a fixed point is
                # not a contraction.  Where the solve fails, Anderson returns an
                # iterate that is merely wandering, and the power iteration on it
                # reports rho ~ 1.000 for every beta -- which drew a spurious
                # closed contour across the top of the plane.  Same lesson as
                # never reporting a branch count without its residual.
                contraction[row, column] = (
                    local_contraction(step_fn, exact["tensors"]["magnetization"])
                    if exact["residual"] < 1e-9
                    else float("nan")
                )
                branches[row, column] = count_branches(step_fn, drive, starts=starts)
        return {
            "us": list(us),
            "betas": list(betas),
            "large_d_error": errors,
            "contraction": contraction,
            "branches": branches,
            "residual": residual,
        }

    return common.cached(
        f"01/plane_d{dim}_{len(us)}x{len(betas)}_{solve_tag(starts)}",
        compute,
        refresh=refresh,
    )


def phase_cell(
    dim: int, u: float, beta: float, sampling: dict, refresh: bool, *, starts: int = 6
) -> dict:
    def compute() -> dict:
        drive, couplings = phase_problem(dim, u, beta)
        exact = mean_field_cell(drive, couplings, beta, large_d=False)
        sampled = sampled_cell(drive, couplings, beta, references=exact, **sampling)
        step_fn = lambda m: mf.step(m, drive, couplings, beta)
        return {
            "mean_field_error": sampled["mean_field_error"],
            # Per-chain magnitude over pooled magnitude.  One when the chain is
            # ergodic; large when every chain orders in its own direction and the
            # pooled average cancels, which is the regime where comparing against
            # that average stops meaning anything.
            "ergodicity": sampled["ergodicity"],
            "chain_saturation": sampled["chain_saturation"],
            "contraction": local_contraction(
                step_fn, exact["tensors"]["magnetization"]
            ),
            "branches": count_branches(step_fn, drive, starts=starts),
            "saturation": exact["values"]["magnetization"],
            "floor": sampled["floor"],
            "floors": sampled["floors"],
            "residual": exact["residual"],
        }

    return common.cached(
        f"01/phase_d{dim}_u{u}_beta{beta}_{sampling_tag(sampling)}_{solve_tag(starts)}",
        compute,
        refresh=refresh,
    )


def phase_sweep(sampling: dict, refresh: bool, *, starts: int = 6) -> dict:
    console = common.Console(
        {
            "D": 4,
            "u": 6,
            "beta": 6,
            "a=u/b": 7,
            "|m|/R": 7,
            "rho": 6,
            "branch": 6,
            "MF err": 9,
            "floor": 9,
        }
    )
    console.rule(f"experiment 01 -- phase plane, N={SITES}")
    console.header()
    cells = {}
    for dim in PHASE_DIMS:
        for beta in BETAS:
            for u in U_VALUES:
                cell = phase_cell(dim, u, beta, sampling, refresh, starts=starts)
                cells[(dim, u, beta)] = cell
                console.row(
                    f"{dim}",
                    f"{u:g}",
                    f"{beta:g}",
                    u / beta,
                    cell["saturation"],
                    cell["contraction"],
                    f"{int(cell['branches'])}",
                    cell["mean_field_error"]["magnetization"],
                    cell["floor"],
                )
    print(f"\nphase sweep complete in {console.elapsed()}")
    return cells


#
# Figures
#


def sequential(name: str, steps: list[str]):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, steps)


BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE = ["#fbe0d4", "#f5bc9c", "#ef9a6c", "#eb6834", "#c74e20", "#9c3a15", "#6b260c"]


def figure_phase(quantity: str, cells: dict, planes: dict) -> None:
    """One phase diagram per quantity: where can this number be trusted?

    Top row is the *mean-field* error -- the exact mean-field fixed point against
    the sampled chain, with the Monte Carlo noise divided out.  Large D plays no
    part in it.  Bottom row is the large-D error against that same exact mean
    field, which is deterministic on both sides, so it costs nothing and is drawn
    on a finer grid.  It was an inset first; at 38% it covered the low-beta,
    high-u corner, which is precisely the region where the approximation works
    and therefore the part a reader most needs to see.

    Two sequential scales share the figure, so they take different hues: blue for
    the sampled comparison, orange for the large-D one.  Neither is categorical.
    """
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(
        2, len(PHASE_DIMS), figsize=(3.9 * len(PHASE_DIMS), 7.2),
        sharex=True, sharey=True, layout="constrained",
    )
    values = [
        cells[(dim, u, beta)]["mean_field_error"][quantity]
        for dim in PHASE_DIMS
        for u in U_VALUES
        for beta in BETAS
    ]
    # Capped at 100%.  Past the contraction boundary the "error" is branch
    # selection and runs past 1000%; letting that set the top of the scale would
    # compress everything the figure is about into the bottom decade.
    top = LogNorm(vmin=1e-3, vmax=1.0)
    bottom = LogNorm(vmin=1e-4, vmax=2.0)
    # Below the bottom of the scale means "too small to resolve", not "no data".
    # Left as bare white it reads as a hole in the sweep.
    blue, orange = sequential("blue", BLUE), sequential("orange", ORANGE)
    blue.set_under(BLUE[0])
    orange.set_under(ORANGE[0])
    blue.set_bad("#e6e5e1")

    for column, dim in enumerate(PHASE_DIMS):
        plane = planes[dim]

        # A cell counts as measured only where its error clears its own noise
        # floor.  Everything else is masked rather than painted at the bottom of
        # the scale: "below 1e-3" and "unresolved at a floor of 0.1" are very
        # different claims and must not share a colour.
        grid = torch.tensor(
            [
                [cells[(dim, u, beta)]["mean_field_error"][quantity] for u in U_VALUES]
                for beta in BETAS
            ]
        )
        floors = torch.tensor(
            [
                [cells[(dim, u, beta)]["floors"][quantity] for u in U_VALUES]
                for beta in BETAS
            ]
        )
        grid = numpy.ma.masked_where(
            ~torch.isfinite(grid).numpy() | (grid <= floors).numpy(), grid.numpy()
        )
        upper = axes[0, column].pcolormesh(
            U_VALUES, BETAS, grid, cmap=blue, norm=top,
            shading="nearest",
        )
        lower = axes[1, column].pcolormesh(
            plane["us"], plane["betas"], plane["large_d_error"][quantity],
            cmap=orange, norm=bottom, shading="auto",
        )

        for row, ax in enumerate(axes[:, column]):
            # rho = 1, measured on the fine deterministic grid so it is smooth,
            # and the post's sufficient condition as its u -> 0 asymptote.
            ax.contour(
                plane["us"], plane["betas"], plane["contraction"],
                levels=[1.0], colors=[common.INK], linewidths=1.4,
            )
            ax.axhline(critical_beta(dim), color=common.INK, lw=1.0, ls="--", alpha=0.55)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(U_VALUES)
            ax.set_xticklabels([f"{u:g}" for u in U_VALUES])
            ax.set_yticks(BETAS)
            ax.set_yticklabels([f"{b:g}" for b in BETAS])
            ax.minorticks_off()
            if row == 1:
                ax.set_xlabel(r"$u = \beta \|x\| / R$")

        # Cell-level caveats, on the sampled row only -- the deterministic row has
        # no sampling and no ambiguity about what it measures.
        for beta in BETAS:
            for u in U_VALUES:
                cell = cells[(dim, u, beta)]
                mark = None
                broken = (
                    cell["chain_saturation"] > 0.3 and cell["ergodicity"] > 1.4
                ) or not math.isfinite(cell["mean_field_error"][quantity])
                if broken:
                    mark = dict(marker="s", markerfacecolor="none", markersize=8)
                elif cell["branches"] > 1:
                    mark = dict(marker="x", markersize=7)
                elif cell["branches"] == 0:
                    mark = dict(marker="+", markersize=8)
                if mark:
                    axes[0, column].plot(
                        u, beta, color="white", mew=1.7, linestyle="none", **mark
                    )

        axes[0, column].set_title(
            rf"$D = {dim}$   ($\beta_c = {critical_beta(dim):.2f}$)", color=common.INK
        )

    axes[0, 0].set_ylabel(r"$\beta$   (coupling)")
    axes[1, 0].set_ylabel(r"$\beta$   (coupling)")
    fig.colorbar(upper, ax=axes[0, :], fraction=0.04, pad=0.02, extend="max").set_label(
        "mean field vs sampled", fontsize=8
    )
    fig.colorbar(lower, ax=axes[1, :], fraction=0.04, pad=0.02, extend="both").set_label(
        "large $D$ vs exact mean field", fontsize=8
    )
    legend = [
        Patch(facecolor="#e6e5e1", edgecolor="none", label="below noise floor"),
        Line2D([], [], color=common.INK, lw=1.4, label=r"$\rho = 1$"),
        Line2D([], [], color=common.INK, lw=1.0, ls="--", label=r"$\beta_c$"),
        Line2D([], [], color=common.INK_SOFT, marker="x", ls="none", label="multistable"),
        Line2D([], [], color=common.INK_SOFT, marker="+", ls="none", label="no fixed point"),
        Line2D(
            [], [], color=common.INK_SOFT, marker="s", markerfacecolor="none",
            ls="none", label="ergodicity broken",
        ),
    ]
    fig.legend(
        handles=legend, loc="outside lower center", ncol=6, fontsize=7.5, frameon=False
    )
    fig.suptitle(
        f"Where the mean field can be trusted \u2014 {quantity}",
        color=common.INK, fontsize=11,
    )
    common.save(fig, f"01_phase_{quantity}")
    plt.close(fig)


def seed_scatter(ax, x, cells: list[dict], key: str, color) -> None:
    """Per-seed points behind the mean line, so the coupling-draw spread stays visible."""
    for cell in cells:
        ax.plot(
            [x], [abs(cell[key])], marker="o", markersize=3, alpha=0.35,
            color=color, linestyle="none",
        )


def figure_robustness(results: dict) -> None:
    """Why sigma_hk survives where the correlations it is built from do not.

    Left: the raw sizes -- the delayed correlations go badly wrong as beta grows
    while the entropy production stays at the percent level.  Right: the
    mechanism -- the antisymmetric error keeps no systematic component along
    ``J - J^T`` (|cos| at the directionless floor), while the signal is pinned to
    that direction almost by construction.  The protection is a projection onto
    one direction out of N(N-1)/2, where the signal lives and the error does not.
    """
    betas = sorted(results)
    mean = lambda key: [
        sum(cell[key] for cell in results[b]) / len(results[b]) for b in betas
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), layout="constrained")

    axes[0].plot(
        betas, mean("delayed_error"), color=common.SAMPLED, marker="o",
        label=r"delayed correlations  $\|\Delta C\| / \|C\|$",
    )
    axes[0].plot(
        betas, mean("entropy_error"), color=common.EXACT, marker="o",
        label=r"entropy production  $|\Delta\sigma| / \sigma$",
    )
    for beta in betas:
        seed_scatter(axes[0], beta, results[beta], "delayed_error", common.SAMPLED)
        seed_scatter(axes[0], beta, results[beta], "entropy_error", common.EXACT)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("relative error of mean field")
    axes[0].set_title("the value survives", color=common.INK, fontsize=9.5)

    signal = [abs(v) for v in mean("signal_alignment")]
    error = [abs(v) for v in mean("alignment")]
    floor = results[betas[0]][0]["random_alignment"]
    axes[1].plot(
        betas, signal, color=common.EXACT, marker="o",
        label=r"signal  $|\cos(C_{\mathrm{anti}},\, J - J^T)|$",
    )
    axes[1].plot(
        betas, error, color=common.SAMPLED, marker="o",
        label=r"error  $|\cos(\Delta C_{\mathrm{anti}},\, J - J^T)|$",
    )
    axes[1].axhline(floor, color=common.INK_FAINT, lw=0.9, ls=":")
    axes[1].annotate(
        r"directionless  $\sqrt{2/N(N-1)}$", xy=(betas[0], floor), xytext=(0, 4),
        textcoords="offset points", fontsize=7.5, color=common.INK_SOFT,
    )
    for beta in betas:
        seed_scatter(axes[1], beta, results[beta], "signal_alignment", common.EXACT)
        seed_scatter(axes[1], beta, results[beta], "alignment", common.SAMPLED)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$|\cos|$ against $J - J^T$")
    axes[1].set_title("because of alignment", color=common.INK, fontsize=9.5)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(betas)
        ax.set_xticklabels([f"{b:g}" for b in betas])
        ax.minorticks_off()
        ax.set_xlabel(r"$\beta$")
        ax.legend(fontsize=7.5)
    fig.suptitle(
        r"Why $\sigma_{\mathrm{hk}}$ outlives the correlations it is built from",
        color=common.INK, fontsize=11,
    )
    common.save(fig, "01_entropy_robustness")
    plt.close(fig)


def figure_scaling(results: dict) -> None:
    """The alignment protection against system size, with the 1/sqrt reference."""
    sites = sorted(results)
    mean = lambda key: [
        sum(abs(cell[key]) for cell in results[n]) / len(results[n]) for n in sites
    ]
    fig, ax = plt.subplots(figsize=(5.4, 3.8), layout="constrained")

    ax.plot(
        sites, mean("signal_alignment"), color=common.EXACT, marker="o",
        label=r"signal  $|\cos(C_{\mathrm{anti}},\, J - J^T)|$",
    )
    ax.plot(
        sites, mean("alignment"), color=common.SAMPLED, marker="o",
        label=r"error  $|\cos(\Delta C_{\mathrm{anti}},\, J - J^T)|$",
    )
    reference = [(2.0 / (n * (n - 1))) ** 0.5 for n in sites]
    ax.plot(
        sites, reference, color=common.INK_FAINT, lw=0.9, ls=":",
        label=r"directionless  $\sqrt{2/N(N-1)}$",
    )
    for n in sites:
        seed_scatter(ax, n, results[n], "signal_alignment", common.EXACT)
        seed_scatter(ax, n, results[n], "alignment", common.SAMPLED)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sites)
    ax.set_xticklabels([str(n) for n in sites])
    ax.minorticks_off()
    ax.set_xlabel(r"$N$  (sites)")
    ax.set_ylabel(r"$|\cos|$ against $J - J^T$")
    ax.legend(fontsize=7.5)
    ax.set_title(
        "The projection protecting $\\sigma_{\\mathrm{hk}}$ strengthens with $N$",
        color=common.INK, fontsize=10,
    )
    common.save(fig, "01_alignment_scaling")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke-sized sampling")
    parser.add_argument("--refresh", action="store_true", help="recompute cached cells")
    parser.add_argument("--only", choices=["phase", "robustness", "scaling"])
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw from cache, computing nothing -- for tuning figures",
    )
    parser.add_argument(
        "--real",
        type=Path,
        help="experiment-02 run cache (experiments/data/02/<run>.pt): rerun the "
        "robustness measurement on its trained couplings instead of the sweeps",
    )
    args = parser.parse_args()

    torch.set_default_dtype(torch.float64)
    common.use_style()
    common.PLOT_ONLY = args.plot_only
    sampling = QUICK if args.quick else SAMPLING
    # The branch check dominates the deterministic cost (one solve per start per
    # cell), so the smoke path takes the minimum that can still see two branches.
    starts = 3 if args.quick else 6
    fine = 7 if args.quick else FINE

    if args.real is not None:
        real_couplings_sweep(args.real, sampling, args.refresh)
        return
    if args.only in (None, "robustness"):
        robustness = robustness_sweep(sampling, args.refresh)
        print("\nfigures")
        figure_robustness(robustness)
    if args.only in (None, "scaling"):
        scaling = scaling_sweep(sampling, args.refresh)
        print("\nfigures")
        figure_scaling(scaling)
    if args.only in (None, "phase"):
        fine_u = torch.logspace(
            math.log10(U_VALUES[0]), math.log10(U_VALUES[-1]), fine
        ).tolist()
        fine_beta = torch.logspace(
            math.log10(BETAS[0]), math.log10(BETAS[-1]), fine
        ).tolist()
        planes = {
            dim: deterministic_grid(
                dim, fine_u, fine_beta, args.refresh, starts=starts
            )
            for dim in PHASE_DIMS
        }
        cells = phase_sweep(sampling, args.refresh, starts=starts)
        print("\nfigures")
        for quantity in QUANTITIES:
            figure_phase(quantity, cells, planes)


if __name__ == "__main__":
    main()
