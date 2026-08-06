"""Synchronous stochastic dynamics of the vector-spin system.

Every site is resampled in parallel from its vMF conditional given the whole
previous configuration, so a trajectory is a Markov chain on ``(N, D)`` states.
This is the ground truth the mean-field recurrences in ``mean_field`` are
supposed to reproduce, and the estimators here (magnetizations, single-site
covariances, delayed correlations) are the sampled counterparts of the closed
forms used there.

Seed with ``torch.manual_seed``; no generator is threaded through.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from einops import einsum
from torch import Tensor

from neqnn import vmf


def _validate_simulation(
    drive: Tensor,
    couplings: Tensor,
    beta: float,
    *,
    num_chains: int,
    num_steps: int,
    burn_in: int,
    estimate: bool,
) -> None:
    if drive.ndim != 2:
        raise ValueError(
            f"drive must have shape (sites, dim), got {tuple(drive.shape)}"
        )
    if not drive.is_floating_point():
        raise TypeError(f"drive must be floating point, got {drive.dtype}")
    sites, dim = drive.shape
    if dim <= 2:
        raise ValueError(f"spin dimension must be > 2, got {dim}")
    if tuple(couplings.shape) != (sites, sites):
        raise ValueError(
            f"couplings must have shape {(sites, sites)}, got {tuple(couplings.shape)}"
        )
    if (
        couplings.dtype != drive.dtype
        or couplings.device != drive.device
        or not couplings.is_floating_point()
    ):
        raise ValueError("couplings must share drive's floating dtype and device")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError(f"beta must be finite and positive, got {beta}")
    minimum_chains = 2 if estimate else 1
    if (
        not isinstance(num_chains, int)
        or isinstance(num_chains, bool)
        or num_chains < minimum_chains
    ):
        raise ValueError(
            f"num_chains must be an integer >= {minimum_chains}, got {num_chains!r}"
        )
    minimum_steps = 2 if estimate else 1
    if (
        not isinstance(num_steps, int)
        or isinstance(num_steps, bool)
        or num_steps < minimum_steps
    ):
        raise ValueError(
            f"num_steps must be an integer >= {minimum_steps}, got {num_steps!r}"
        )
    if not isinstance(burn_in, int) or isinstance(burn_in, bool) or burn_in < 0:
        raise ValueError(f"burn_in must be a non-negative integer, got {burn_in!r}")


def effective_field(state: Tensor, drive: Tensor, couplings: Tensor) -> Tensor:
    """h_i = x_i + sum_j J_ij s_j, for states shaped (..., N, D)."""
    return drive + einsum(couplings, state, "... i j, ... j d -> ... i d")


def step(state: Tensor, drive: Tensor, couplings: Tensor, beta: float) -> Tensor:
    """One synchronous update: resample every site from its vMF conditional."""
    return vmf.sample_from_field(effective_field(state, drive, couplings), beta)


def random_state(shape: tuple[int, ...], dim: int, **kwargs) -> Tensor:
    """Uniformly random spins on the sphere of radius R."""
    values = torch.randn(*shape, dim, **kwargs)
    return vmf.radius(dim) * values / values.norm(dim=-1, keepdim=True)


def simulate(
    drive: Tensor,
    couplings: Tensor,
    beta: float,
    *,
    num_chains: int,
    num_steps: int,
    burn_in: int,
) -> Tensor:
    """Run independent chains from random starts, returning states (T, C, N, D).

    Chains are independent replicates, so pooling over both axes is legitimate
    for stationary estimates while the chain axis still exposes seed spread.

    Holds the whole trajectory, so it costs ``T C N D`` numbers -- fine when the
    path itself is wanted (plots, autocorrelations, D=3 visualizations) and
    hopeless at the sizes the first experiment runs: N=128, D=512, 64 chains,
    4000 steps is 125 GiB.  Use ``estimate`` for stationary averages, which
    streams the same chains and keeps only the accumulators.
    """
    _validate_simulation(
        drive,
        couplings,
        beta,
        num_chains=num_chains,
        num_steps=num_steps,
        burn_in=burn_in,
        estimate=False,
    )
    sites, dim = drive.shape
    kwargs = dict(dtype=drive.dtype, device=drive.device)
    state = random_state((num_chains, sites), dim, **kwargs)
    for _ in range(burn_in):
        state = step(state, drive, couplings, beta)
    states = torch.empty(num_steps, num_chains, sites, dim, **kwargs)
    for index in range(num_steps):
        state = step(state, drive, couplings, beta)
        states[index] = state
    return states


class Estimates(NamedTuple):
    """Stationary estimates pooled over chains and time, matching the estimators below.

    ``chain_magnetizations`` is the per-chain magnetization *magnitude*, averaged
    over chains, against which ``magnetizations`` is the pooled vector average.
    The two agree when the chain is ergodic and separate sharply when it is not:
    at strong coupling and weak drive every chain magnetizes to nearly the same
    magnitude but in its own arbitrary direction, so the pooled average collapses
    while the per-chain magnitude does not.  Comparing mean field against the
    pooled average there compares an ordered state to an average over rotations,
    which is a different question from how good the approximation is.
    """

    magnetizations: Tensor  # (N, D)
    covariances: Tensor  # (N, D, D)
    delayed_correlations: Tensor  # (N, N)
    standard_error: Tensor  # (N,)
    chain_magnetizations: Tensor  # (N,)


class ReplicateEstimates(NamedTuple):
    """Independent stationary estimates, retaining a leading replicate axis.

    This is the lightweight route for Monte Carlo error bars.  It deliberately
    omits the per-site ``D x D`` covariance matrices: experiments comparing
    magnetizations and delayed correlations do not consume them, and forming
    them can cost as much as the dynamics themselves.
    """

    magnetizations: Tensor  # (R, N, D)
    delayed_correlations: Tensor  # (R, N, N)
    chain_magnetizations: Tensor  # (R, N)


def estimate_replicates(
    drive: Tensor,
    couplings: Tensor,
    beta: float,
    *,
    num_repeats: int,
    num_chains: int,
    num_steps: int,
    burn_in: int,
) -> ReplicateEstimates:
    """Stream several independent estimates together without unused moments.

    Replicates and chains are statistically independent axes.  Advancing them
    in one tensor removes repeated Python dispatch and gives the matrix
    multiplications enough batch work to run efficiently, while retaining each
    replicate for cross-replicate bias correction.
    """
    _validate_simulation(
        drive,
        couplings,
        beta,
        num_chains=num_chains,
        num_steps=num_steps,
        burn_in=burn_in,
        estimate=True,
    )
    if (
        not isinstance(num_repeats, int)
        or isinstance(num_repeats, bool)
        or num_repeats < 1
    ):
        raise ValueError(f"num_repeats must be a positive integer, got {num_repeats!r}")

    sites, dim = drive.shape
    kwargs = dict(dtype=drive.dtype, device=drive.device)
    state = random_state((num_repeats, num_chains, sites), dim, **kwargs)
    for _ in range(burn_in):
        state = step(state, drive, couplings, beta)

    total = torch.zeros(num_repeats, sites, dim, **kwargs)
    lagged = torch.zeros(num_repeats, sites, sites, **kwargs)
    chain_total = torch.zeros(num_repeats, num_chains, sites, dim, **kwargs)
    for index in range(num_steps):
        previous, state = state, step(state, drive, couplings, beta)
        total += state.sum(1)
        chain_total += state
        if index:
            lagged += einsum(
                state,
                previous,
                "r c i d, r c j d -> r i j",
            )

    count = num_steps * num_chains
    magnetizations = total / count
    delayed = lagged / ((num_steps - 1) * num_chains) - einsum(
        magnetizations,
        magnetizations,
        "r i d, r j d -> r i j",
    )
    chain_means = chain_total / num_steps
    return ReplicateEstimates(
        magnetizations=magnetizations,
        delayed_correlations=delayed,
        chain_magnetizations=chain_means.norm(dim=-1).mean(1),
    )


def estimate(
    drive: Tensor,
    couplings: Tensor,
    beta: float,
    *,
    num_chains: int,
    num_steps: int,
    burn_in: int,
) -> Estimates:
    """Same chains as ``simulate``, but accumulated instead of stored.

    Every stationary estimator we need is a sum over ``(t, c)`` of something
    local in time -- first moment, second moment, and the one-step-lagged
    product -- so none of them requires the trajectory to be kept.  Memory drops
    from ``T C N D`` to ``N D^2``, which at N=128, D=512 is 256 MiB regardless
    of how long the chains run, and the run length becomes purely a time budget.

    The connected quantities are formed at the end from the raw moments.  That
    is the numerically weaker order of operations than centering first, but the
    alternative needs a mean that is not known until the run is over, and in
    float64 at these magnitudes the cancellation costs a few digits out of
    fifteen -- far below Monte Carlo error.
    """
    _validate_simulation(
        drive,
        couplings,
        beta,
        num_chains=num_chains,
        num_steps=num_steps,
        burn_in=burn_in,
        estimate=True,
    )
    sites, dim = drive.shape
    kwargs = dict(dtype=drive.dtype, device=drive.device)
    state = random_state((num_chains, sites), dim, **kwargs)
    for _ in range(burn_in):
        state = step(state, drive, couplings, beta)

    total = torch.zeros(sites, dim, **kwargs)
    second = torch.zeros(sites, dim, dim, **kwargs)
    lagged = torch.zeros(sites, sites, **kwargs)
    chain_total = torch.zeros(num_chains, sites, dim, **kwargs)

    for index in range(num_steps):
        previous, state = state, step(state, drive, couplings, beta)
        total += state.sum(0)
        chain_total += state
        second += einsum(state, state, "c n d, c n e -> n d e")
        if index:
            lagged += einsum(state, previous, "c i d, c j d -> i j")

    count = num_steps * num_chains
    magnetizations = total / count
    covariances = second / count - einsum(
        magnetizations, magnetizations, "n d, n e -> n d e"
    )
    delayed = lagged / ((num_steps - 1) * num_chains) - einsum(
        magnetizations, magnetizations, "i d, j d -> i j"
    )
    chain_means = chain_total / num_steps
    return Estimates(
        magnetizations=magnetizations,
        covariances=covariances,
        delayed_correlations=delayed,
        standard_error=chain_means.std(0, correction=1).norm(dim=-1) / num_chains**0.5,
        chain_magnetizations=chain_means.norm(dim=-1).mean(0),
    )


def transition_logp(
    target: Tensor, source: Tensor, drive: Tensor, couplings: Tensor, beta: float
) -> Tensor:
    """log P(target | source), summed over sites, up to the fixed-radius constant.

    Since ``kappa mu . u`` is just ``beta h . s'``, the exponent never needs the
    field to be normalized.
    """
    field = effective_field(source, drive, couplings)
    log_norm = vmf.log_normalizer(vmf.concentration(field, beta), field.shape[-1])
    return (log_norm + beta * (target * field).sum(-1)).sum(-1)


#
# Sampled estimators.  All take states shaped (T, C, N, D) and pool over T and C.
#


def magnetizations(states: Tensor) -> Tensor:
    """Stationary magnetizations m_i = <s_i>, shape (N, D)."""
    return states.flatten(0, 1).mean(0)


def covariances(states: Tensor) -> Tensor:
    """Single-site covariances Sigma_i = Cov[s_i], shape (N, D, D)."""
    flat = states.flatten(0, 1)
    centered = flat - flat.mean(0)
    return einsum(centered, centered, "t n d, t n e -> n d e") / centered.shape[0]


def delayed_correlations(states: Tensor) -> Tensor:
    """Connected delayed correlations Tr <s_{i,t+1} s_{j,t}^T>_c, shape (N, N).

    Written as ``<a b> - <a><b>`` rather than by centering each leg first.  The
    two differ here, unlike for the single-time covariance: there are only T-1
    lagged pairs against T states, so centering on the pooled mean leaves a
    residual O(1/T) that the raw-moment form does not have.  It also makes this
    agree with ``estimate`` to machine precision, which is what lets the cheap
    streaming path be checked against the explicit one.
    """
    mean = states.flatten(0, 1).mean(0)
    later, earlier = states[1:].flatten(0, 1), states[:-1].flatten(0, 1)
    return einsum(later, earlier, "t i d, t j d -> i j") / later.shape[0] - einsum(
        mean, mean, "i d, j d -> i j"
    )


def standard_error(states: Tensor) -> Tensor:
    """Per-site Monte Carlo standard error of the magnetization, shape (N,).

    Chains are independent, so the spread across chain means is an honest error
    bar without having to model the within-chain autocorrelation.
    """
    chain_means = states.mean(0)
    return chain_means.std(0, correction=1).norm(dim=-1) / chain_means.shape[0] ** 0.5
