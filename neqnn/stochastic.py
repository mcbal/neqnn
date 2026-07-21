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

import torch
from einops import einsum
from torch import Tensor

from neqnn import vmf


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
    """
    sites, dim = drive.shape[-2:]
    kwargs = dict(dtype=drive.dtype, device=drive.device)
    state = random_state((num_chains, sites), dim, **kwargs)
    for _ in range(burn_in):
        state = step(state, drive, couplings, beta)
    states = torch.empty(num_steps, num_chains, sites, dim, **kwargs)
    for index in range(num_steps):
        state = step(state, drive, couplings, beta)
        states[index] = state
    return states


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
    """Connected delayed correlations Tr <s_{i,t+1} s_{j,t}^T>_c, shape (N, N)."""
    mean = states.flatten(0, 1).mean(0)
    later = states[1:].flatten(0, 1) - mean
    earlier = states[:-1].flatten(0, 1) - mean
    return einsum(later, earlier, "t i d, t j d -> i j") / later.shape[0]


def standard_error(states: Tensor) -> Tensor:
    """Per-site Monte Carlo standard error of the magnetization, shape (N,).

    Chains are independent, so the spread across chain means is an honest error
    bar without having to model the within-chain autocorrelation.
    """
    chain_means = states.mean(0)
    return chain_means.std(0, correction=1).norm(dim=-1) / chain_means.shape[0] ** 0.5
