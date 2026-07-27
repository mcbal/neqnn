"""The sampled chain: streaming estimators and the transition kernel."""

from __future__ import annotations

import torch

from helpers import random_problem, relative
from neqnn import stochastic, vmf


def test_streaming_estimates_match_the_stored_trajectory():
    dim, sites, beta = 24, 6, 1.0
    drive, couplings = random_problem(dim, sites)
    settings = dict(num_chains=32, num_steps=400, burn_in=100)

    torch.manual_seed(7)
    states = stochastic.simulate(drive, couplings, beta, **settings)
    torch.manual_seed(7)
    streamed = stochastic.estimate(drive, couplings, beta, **settings)

    assert relative(streamed.magnetizations, stochastic.magnetizations(states)) < 1e-12
    assert relative(streamed.covariances, stochastic.covariances(states)) < 1e-12
    assert (
        relative(streamed.delayed_correlations, stochastic.delayed_correlations(states))
        < 1e-10
    )
    assert relative(streamed.standard_error, stochastic.standard_error(states)) < 1e-12


def test_transition_logp_is_a_normalized_density():
    """The dropped constant is exactly the uniform density, so exp(logp) is a ratio.

    Averaging that ratio over uniform draws integrates the transition kernel, and
    it has to come out at one.  Weak coupling keeps the importance weights tame
    enough for 400k draws to resolve it.
    """
    dim, sites, beta, draws = 8, 2, 1.0, 400_000
    drive, couplings = random_problem(dim, sites)
    source = stochastic.random_state((sites,), dim)
    targets = stochastic.random_state((draws, sites), dim)
    weights = stochastic.transition_logp(
        targets, source.expand_as(targets), drive, couplings, beta
    ).exp()
    error = float(weights.std() / draws**0.5)
    assert abs(float(weights.mean()) - 1.0) < 4 * error


def test_transition_logp_ratio_is_normalized_under_the_proposal():
    """Reweighting draws from one source to another must also integrate to one."""
    dim, sites, beta, draws = 8, 2, 1.0, 400_000
    drive, couplings = random_problem(dim, sites)
    source, other = (stochastic.random_state((sites,), dim) for _ in range(2))
    field = stochastic.effective_field(source, drive, couplings)
    targets = vmf.sample_from_field(field.expand(draws, sites, dim), beta)
    logp = lambda s: stochastic.transition_logp(
        targets, s.expand_as(targets), drive, couplings, beta
    )
    weights = (logp(other) - logp(source)).exp()
    error = float(weights.std() / draws**0.5)
    assert abs(float(weights.mean()) - 1.0) < 4 * error
