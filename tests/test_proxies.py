"""Nonequilibrium diagnostics: entropy production and the post-quench mismatch."""

from __future__ import annotations

import pytest
import torch

from helpers import random_problem
from neqnn import fixed_point as fp, mean_field as mf, proxies, stochastic, vmf


def test_housekeeping_entropy_production_vanishes_for_symmetric_couplings():
    dim, beta = 64, 1.0
    drive, couplings = random_problem(dim)
    symmetric = 0.5 * (couplings + couplings.T)
    # A nonzero magnetization, so the field actually depends on the couplings --
    # at m = 0 the effective field is the bare drive whatever J is, and the
    # symmetric and asymmetric cases would be measured on identical fields.
    magnetizations = vmf.response_large_d(drive, beta)

    field = mf.effective_field(magnetizations, drive, symmetric)
    traces = mf.covariance_traces_large_d(field, field, beta)
    assert float(
        proxies.housekeeping_entropy_production(symmetric, traces, beta)
    ) == pytest.approx(0.0, abs=1e-30)

    field = mf.effective_field(magnetizations, drive, couplings)
    traces = mf.covariance_traces_large_d(field, field, beta)
    assert float(proxies.housekeeping_entropy_production(couplings, traces, beta)) > 0


def test_housekeeping_is_the_mean_field_substitution_into_the_exact_relation():
    """sigma_hk is entropy_production fed the mean-field delayed correlations."""
    dim, beta = 64, 1.0
    drive, couplings = random_problem(dim)
    field = mf.effective_field(torch.zeros_like(drive), drive, couplings)
    traces = mf.covariance_traces_large_d(field, field, beta)
    assert torch.allclose(
        proxies.entropy_production(couplings, beta * couplings * traces, beta),
        proxies.housekeeping_entropy_production(couplings, traces, beta),
        rtol=1e-12,
    )


def test_entropy_production_matches_the_path_log_ratio_of_the_chain():
    """The exact relation must reproduce <log P(s'|s) - log P(s|s')> on the chain.

    This is the only fully independent check on the entropy production: one side
    is a correlation function, the other is the definition of irreversibility.
    """
    dim, sites, beta = 32, 16, 1.0
    drive, couplings = random_problem(dim, sites)
    chains, steps = 32, 200

    torch.manual_seed(5)
    state = stochastic.random_state((chains, sites), dim)
    for _ in range(100):
        state = stochastic.step(state, drive, couplings, beta)

    total = torch.zeros(sites, dim)
    lagged = torch.zeros(sites, sites)
    log_ratio = 0.0
    for index in range(steps):
        previous, state = state, stochastic.step(state, drive, couplings, beta)
        total += state.sum(0)
        if index:
            lagged += torch.einsum("cid,cjd->ij", state, previous)
            log_ratio += float(
                (
                    stochastic.transition_logp(state, previous, drive, couplings, beta)
                    - stochastic.transition_logp(
                        previous, state, drive, couplings, beta
                    )
                ).sum()
            )

    magnetizations = total / (steps * chains)
    delayed = lagged / ((steps - 1) * chains) - torch.einsum(
        "id,jd->ij", magnetizations, magnetizations
    )
    from_correlations = float(proxies.entropy_production(couplings, delayed, beta))
    from_paths = log_ratio / ((steps - 1) * chains)

    assert from_correlations > 0
    assert abs(from_correlations - from_paths) < 0.05 * from_paths


def test_entropy_production_only_sees_the_antisymmetric_part():
    """A symmetric perturbation of the delayed correlations must not move sigma.

    This is why sigma_hk stays accurate in a regime where the delayed
    correlations themselves are badly wrong: the mean-field error is mostly
    symmetric, and sigma is blind to that subspace by construction.
    """
    dim, beta = 32, 1.0
    _, couplings = random_problem(dim)
    delayed = torch.randn(8, 8)
    symmetric = torch.randn(8, 8)
    symmetric = symmetric + symmetric.T
    assert torch.allclose(
        proxies.entropy_production(couplings, delayed + symmetric, beta),
        proxies.entropy_production(couplings, delayed, beta),
    )


def test_mismatch_vanishes_at_the_fixed_point_and_is_positive_away_from_it():
    dim, beta = 64, 1.0
    drive, couplings = random_problem(dim)
    step_fn = lambda m: mf.step_large_d(m, drive, couplings, beta)
    solved = fp.anderson(step_fn, torch.zeros_like(drive), max_iter=60, tol=1e-13).solution
    steady_field = mf.effective_field(solved, drive, couplings)
    perturbed = mf.effective_field(0.5 * solved, drive, couplings)

    for exact in (False, True):
        assert float(
            proxies.mismatch(steady_field, steady_field, beta, exact=exact)
        ) == pytest.approx(0.0, abs=1e-16)
        assert float(proxies.mismatch(perturbed, steady_field, beta, exact=exact)) > 0


def test_mismatch_decreases_along_a_contracting_relaxation():
    dim, beta = 64, 1.0
    drive, couplings = random_problem(dim)
    assert mf.contraction_factor_large_d(couplings, beta) < 1
    step_fn = lambda m: mf.step_large_d(m, drive, couplings, beta)
    solved = fp.anderson(step_fn, torch.zeros_like(drive), max_iter=60, tol=1e-13).solution
    trajectory = mf.relax_large_d(
        torch.zeros_like(drive), drive, couplings, beta, num_steps=40
    )
    fields = mf.effective_field(trajectory, drive, couplings)
    steady_field = mf.effective_field(solved, drive, couplings)
    values = proxies.mismatch(fields, steady_field, beta)
    # It falls about a decade per step from O(1) and then sits on the float64
    # floor around 4e-15, where the differences are roundoff.  Monotonicity is
    # only a claim about the part above that floor.
    above_floor = values[:-1] > 1e-12
    assert above_floor.sum() > 10
    assert torch.all(values[1:][above_floor] < values[:-1][above_floor])
