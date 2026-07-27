"""Mean-field recurrences: relaxation, contraction bounds, delayed correlations."""

from __future__ import annotations

import torch

from helpers import DIMS, random_problem, relative, slope_in_dim
from neqnn import fixed_point as fp, mean_field as mf, stochastic, vmf


def test_relax_stops_early_at_tolerance():
    dim, beta = 64, 1.0
    drive, couplings = random_problem(dim)
    trajectory = mf.relax(
        torch.zeros_like(drive), drive, couplings, beta, num_steps=500, tol=1e-10
    )
    assert trajectory.shape[0] < 500
    assert fp.residual(trajectory[-1], trajectory[-2]) < 1e-10


def test_contraction_factor_bounds_the_observed_lipschitz_constant():
    dim, beta = 64, 1.0
    drive, couplings = random_problem(dim)
    bound = mf.contraction_factor(couplings, beta, dim)
    worst = 0.0
    for _ in range(20):
        a = stochastic.random_state((8,), dim)
        b = a + 1e-3 * torch.randn_like(a)
        gap = mf.step(a, drive, couplings, beta) - mf.step(b, drive, couplings, beta)
        worst = max(worst, float(gap.norm() / (a - b).norm()))
    assert worst <= bound
    assert mf.contraction_factor_large_d(couplings, beta) >= bound


def test_covariance_traces_large_d_converges_as_one_over_dim():
    beta = 1.0
    errors = []
    for dim in DIMS:
        field, _ = random_problem(dim)
        other = stochastic.random_state((8,), dim)
        errors.append(
            relative(
                mf.covariance_traces_large_d(field, other, beta),
                mf.covariance_traces(field, other, beta),
            )
        )
    assert slope_in_dim(errors) < -0.9
    assert errors[-1] < 0.01


def test_covariance_traces_large_d_matches_direct_contraction():
    """The Gram-matrix shortcut must equal contracting the large-D covariances."""
    dim, beta = 64, 1.0
    field, _ = random_problem(dim)
    other = stochastic.random_state((8,), dim)
    direct = torch.einsum(
        "ide,jed->ij",
        vmf.covariance_large_d(field, beta),
        vmf.covariance_large_d(other, beta),
    )
    assert torch.allclose(
        mf.covariance_traces_large_d(field, other, beta), direct, rtol=1e-10
    )


def test_mean_field_reproduces_the_chain_at_weak_coupling():
    dim, sites, beta = 64, 8, 1.0
    drive, couplings = random_problem(dim, sites, seed=3)
    sampled = stochastic.estimate(
        drive, couplings, beta, num_chains=64, num_steps=1500, burn_in=300
    )
    predicted = mf.relax(
        torch.zeros_like(drive), drive, couplings, beta, num_steps=300, tol=1e-12
    )[-1]
    gap = (sampled.magnetizations - predicted).norm(dim=-1)
    assert float(gap.max()) < 5 * float(sampled.standard_error.max())
