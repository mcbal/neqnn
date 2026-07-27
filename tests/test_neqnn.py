"""Checks that the closed forms agree with what they approximate.

Everything here is a cross-check between two independent routes to the same
number -- Bessel against sampling, large-D against exact, streaming against
stored, implicit gradient against unrolled -- rather than against stored
constants.  Monte Carlo tolerances are set from the standard error, not tuned.
"""

from __future__ import annotations

import math

import numpy
import pytest
import torch

from neqnn import (
    SpinModelTransformerModule,
    advance,
    fixed_point as fp,
    mean_field as mf,
    proxies,
    stochastic,
    vmf,
)

DIMS = [16, 32, 64, 128, 256, 512]


@pytest.fixture(autouse=True)
def double_precision():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(previous)


def relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).norm() / expected.norm())


def slope_in_dim(errors: list[float]) -> float:
    """Fitted exponent of ``error ~ D^slope``; the large-D forms should give -1."""
    design = torch.stack(
        [torch.log(torch.tensor(DIMS, dtype=torch.float64)), torch.ones(len(DIMS))], -1
    )
    return float(
        torch.linalg.lstsq(design, torch.log(torch.tensor(errors))[:, None]).solution[0, 0]
    )


def random_problem(dim: int, sites: int = 8, seed: int = 1):
    torch.manual_seed(seed)
    drive = stochastic.random_state((sites,), dim)
    couplings = 0.9 * torch.softmax(torch.randn(sites, sites), -1)
    return drive, couplings


#
# vMF single-site mathematics
#


def test_bessel_ratio_matches_scipy_where_scipy_survives():
    ive = pytest.importorskip("scipy.special").ive
    x = torch.tensor([1e-2, 0.1, 1.0, 10.0, 100.0, 500.0])
    for dim in DIMS:
        ours = vmf.bessel_ratio(x, dim / 2 - 1)
        with numpy.errstate(invalid="ignore", divide="ignore"):
            reference = torch.as_tensor(
                ive(dim / 2, x.numpy()) / ive(dim / 2 - 1, x.numpy())
            )
        finite = torch.isfinite(reference)
        assert torch.allclose(ours[finite], reference[finite], rtol=1e-7)
        # Underflow kills scipy at large order and small argument; the backward
        # recurrence returns the correct x/D there, which is the reason it exists.
        assert torch.all(ours > 0)


def test_bessel_ratio_small_argument_limit():
    for dim in DIMS:
        x = torch.tensor([1e-8])
        assert vmf.bessel_ratio(x, dim / 2 - 1).item() == pytest.approx(
            float(x) / dim, rel=1e-6
        )


def test_log_normalizer_matches_scipy():
    special = pytest.importorskip("scipy.special")
    kappa = torch.tensor([0.5, 5.0, 50.0, 200.0])
    for dim in (8, 64):
        order = dim / 2 - 1
        reference = (
            order * torch.log(kappa)
            - (torch.log(torch.as_tensor(special.ive(order, kappa.numpy()))) + kappa)
            - order * math.log(2)
            - special.gammaln(order + 1)
        )
        assert torch.allclose(vmf.log_normalizer(kappa, dim), reference, atol=1e-9)


def test_sampler_reproduces_closed_form_moments():
    dim, sites, beta, draws = 32, 4, 1.0, 200_000
    field = stochastic.random_state((sites,), dim)
    samples = vmf.sample_from_field(field.expand(draws, sites, dim), beta)

    mean = samples.mean(0)
    error = float(samples.std(0).norm() / draws**0.5)
    assert float((mean - vmf.response(field, beta)).norm()) < 4 * error

    centered = samples - mean
    covariance = torch.einsum("tnd,tne->nde", centered, centered) / draws
    assert relative(covariance, vmf.covariance(field, beta)) < 0.02


def test_covariance_agrees_with_its_variances():
    dim, beta = 64, 1.0
    field = stochastic.random_state((6,), dim)
    tangential, radial = vmf.variances(field, beta)
    covariance = vmf.covariance(field, beta)
    direction = torch.nn.functional.normalize(field, dim=-1)
    along = torch.einsum("nd,nde,ne->n", direction, covariance, direction)
    assert torch.allclose(along, radial)
    trace = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    assert torch.allclose(trace, radial + (dim - 1) * tangential)


#
# The large-D limit: every pair must converge as 1/D
#


@pytest.mark.parametrize(
    "quantity", ["response", "covariance", "covariance_traces", "kl"]
)
def test_large_d_forms_converge_as_one_over_dim(quantity):
    beta = 1.0
    errors = []
    for dim in DIMS:
        field, _ = random_problem(dim)
        other = stochastic.random_state((8,), dim)
        if quantity == "response":
            pair = vmf.response_large_d(field, beta), vmf.response(field, beta)
        elif quantity == "covariance":
            pair = vmf.covariance_large_d(field, beta), vmf.covariance(field, beta)
        elif quantity == "covariance_traces":
            pair = (
                mf.covariance_traces_large_d(field, other, beta),
                mf.covariance_traces(field, other, beta),
            )
        else:
            pair = vmf.kl_large_d(field, other, beta), vmf.kl(field, other, beta)
        errors.append(relative(*pair))
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


#
# Mean field, fixed points, gradients
#


def test_relax_stops_early_at_tolerance():
    dim, beta = 64, 1.0
    drive, couplings = random_problem(dim)
    trajectory = mf.relax(
        torch.zeros_like(drive), drive, couplings, beta, num_steps=500, tol=1e-10
    )
    assert trajectory.shape[0] < 500
    assert fp.residual(trajectory[-1], trajectory[-2]) < 1e-10


@pytest.mark.parametrize("large_d", [False, True])
def test_anderson_finds_the_same_point_as_picard(large_d):
    dim, beta = 64, 1.0
    drive, couplings = random_problem(dim)
    relax = mf.relax_large_d if large_d else mf.relax
    step = mf.step_large_d if large_d else mf.step
    step_fn = lambda m: step(m, drive, couplings, beta)

    picard = relax(torch.zeros_like(drive), drive, couplings, beta, num_steps=500)[-1]
    solved = fp.anderson(step_fn, torch.zeros_like(drive), max_iter=50, tol=1e-12)
    assert fp.residual(step_fn(solved), solved) < 1e-10
    assert relative(solved, picard) < 1e-8


def test_anderson_handles_an_already_converged_problem():
    """Zero drive and zero start: every gap vanishes and the Gram matrix is singular."""
    dim = 64
    drive = torch.zeros(8, dim)
    couplings = torch.softmax(torch.randn(8, 8), -1)
    step_fn = lambda m: mf.step_large_d(m, drive, couplings, 1.0)
    solved = fp.anderson(step_fn, torch.zeros_like(drive), max_iter=30, tol=1e-10)
    assert torch.allclose(solved, torch.zeros_like(solved))


def test_implicit_gradient_matches_unrolled_autograd():
    dim, beta = 64, 1.0
    drive, couplings = random_problem(dim)
    drive, couplings = drive.requires_grad_(True), couplings.requires_grad_(True)

    unrolled = mf.relax_large_d(
        torch.zeros_like(drive.detach()), drive, couplings, beta, num_steps=500
    )[-1]
    expected = torch.autograd.grad(unrolled.sum(), [drive, couplings])

    leaves = [t.detach().clone().requires_grad_(True) for t in (drive, couplings)]
    step_fn = lambda m: mf.step_large_d(m, leaves[0], leaves[1], beta)
    with torch.no_grad():
        solved = fp.anderson(
            step_fn, torch.zeros_like(drive.detach()), max_iter=60, tol=1e-13
        )
    actual = torch.autograd.grad(fp.implicit_grad(step_fn, solved).sum(), leaves)

    for got, want in zip(actual, expected):
        assert relative(got, want) < 1e-6


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


#
# Stochastic chain, and mean field against it
#


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


#
# Diagnostics
#


def test_housekeeping_entropy_production_vanishes_for_symmetric_couplings():
    dim, beta = 64, 1.0
    drive, couplings = random_problem(dim)
    symmetric = 0.5 * (couplings + couplings.T)
    field = mf.effective_field(torch.zeros_like(drive), drive, symmetric)
    traces = mf.covariance_traces_large_d(field, field, beta)
    assert float(
        proxies.housekeeping_entropy_production(symmetric, traces, beta)
    ) == pytest.approx(0.0, abs=1e-30)
    asymmetric_traces = mf.covariance_traces_large_d(field, field, beta)
    assert float(proxies.housekeeping_entropy_production(couplings, asymmetric_traces, beta)) > 0


def test_mismatch_vanishes_at_the_fixed_point_and_is_positive_away_from_it():
    dim, beta = 64, 1.0
    drive, couplings = random_problem(dim)
    step_fn = lambda m: mf.step_large_d(m, drive, couplings, beta)
    solved = fp.anderson(step_fn, torch.zeros_like(drive), max_iter=60, tol=1e-13)
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
    solved = fp.anderson(step_fn, torch.zeros_like(drive), max_iter=60, tol=1e-13)
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


#
# The module
#


@pytest.mark.parametrize("num_steps", [1, 4, None])
@pytest.mark.parametrize("init", ["reset", "amortized", "carried"])
def test_every_quadrant_runs_forward_and_backward(num_steps, init):
    module = SpinModelTransformerModule(
        dim=64,
        num_heads=4,
        num_steps=num_steps,
        init=init,
        beta=1.0,
        rope=True,
        measure_entropy_production=True,
    )
    x = torch.randn(2, 16, 64)
    readout = module(x)
    assert readout.magnetizations.shape == x.shape
    assert readout.state.magnetizations.shape == (2, 4, 16, 16)
    assert float(readout.entropy_production.detach().min()) >= 0
    readout.magnetizations.sum().backward()
    assert module.drive_norm.weight.grad is not None


def test_magnetizations_respect_the_head_radius():
    dim, num_heads = 128, 8
    module = SpinModelTransformerModule(
        dim=dim, num_heads=num_heads, num_steps=8, beta=2.0
    )
    state = module(torch.randn(2, 16, dim)).state.magnetizations
    assert float(state.detach().norm(dim=-1).max()) <= module.radius_head + 1e-9


def test_one_step_from_reset_cannot_see_the_couplings():
    """m_0 = 0 kills the coupling term outright, so nothing routing-related learns."""
    module = SpinModelTransformerModule(dim=64, num_steps=1, init="reset", beta=1.0)
    module(torch.randn(2, 16, 64)).magnetizations.sum().backward()
    assert module.to_qk.weight.grad.abs().max() == 0
    assert module.attn_temperature.grad == 0


def test_the_fixed_point_forgets_its_initialization():
    """At K -> inf the init is inert, so to_v receives no gradient at all."""
    module = SpinModelTransformerModule(
        dim=64, num_steps=None, init="amortized", beta=1.0
    )
    module(torch.randn(2, 16, 64)).magnetizations.sum().backward()
    # Not merely zero: the solve runs under no_grad and the implicit gradient is
    # reattached at the solution, so the init never enters the graph at all.
    assert module.to_v.weight.grad is None


def test_causal_masking_keeps_the_prefix_independent_of_the_suffix():
    module = SpinModelTransformerModule(
        dim=64, num_heads=2, num_steps=4, init="reset", beta=1.0, causal=True
    )
    x = torch.randn(1, 12, 64)
    changed = x.clone()
    changed[:, 6:] = torch.randn_like(changed[:, 6:])
    assert torch.allclose(
        module(x).magnetizations[:, :6], module(changed).magnetizations[:, :6], atol=1e-10
    )


def test_advance_realigns_the_window():
    state = SpinModelTransformerModule(dim=64, num_heads=2, num_steps=2).forward(
        torch.randn(1, 10, 64)
    ).state
    moved = advance(state)
    assert moved.magnetizations.shape == state.magnetizations.shape
    assert torch.allclose(moved.magnetizations[..., :-1, :], state.magnetizations[..., 1:, :])
    assert torch.all(moved.magnetizations[..., -1, :] == 0)


def test_relaxation_traces_a_converging_path():
    module = SpinModelTransformerModule(dim=64, num_heads=2, num_steps=4, beta=1.0)
    trace = module.relaxation(torch.randn(1, 16, 64), num_steps=48)
    assert trace.magnetizations.shape[0] == 49
    # Leading axis is k, the rest are batch and head.
    assert float(trace.residual[-1].max()) < float(trace.residual[0].max())
    assert float(trace.mismatch[-1].max()) < float(trace.mismatch[0].max())
    # Housekeeping cost is what survives at the steady state, so it must not
    # decay to zero the way the mismatch does.
    assert float(trace.entropy_production[-1].min()) > 0
