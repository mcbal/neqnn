"""Single-site vMF mathematics: Bessel routines, moments, sampler, large-D forms."""

from __future__ import annotations

import math

import numpy
import pytest
import torch

from helpers import DIMS, random_problem, relative, slope_in_dim
from neqnn import stochastic, vmf


def test_order_and_radius_enforce_the_convention():
    for dim in DIMS:
        assert vmf.radius(dim) == pytest.approx(math.sqrt(dim / 2 - 1))
    for dim in (0, 1, 2):
        with pytest.raises(ValueError):
            vmf.order(dim)


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


def test_magnetization_squash_is_bounded_and_locally_identity():
    dim = 64
    value = torch.randn(8, dim)
    squashed = vmf.magnetization_squash(value)
    assert float(squashed.norm(dim=-1).max()) < vmf.radius(dim)
    assert relative(vmf.magnetization_squash(value * 1e-4), value * 1e-4) < 1e-7
    for beta in (0.5, 1.0, 2.0):
        assert torch.allclose(
            squashed, vmf.response_large_d(2 * value / beta, beta)
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_inverse_large_d_response_round_trips_the_open_ball(dtype, beta):
    dim = 32
    direction = torch.nn.functional.normalize(
        torch.randn(5, dim, dtype=dtype), dim=-1
    )
    fractions = torch.tensor([0.0, 0.1, 0.5, 0.9, 0.999], dtype=dtype)
    magnetization = direction * (fractions * vmf.radius(dim))[:, None]
    field = vmf.inverse_response_large_d(magnetization, beta)
    recovered = vmf.response_large_d(field, beta)
    tolerance = 3e-5 if dtype == torch.float32 else 1e-11
    assert torch.allclose(recovered, magnetization, rtol=tolerance, atol=tolerance)


def test_inverse_large_d_response_rejects_nonphysical_inputs():
    dim = 32
    boundary = torch.zeros(2, dim)
    boundary[0, 0] = vmf.radius(dim)
    boundary[1, 0] = 1.01 * vmf.radius(dim)
    with pytest.raises(ValueError, match="strictly inside"):
        vmf.inverse_response_large_d(boundary, 1.0)
    with pytest.raises(ValueError, match="finite values"):
        vmf.inverse_response_large_d(torch.full((1, dim), torch.nan), 1.0)
    with pytest.raises(TypeError, match="floating point"):
        vmf.inverse_response_large_d(torch.ones(1, dim, dtype=torch.int64), 1.0)
    with pytest.raises(ValueError, match="finite and positive"):
        vmf.inverse_response_large_d(torch.zeros(1, dim), 0.0)


@pytest.mark.parametrize("quantity", ["response", "covariance", "kl"])
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
        else:
            pair = vmf.kl_large_d(field, other, beta), vmf.kl(field, other, beta)
        errors.append(relative(*pair))
    assert slope_in_dim(errors) < -0.9
    assert errors[-1] < 0.01
