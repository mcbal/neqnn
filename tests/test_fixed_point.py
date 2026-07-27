"""The solver and its evidence: Anderson, implicit gradients, rho, branch counts."""

from __future__ import annotations

import pytest
import torch

from helpers import random_problem, relative
from neqnn import fixed_point as fp, mean_field as mf


@pytest.mark.parametrize("large_d", [False, True])
def test_anderson_finds_the_same_point_as_picard(large_d):
    dim, beta = 64, 1.0
    drive, couplings = random_problem(dim)
    relax = mf.relax_large_d if large_d else mf.relax
    step = mf.step_large_d if large_d else mf.step
    step_fn = lambda m: step(m, drive, couplings, beta)

    picard = relax(torch.zeros_like(drive), drive, couplings, beta, num_steps=500)[-1]
    solve = fp.anderson(step_fn, torch.zeros_like(drive), max_iter=50, tol=1e-12)
    assert solve.converged
    assert solve.residual < 1e-12
    assert fp.residual(step_fn(solve.solution), solve.solution) < 1e-10
    assert relative(solve.solution, picard) < 1e-8


def test_anderson_reports_non_convergence_honestly():
    """Starved of iterations the solve must say so, not hand back a clean tensor."""
    dim, beta = 64, 1.0
    drive, couplings = random_problem(dim)
    step_fn = lambda m: mf.step_large_d(m, drive, couplings, beta)
    solve = fp.anderson(step_fn, torch.zeros_like(drive), max_iter=3, tol=1e-12)
    assert not solve.converged
    assert solve.residual > 1e-12
    # Evidence must describe the exact tensor returned, including outside the
    # regime where applying the map once more is guaranteed to improve it.
    assert solve.residual == pytest.approx(
        fp.residual(step_fn(solve.solution), solve.solution)
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_iter": 1}, "max_iter"),
        ({"memory": 1}, "memory"),
        ({"tol": 0}, "tol"),
        ({"ridge": -1}, "ridge"),
    ],
)
def test_anderson_validates_solver_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        fp.anderson(lambda x: x, torch.zeros(2, 3), **kwargs)


def test_anderson_handles_an_already_converged_problem():
    """Zero drive and zero start: every gap vanishes and the Gram matrix is singular."""
    dim = 64
    drive = torch.zeros(8, dim)
    couplings = torch.softmax(torch.randn(8, 8), -1)
    step_fn = lambda m: mf.step_large_d(m, drive, couplings, 1.0)
    solve = fp.anderson(step_fn, torch.zeros_like(drive), max_iter=30, tol=1e-10)
    assert solve.converged
    assert torch.allclose(solve.solution, torch.zeros_like(solve.solution))


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
        ).solution
    actual = torch.autograd.grad(fp.implicit_grad(step_fn, solved).sum(), leaves)

    for got, want in zip(actual, expected):
        assert relative(got, want) < 1e-6


def test_local_contraction_recovers_a_known_spectral_radius():
    """On a linear map the power iteration must return the top |eigenvalue|."""
    sites, dim = 6, 4
    torch.manual_seed(2)
    matrix = torch.randn(sites * dim, sites * dim)
    matrix = 0.5 * (matrix + matrix.T)
    rho = float(torch.linalg.eigvalsh(matrix).abs().max())
    step_fn = lambda m: (matrix @ m.reshape(-1)).reshape(sites, dim)
    measured = fp.local_contraction(step_fn, torch.zeros(sites, dim), iterations=300)
    assert measured == pytest.approx(rho, rel=1e-3)


def test_distinct_fixed_points_counts_branches_and_drops_unconverged():
    """m -> tanh(3m) has fixed points at 0 and at +-a; starts select among them."""
    shape = (4, 3)
    step_fn = lambda m: torch.tanh(3 * m)
    starts = [torch.full(shape, v) for v in (1.0, 2.0, -1.0)]
    found = fp.distinct_fixed_points(step_fn, starts)
    assert len(found) == 2  # the two ordered branches; 1.0 and 2.0 share a basin

    # The repelling point at zero is still a genuine root, and an exact start
    # lands on it: whether it is *physical* is local_contraction's question.
    found = fp.distinct_fixed_points(step_fn, starts + [torch.zeros(shape)])
    assert len(found) == 3

    # A start that cannot converge within the budget must not be counted.
    assert (
        fp.distinct_fixed_points(step_fn, [torch.full(shape, 0.7)], max_iter=3) == []
    )
