"""Solving ``m = f(m)``, diagnosing the solution, and differentiating it.

Nothing here knows about spins.  ``anderson`` accelerates the search for a fixed
point of any map on ``(..., N, D)`` and ``implicit_grad`` re-attaches the exact
gradient afterwards, so the same two functions serve the forward relaxation and
the linear adjoint problem in the backward pass.  ``local_contraction`` and
``distinct_fixed_points`` are the evidence that belongs beside every reported
fixed point: a converged residual alone says nothing about whether the point is
attracting or unique.

The trajectory of the physical relaxation is a different object and lives in
``mean_field.relax*``; the iterates here are solver states with no k attached.
"""

from __future__ import annotations

import warnings
from typing import Iterable, NamedTuple

import torch
from torch import Tensor


def residual(magnetizations: Tensor, previous: Tensor) -> float:
    """Worst-site update size, the convergence measure used throughout."""
    return float((magnetizations - previous).norm(dim=-1).max())


class Solve(NamedTuple):
    """A fixed point together with the evidence for it.

    ``residual`` is the worst-site update at the last accepted iterate; the
    returned ``solution`` is that iterate's image, so its true residual is
    smaller still wherever the map contracts.  Returning the evidence with the
    point is deliberate: a solution reported without its residual invites
    exactly the mistake of trusting an unconverged solve.
    """

    solution: Tensor
    residual: float
    converged: bool


def anderson(
    step_fn,
    initial: Tensor,
    *,
    max_iter: int = 40,
    tol: float = 1e-5,
    memory: int = 5,
    ridge: float = 1e-6,
) -> Solve:
    """Anderson-accelerated solve of ``m = step_fn(m)``, for ``initial`` of (..., N, D).

    Rather than taking the raw update, mix the last ``memory`` iterates with the
    weights that minimize the residual in least squares, subject to summing to
    one.  That reaches the same fixed point in far fewer evaluations than plain
    successive substitution ``m <- step_fn(m)`` -- Picard iteration, what
    ``relax*`` does -- which is why it is what runs when only the endpoint is
    wanted.

    The intermediate iterates are solver states, not the physical relaxation
    path: use ``relax*`` when the trajectory itself is the object of study.
    Which fixed point is reached still depends on ``initial`` wherever more than
    one exists, so this does not paper over basin structure.

    Leading axes are independent problems, solved in parallel; the residual and
    the convergence flag are taken over the worst of them.
    """
    sites, dim = initial.shape[-2:]
    batch_shape = initial.shape[:-2]
    width = sites * dim
    to_flat = lambda t: t.reshape(-1, width)
    to_state = lambda t: t.reshape(*batch_shape, sites, dim)

    memory = max(2, min(memory, max_iter))
    iterates = to_flat(initial).new_zeros(to_flat(initial).shape[0], memory, width)
    images = torch.zeros_like(iterates)
    iterates[:, 0] = to_flat(initial)
    images[:, 0] = to_flat(step_fn(initial))
    iterates[:, 1] = images[:, 0]
    images[:, 1] = to_flat(step_fn(to_state(images[:, 0])))

    # Bordered system: a row and column of ones impose sum(weights) = 1.
    system = iterates.new_zeros(iterates.shape[0], memory + 1, memory + 1)
    target = iterates.new_zeros(iterates.shape[0], memory + 1, 1)
    system[:, 0, 1:] = 1.0
    system[:, 1:, 0] = 1.0
    target[:, 0] = 1.0

    gap = lambda slot: float((images[:, slot] - iterates[:, slot]).norm(dim=-1).max())

    slot = 1
    for step in range(2, max_iter):
        # Tested before the solve, not after: on an already-converged problem
        # every gap is zero, the Gram matrix is singular, and the bordered
        # system has no unique solution.  An all-zero drive does exactly that.
        if gap(slot) < tol:
            break

        window = min(step, memory)
        gaps = images[:, :window] - iterates[:, :window]
        gram = gaps @ gaps.transpose(1, 2)
        scale = gram.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None].clamp_min(1e-30)
        eye = torch.eye(window, dtype=gram.dtype, device=gram.device)
        system[:, 1 : window + 1, 1 : window + 1] = gram + ridge * scale * eye
        weights = torch.linalg.solve(
            system[:, : window + 1, : window + 1], target[:, : window + 1]
        )[:, 1:, 0].unsqueeze(1)

        slot = step % memory
        iterates[:, slot] = (weights @ images[:, :window])[:, 0]
        images[:, slot] = to_flat(step_fn(to_state(iterates[:, slot])))
    final = gap(slot)
    return Solve(to_state(images[:, slot]), final, final < tol)


def implicit_grad(
    step_fn, solved: Tensor, *, max_iter: int = 40, tol: float = 1e-7
) -> Tensor:
    """Re-attach an exact gradient to a fixed point that was solved without one.

    At a fixed point ``z = f(z)`` the sensitivity is not the sensitivity of the
    solver: differentiating through the iterates would store every one of them,
    and answer a question about the path rather than about the solution.  The
    implicit function theorem gives it directly -- with ``u = dL/dz``, the
    quantity that must be propagated into ``f``'s parameters is

        adjoint = u (I - df/dz)^-1,

    which is itself the fixed point of ``a <- u + a df/dz`` and so can be found
    with vector-Jacobian products alone, no Jacobian ever formed.  One extra
    evaluation of ``f`` carries it into the parameters.

    The adjoint problem is *linear*, so the same Anderson solver that finds the
    forward fixed point accelerates it, one vector-Jacobian product per
    evaluation.  That matters because the adjoint series converges only under
    the same contraction that controls the forward map: as rho -> 1 both expire
    together, and a truncated adjoint is a silently wrong gradient.  So a
    warning is raised whenever the adjoint solve reports non-convergence -- the
    gradient counterpart of never reporting a branch count without a residual
    beside it.

    Costs O(1) memory in the number of solver steps, and is exact rather than
    the one-step approximation it replaces.  Double backward is not supported:
    the adjoint is computed outside the graph, so grad-of-grad through this
    fixed point would be silently wrong -- fail loudly if that is ever needed.
    """
    if not torch.is_grad_enabled():
        return solved

    # Two evaluations, deliberately.  ``output`` is the one on the real graph and
    # carries the gradient into the parameters.  ``image`` hangs off a detached
    # leaf and exists only to supply vector-Jacobian products. Keeping them
    # separate is what stops the hook from re-entering itself: differentiating
    # the tensor the hook is attached to would fire the hook again, and recurse
    # until memory runs out.
    output = step_fn(solved.detach())
    point = solved.detach().requires_grad_(True)
    image = step_fn(point)

    tiny = torch.finfo(output.dtype).tiny

    def backward(grad: Tensor) -> Tensor:
        def adjoint_step(a: Tensor) -> Tensor:
            return grad + torch.autograd.grad(image, point, a, retain_graph=True)[0]

        scale = float(grad.norm().clamp_min(tiny))
        solve = anderson(adjoint_step, grad, max_iter=max_iter, tol=tol * scale)
        if not solve.converged:
            warnings.warn(
                f"implicit gradient adjoint residual {solve.residual / scale:.2e} "
                f"> tol {tol:.2e}; the fixed-point map is likely not contracting "
                "(rho too close to 1) and this gradient is untrustworthy",
                stacklevel=2,
            )
        return solve.solution

    output.register_hook(backward)
    return output


def local_contraction(step_fn, point: Tensor, *, iterations: int = 60) -> float:
    """Spectral radius of d step_fn/dm at ``point``, by power iteration on VJPs.

    This is the number that actually decides whether the iteration converges:
    Lipschitz bounds like ``contraction_factor`` are evaluated at h = 0, and the
    response is concave, so a site sitting in a strong field is far less
    responsive than the bound allows.

    Meaningful only where ``point`` is genuinely a fixed point.  Measured at an
    iterate the solver never settled -- residual not at tolerance -- the power
    iteration rides a wandering point and reports rho ~ 1 regardless of beta,
    which reads as physics and is not.  Always check the residual first.
    """
    point = point.detach().clone().requires_grad_(True)
    image = step_fn(point)
    vector = torch.randn_like(point)
    vector = vector / vector.norm()
    value = 0.0
    for _ in range(iterations):
        product = torch.autograd.grad(image, point, vector, retain_graph=True)[0]
        value = float(product.norm())
        vector = product / product.norm().clamp_min(
            torch.finfo(product.dtype).tiny
        )
    return value


def distinct_fixed_points(
    step_fn,
    starts: Iterable[Tensor],
    *,
    max_iter: int = 200,
    tol: float = 1e-13,
    residual_tol: float = 1e-8,
    match_tol: float = 1e-3,
) -> list[Tensor]:
    """The distinct converged fixed points reached from the given starts.

    Anderson solves the *root-finding* problem, so it converges to repelling
    fixed points as happily as attracting ones: every point returned here still
    needs ``local_contraction`` beside it before it can be called physical.
    But more than one entry is already a diagnosis -- the map is multistable,
    and any single-start "error" measured on it is silently a statement about
    which basin the solver fell into.

    Solutions whose residual misses ``residual_tol`` are dropped rather than
    counted: an unconverged iterate is not a branch.
    """
    found: list[Tensor] = []
    for start in starts:
        solve = anderson(step_fn, start, max_iter=max_iter, tol=tol)
        if solve.residual > residual_tol:
            continue
        if not any(
            float((solve.solution - other).norm()) < match_tol for other in found
        ):
            found.append(solve.solution)
    return found
