from __future__ import annotations

import warnings

import torch
from einops import einsum, rearrange
from torch import Tensor


def residual(magnetizations: Tensor, previous: Tensor) -> float:
    """Worst-site update size, the convergence measure used throughout."""
    return float((magnetizations - previous).norm(dim=-1).max())


def anderson(
    step_fn,
    initial: Tensor,
    *,
    max_iter: int = 40,
    tol: float = 1e-5,
    memory: int = 5,
    damping: float = 1.0,
    ridge: float = 1e-6,
) -> Tensor:
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

    Leading axes are independent problems, solved in parallel.
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

    slot = 1
    for step in range(2, max_iter):
        # Tested before the solve, not after: on an already-converged problem
        # every gap is zero, the Gram matrix is singular, and the bordered
        # system has no unique solution.  An all-zero drive does exactly that.
        if (images[:, slot] - iterates[:, slot]).norm(dim=-1).max() < tol:
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
        iterates[:, slot] = (
            damping * (weights @ images[:, :window])[:, 0]
            + (1 - damping) * (weights @ iterates[:, :window])[:, 0]
        )
        images[:, slot] = to_flat(step_fn(to_state(iterates[:, slot])))
    return to_state(images[:, slot])


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
    together, and a truncated adjoint is a silently wrong gradient.  So the
    residual is checked after the solve and a warning raised when it is not
    met -- the gradient counterpart of never reporting a branch count without a
    residual beside it.

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
        adjoint = anderson(adjoint_step, grad, max_iter=max_iter, tol=tol * scale)
        residual = float((adjoint_step(adjoint) - adjoint).norm()) / scale
        if residual > tol:
            warnings.warn(
                f"implicit gradient adjoint residual {residual:.2e} > tol {tol:.2e}; "
                "the fixed-point map is likely not contracting (rho too close to 1) "
                "and this gradient is untrustworthy",
                stacklevel=2,
            )
        return adjoint

    output.register_hook(backward)
    return output
