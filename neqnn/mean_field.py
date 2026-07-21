"""Naive mean-field (Plefka[t-1,t]) recurrences for the vector-spin system.

The stochastic update is replaced by its deterministic average: each site sees
the magnetizations of the others instead of their fluctuating spins, so a
trajectory is the iteration of

    m_{k+1} = phi(x + J m_k).

Every quantity comes in two flavours, matching ``vmf``: a plain name built on
the exact vMF expressions, and a ``_large_d`` one built on the closed forms that
drop the Bessels.  The large-D pair is what the network module iterates; the
exact pair is the reference it is measured against, and it stays usable at
D=3-ish where the vectors can simply be plotted.

Exact and large-D expressions are never mixed inside a single formula.  They
differ at O(1) in places that look harmless -- most importantly
``Tr(Sigma_largeD) = (R^2 - ||m||^2) + 1/gamma`` -- so a hybrid silently sits
between the two approximations and is a controlled limit of neither.
"""

from __future__ import annotations

import torch
from einops import einsum, rearrange
from torch import Tensor

from neqnn import vmf


def effective_field(magnetizations: Tensor, drive: Tensor, couplings: Tensor) -> Tensor:
    """h_i = x_i + sum_j J_ij m_j, the mean-field counterpart of the sampled field.

    Flavour-independent: only the response applied to it differs.
    """
    return drive + einsum(couplings, magnetizations, "... i j, ... j d -> ... i d")


def step(magnetizations: Tensor, drive: Tensor, couplings: Tensor, beta: float) -> Tensor:
    """One mean-field iteration using the exact vMF response."""
    return vmf.response(effective_field(magnetizations, drive, couplings), beta)


def step_large_d(
    magnetizations: Tensor, drive: Tensor, couplings: Tensor, beta: float
) -> Tensor:
    """One mean-field iteration using the large-D response."""
    return vmf.response_large_d(effective_field(magnetizations, drive, couplings), beta)


def _relax(
    step_fn,
    magnetizations: Tensor,
    drive: Tensor,
    couplings: Tensor,
    beta: float,
    *,
    num_steps: int,
    tol: float | None,
) -> Tensor:
    trajectory = [magnetizations]
    for _ in range(num_steps):
        trajectory.append(step_fn(trajectory[-1], drive, couplings, beta))
        if tol is not None and residual(trajectory[-1], trajectory[-2]) < tol:
            break
    return torch.stack(trajectory)


def relax(
    magnetizations: Tensor,
    drive: Tensor,
    couplings: Tensor,
    beta: float,
    *,
    num_steps: int,
    tol: float | None = None,
) -> Tensor:
    """Iterate the exact mean-field map, returning the trajectory (K+1, ..., N, D).

    Returning every iterate rather than just the endpoint is what lets the
    relaxation diagnostics be read off as a function of k.  With ``tol`` set the
    iteration stops early once the update is smaller than it, so the leading
    axis is the realised horizon rather than ``num_steps``.
    """
    return _relax(step, magnetizations, drive, couplings, beta, num_steps=num_steps, tol=tol)


def relax_large_d(
    magnetizations: Tensor,
    drive: Tensor,
    couplings: Tensor,
    beta: float,
    *,
    num_steps: int,
    tol: float | None = None,
) -> Tensor:
    """Iterate the large-D mean-field map, returning the trajectory (K+1, ..., N, D)."""
    return _relax(
        step_large_d, magnetizations, drive, couplings, beta, num_steps=num_steps, tol=tol
    )


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


def contraction_factor(couplings: Tensor, beta: float, dim: int) -> float:
    """Lipschitz bound rho = beta (R^2/D) max_i sum_j |J_ij| on the exact map.

    The exact response has gain ``beta R^2 A_D'(0) = beta (D/2 - 1)/D``, attained
    at h=0 since A_D is concave.  rho < 1 gives a unique fixed point reached
    geometrically.
    """
    return beta * (dim / 2 - 1) / dim * float(couplings.abs().sum(-1).max())


def contraction_factor_large_d(couplings: Tensor, beta: float) -> float:
    """Lipschitz bound rho = (beta/2) max_i sum_j |J_ij| on the large-D map.

    The same bound as ``contraction_factor`` with ``R^2/D -> 1/2``, so it is
    slightly the looser of the two at finite D.
    """
    return 0.5 * beta * float(couplings.abs().sum(-1).max())


#
# Delayed correlations
#


def covariance_traces(field: Tensor, previous_field: Tensor, beta: float) -> Tensor:
    """C*_ij = Tr(Sigma_i Sigma_j) from exact vMF covariances, shape (..., N, N).

    Costs O(N^2 D^2) by contracting the covariance matrices directly, which is
    fine for reference work but is why the large-D version below exists.
    """
    later = vmf.covariance(field, beta)
    earlier = vmf.covariance(previous_field, beta)
    return einsum(later, earlier, "... i d e, ... j e d -> ... i j")


def covariance_traces_large_d(
    field: Tensor, previous_field: Tensor, beta: float
) -> Tensor:
    """C*_ij = Tr(Sigma_i Sigma_j) in the large-D form, shape (..., N, N).

    Expanding ``Sigma = a I - b m m^T`` on both legs turns the trace into a Gram
    matrix, so this costs O(N^2 D) instead of O(N^2 D^2).
    """
    dim = field.shape[-1]
    a1, b1, m1 = _covariance_parts_large_d(field, beta)
    a0, b0, m0 = _covariance_parts_large_d(previous_field, beta)
    gram = einsum(m1, m0, "... i d, ... j d -> ... i j")
    row = lambda t: rearrange(t, "... i -> ... i 1")
    col = lambda t: rearrange(t, "... j -> ... 1 j")
    return (
        dim * row(a1) * col(a0)
        - row(a1) * col(b0 * m0.pow(2).sum(-1))
        - col(a0) * row(b1 * m1.pow(2).sum(-1))
        + row(b1) * col(b0) * gram**2
    )


def _covariance_parts_large_d(field: Tensor, beta: float) -> tuple[Tensor, Tensor, Tensor]:
    """Write the large-D covariance as ``Sigma = a I - b m m^T``."""
    r2 = field.shape[-1] / 2 - 1
    stiffness = vmf.gamma(field, beta).squeeze(-1)
    return 1 / (1 + stiffness), 1 / (r2 * stiffness), vmf.response_large_d(field, beta)


def delayed_correlations(
    field: Tensor, previous_field: Tensor, couplings: Tensor, beta: float
) -> Tensor:
    """Exact <s_i(t+1) s_j(t)>_c = beta J_ij C*_ij, shape (..., N, N)."""
    return beta * couplings * covariance_traces(field, previous_field, beta)


def delayed_correlations_large_d(
    field: Tensor, previous_field: Tensor, couplings: Tensor, beta: float
) -> Tensor:
    """Large-D <s_i(t+1) s_j(t)>_c = beta J_ij C*_ij, shape (..., N, N)."""
    return beta * couplings * covariance_traces_large_d(field, previous_field, beta)
