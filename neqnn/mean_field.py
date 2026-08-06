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
"""

from __future__ import annotations

import torch
from einops import einsum, rearrange
from torch import Tensor

from neqnn import vmf
from neqnn.fixed_point import residual

#
# Magnetizations
#


def effective_field(magnetizations: Tensor, drive: Tensor, couplings: Tensor) -> Tensor:
    """h_i = x_i + sum_j J_ij m_j, the mean-field counterpart of the sampled field."""
    return drive + einsum(couplings, magnetizations, "... i j, ... j d -> ... i d")


def step(
    magnetizations: Tensor, drive: Tensor, couplings: Tensor, beta: float
) -> Tensor:
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
    relaxation diagnostics be read off as a function of k. With ``tol`` set the
    iteration stops early once the update is smaller than it, so the leading
    axis is the realised horizon rather than the cutoff ``num_steps``.
    """
    return _relax(
        step, magnetizations, drive, couplings, beta, num_steps=num_steps, tol=tol
    )


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
        step_large_d,
        magnetizations,
        drive,
        couplings,
        beta,
        num_steps=num_steps,
        tol=tol,
    )


#
# Contraction bound of magnetization map
#


def contraction_factor(couplings: Tensor, beta: float, dim: int) -> float:
    """Lipschitz bound rho = beta (R^2/D) max_i sum_j |J_ij| on the exact map.

    The exact response has gain ``beta R^2 A_D'(0) = beta (D/2 - 1)/D``, attained
    at h=0 since A_D is concave.
    """
    return beta * vmf.order(dim) / dim * float(couplings.abs().sum(-1).max())


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

    The exact vMF covariance is isotropic plus rank one,
    ``Sigma = tau I + (rho - tau) mu mu^T``. Expanding that structure before
    taking the pairwise traces costs O(N^2 D), without constructing any D x D
    covariance matrices.
    """
    dim = field.shape[-1]
    tau1, rho1 = vmf.variances(field, beta)
    tau0, rho0 = vmf.variances(previous_field, beta)
    delta1, delta0 = rho1 - tau1, rho0 - tau0
    tiny = torch.finfo(field.dtype).tiny
    direction1 = field / field.norm(dim=-1, keepdim=True).clamp_min(tiny)
    direction0 = previous_field / previous_field.norm(dim=-1, keepdim=True).clamp_min(
        tiny
    )
    gram = einsum(direction1, direction0, "... i d, ... j d -> ... i j")
    row = lambda t: rearrange(t, "... i -> ... i 1")
    col = lambda t: rearrange(t, "... j -> ... 1 j")
    return (
        dim * row(tau1) * col(tau0)
        + row(tau1) * col(delta0)
        + row(delta1) * col(tau0)
        + row(delta1) * col(delta0) * gram**2
    )


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


def _covariance_parts_large_d(
    field: Tensor, beta: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Write the large-D covariance as ``Sigma = a I - b m m^T``."""
    r2 = vmf.order(field.shape[-1])
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
