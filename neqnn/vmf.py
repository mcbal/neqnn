"""von Mises-Fisher mathematics for vector spins on a sphere of radius R.

A spin is ``s = R u`` with the unit vector ``u`` distributed as

    p(u | mu, kappa) = C_D(kappa) exp(kappa mu . u),

so a site sitting in effective field ``h`` at inverse temperature ``beta`` has
mean direction ``mu = h / ||h||`` and concentration ``kappa = beta R ||h||``.

The radius convention ``R^2 = D/2 - 1`` is what makes the large-D response
collapse to the closed form used inside the network module.  Every quantity
here comes in two flavours: an exact one built on Bessel ratios, and a
``_large_d`` one that drops the Bessels entirely.  Comparing the two as a
function of D is the point of the first experiment.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

TINY = 1e-30


def radius(dim: int) -> float:
    """Spin radius R, fixed by the convention R^2 = D/2 - 1."""
    if dim <= 2:
        raise ValueError(f"radius convention requires dim > 2, got {dim}")
    return math.sqrt(dim / 2 - 1)


def concentration(field: Tensor, beta: float) -> Tensor:
    """vMF concentration kappa = beta R ||h|| of a site in field ``h``."""
    return beta * radius(field.shape[-1]) * field.norm(dim=-1)


def gamma(field: Tensor, beta: float) -> Tensor:
    """Local stiffness gamma = sqrt(1 + beta^2 ||h||^2 / R^2), keeping the vector axis."""
    r2 = field.shape[-1] / 2 - 1
    return torch.sqrt(1 + beta**2 * field.pow(2).sum(-1, keepdim=True) / r2)


#
# Bessel ratios
#


def bessel_ratio(x: Tensor, order: float, num_iter: int = 64) -> Tensor:
    """Ratio I_{order+1}(x) / I_{order}(x) by backward recurrence in the order.

    The recurrence ``r_{v-1} = x / (2v + x r_v)`` follows from
    ``I_{v-1} - I_{v+1} = (2v/x) I_v`` and contracts by ``r^2 < 1`` per step, so
    running it downwards is the stable direction.  It is seeded ``num_iter``
    orders up with the uniform estimate of Amos (1974), which is already close
    enough that the recurrence only has to polish.

    Convergence is set by the ratio ``u = x / order``, not by ``x`` alone.  For
    us ``x = kappa = beta R ||h||`` and ``order = R^2``, so ``u = beta ||h|| / R``.
    Note kappa itself is *not* small: it grows like ``beta D / 2``, reaching
    ~255 at D=512.  That is harmless.  Since ``||h||`` stays O(R) -- inputs are
    normalized to the sphere of radius R, softmax couplings are row-stochastic
    and ``||m|| <= R`` -- ``u`` stays O(beta) at any D, and 32 iterations reach
    machine precision across u in [0.1, 5].  The default doubles that.  Only
    ``u >> 10``, a deep low-temperature corner, would need more.

    Working with the ratio directly also dodges an underflow that kills the
    obvious ``ive(order+1, x) / ive(order, x)``: at D=512, x=0.01 both Bessels
    flush to zero, while the recurrence still returns the correct ``x/D``.
    """
    top = order + num_iter
    r = x / (top + 0.5 + torch.sqrt((top + 1.5) ** 2 + x**2))
    for k in range(num_iter, 0, -1):
        r = x / (2 * (order + k) + x * r)
    return r


def mean_resultant(kappa: Tensor, dim: int) -> Tensor:
    """A_D(kappa) = I_{D/2}(kappa) / I_{D/2-1}(kappa) = E[mu . u]."""
    return bessel_ratio(kappa, dim / 2 - 1)


def mean_resultant_large_d(kappa: Tensor, dim: int) -> Tensor:
    """Large-D form of A_D, obtained by dropping O(1) terms in the Amos estimate."""
    r2 = dim / 2 - 1
    return kappa / (r2 * (1 + torch.sqrt(1 + (kappa / r2) ** 2)))


def log_normalizer(kappa: Tensor, dim: int, num_nodes: int = 64) -> Tensor:
    """log C_D(kappa) - log C_D(0), by quadrature of d/dkappa log C_D = -A_D.

    Only differences of log-normalizers are ever observable, so the kappa=0
    constant is dropped rather than computed.  This also means we never need
    log I_nu itself, just the ratio above.

    A_D is smooth and monotone, so Gauss-Legendre converges spectrally: at the
    kappa reached in practice this is at machine precision from 32 nodes up.
    """
    nodes, weights = np.polynomial.legendre.leggauss(num_nodes)
    nodes = torch.as_tensor(nodes, dtype=kappa.dtype, device=kappa.device)
    weights = torch.as_tensor(weights, dtype=kappa.dtype, device=kappa.device)
    half = 0.5 * kappa.unsqueeze(-1)
    integrand = mean_resultant(half * (nodes + 1), dim)
    return -(half.squeeze(-1) * (weights * integrand).sum(-1))


#
# Single-site moments
#


def response(field: Tensor, beta: float) -> Tensor:
    """Magnetization m = E[s] of a site in effective field ``h``."""
    dim = field.shape[-1]
    norm = field.norm(dim=-1, keepdim=True)
    amplitude = radius(dim) * mean_resultant(beta * radius(dim) * norm, dim)
    return amplitude * field / norm.clamp_min(TINY)


def response_large_d(field: Tensor, beta: float) -> Tensor:
    """Large-D magnetization, the form the network module actually iterates."""
    return beta * field / (1 + gamma(field, beta))


def variances(field: Tensor, beta: float) -> tuple[Tensor, Tensor]:
    """Tangential and radial variances of a single spin.

    The covariance is ``Sigma = tau I + (rho - tau) mu mu^T`` with ``tau`` the
    variance along any direction orthogonal to the field and ``rho`` the
    variance along it.
    """
    dim = field.shape[-1]
    r2 = dim / 2 - 1
    kappa = concentration(field, beta)
    resultant = mean_resultant(kappa, dim)
    # A_D(kappa) / kappa -> 1/D as kappa -> 0.
    ratio = torch.where(
        kappa < 1e-6,
        torch.full_like(kappa, 1.0 / dim),
        resultant / kappa.clamp_min(TINY),
    )
    tangential = r2 * ratio
    radial = r2 * (1 - (dim - 1) * ratio - resultant**2)
    return tangential, radial


def variances_large_d(field: Tensor, beta: float) -> tuple[Tensor, Tensor]:
    """Large-D tangential and radial variances, 1/(1+gamma) and 1/(gamma(1+gamma)).

    The radial one has to come from differentiating the large-D A_D rather than
    from expanding the exact ``rho`` term by term: the exact expression is a
    cancellation that only survives at next order.
    """
    g = gamma(field, beta).squeeze(-1)
    return 1 / (1 + g), 1 / (g * (1 + g))


def _covariance_from_variances(
    field: Tensor, tangential: Tensor, radial: Tensor
) -> Tensor:
    dim = field.shape[-1]
    direction = field / field.norm(dim=-1, keepdim=True).clamp_min(TINY)
    eye = torch.eye(dim, dtype=field.dtype, device=field.device)
    outer = direction.unsqueeze(-1) * direction.unsqueeze(-2)
    return tangential[..., None, None] * eye + (radial - tangential)[..., None, None] * outer


def covariance(field: Tensor, beta: float) -> Tensor:
    """Single-site covariance Sigma = Cov[s], exact vMF."""
    return _covariance_from_variances(field, *variances(field, beta))


def covariance_large_d(field: Tensor, beta: float) -> Tensor:
    """Single-site covariance in the large-D form ``I/(1+gamma) - m m^T/(R^2 gamma)``."""
    return _covariance_from_variances(field, *variances_large_d(field, beta))


#
# Relative entropy between two single-site distributions
#


def kl(field_p: Tensor, field_q: Tensor, beta: float) -> Tensor:
    """KL(p || q) between the vMF laws induced by two effective fields.

    Since ``kappa mu = beta R h``, the cross term collapses to
    ``beta (h_p - h_q) . m_p`` and the rest is a normalizer difference.
    """
    dim = field_p.shape[-1]
    log_ratio = log_normalizer(concentration(field_p, beta), dim) - log_normalizer(
        concentration(field_q, beta), dim
    )
    return log_ratio + beta * ((field_p - field_q) * response(field_p, beta)).sum(-1)


def kl_large_d(field_p: Tensor, field_q: Tensor, beta: float) -> Tensor:
    """Large-D KL(p || q), expanded to second order around q.

    This is the post-quench mismatch proxy: with ``q`` the frozen-drive steady
    state it penalizes both norm and angular departure of ``p`` from it, and
    vanishes exactly at convergence.
    """
    r2 = field_p.shape[-1] / 2 - 1
    m_p, m_q = response_large_d(field_p, beta), response_large_d(field_q, beta)
    norm_p, norm_q = m_p.pow(2).sum(-1), m_q.pow(2).sum(-1)
    slack = r2 - norm_q
    return r2 * torch.log(slack / (r2 - norm_p)) + 2 * r2 * (
        norm_q - (m_p * m_q).sum(-1)
    ) / slack


#
# Sampling
#


def _uniform_sphere(shape: tuple[int, ...], dim: int, **kwargs) -> Tensor:
    values = torch.randn(*shape, dim, **kwargs)
    return values / values.norm(dim=-1, keepdim=True)


def sample(mean_direction: Tensor, kappa: Tensor) -> Tensor:
    """Draw one unit vector per leading index from vMF, using Wood (1994).

    Seed with ``torch.manual_seed`` -- no generator is threaded through.
    """
    dim = mean_direction.shape[-1]
    shape = torch.broadcast_shapes(mean_direction.shape[:-1], kappa.shape)
    mu = mean_direction.expand(*shape, dim).reshape(-1, dim)
    kappa = kappa.expand(shape).reshape(-1)
    kwargs = dict(dtype=mu.dtype, device=mu.device)

    dm1 = dim - 1
    # Algebraically Wood's b, but written to avoid cancellation at small kappa.
    b = dm1 / (2 * kappa + torch.sqrt(4 * kappa**2 + dm1**2))
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + dm1 * torch.log1p(-x0 * x0)

    beta_dist = torch.distributions.Beta(
        torch.tensor(0.5 * dm1, **kwargs), torch.tensor(0.5 * dm1, **kwargs)
    )
    w = torch.zeros_like(kappa)
    pending = torch.ones_like(kappa, dtype=torch.bool)
    while pending.any():
        z = beta_dist.sample((int(pending.sum()),))
        proposal = (1 - (1 + b[pending]) * z) / (1 - (1 - b[pending]) * z)
        accept = kappa[pending] * proposal + dm1 * torch.log1p(
            -x0[pending] * proposal
        ) - c[pending] >= torch.rand(z.shape, **kwargs).log()
        index = pending.nonzero(as_tuple=True)[0]
        w[index[accept]] = proposal[accept]
        pending[index[accept]] = False

    tangent = _uniform_sphere((w.shape[0],), dim - 1, **kwargs)
    canonical = torch.cat(
        [w.unsqueeze(-1), (1 - w * w).clamp_min(0).sqrt().unsqueeze(-1) * tangent],
        dim=-1,
    )

    # Householder reflection carrying e_1 onto each requested mean direction.
    reflection = torch.zeros_like(mu)
    reflection[:, 0] = 1.0
    reflection = reflection - mu
    denom = reflection.pow(2).sum(-1, keepdim=True)
    projected = (reflection * canonical).sum(-1, keepdim=True)
    rotated = torch.where(
        denom > 1e-28, canonical - 2 * reflection * projected / denom.clamp_min(TINY), canonical
    )
    return rotated.reshape(*shape, dim)


def sample_from_field(field: Tensor, beta: float) -> Tensor:
    """Draw scaled spins ``s = R u`` for sites sitting in effective field ``h``."""
    norm = field.norm(dim=-1, keepdim=True)
    direction = field / norm.clamp_min(TINY)
    r = radius(field.shape[-1])
    return r * sample(direction, beta * r * norm.squeeze(-1))
