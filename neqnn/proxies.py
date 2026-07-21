"""Nonequilibrium diagnostics: what we actually measure during relaxation.

Two proxies, both deliberately written to take plain tensors rather than a
system object, so the same function reads a sampled trajectory or a mean-field
one.  Where they get their arguments from is the caller's business:

- ``housekeeping_entropy_production`` wants couplings and the geometry factor
  C*_ij, from ``mean_field.covariance_traces`` or from sampled covariances via
  ``covariance_traces_from_samples``.
- ``mismatch`` wants two effective fields, one of them the frozen-drive fixed
  point.

They measure opposite things and it matters not to conflate them.  The fixed
point reached under a frozen drive is a *nonequilibrium* steady state: softmax
couplings are generically asymmetric, so detailed balance is broken, probability
currents persist, and sigma_hk stays strictly positive there forever -- that is
what makes it housekeeping rather than transient.  Symmetric couplings would
restore detailed balance and genuine equilibrium, at which point sigma_hk
vanishes and nothing interesting happens.

``mismatch`` is the excess riding on top of that.  In the microscopic chain it
has to decay to zero, since the chain converges to its stationary law.  Under
the mean-field recurrence it does not have to, and we should not assume it:
whether it decreases from one iterate to the next, and whether it reaches zero
at all, is a property of the iteration to be measured rather than asserted.
Once the map stops contracting it can stall, ring, or settle elsewhere.

Throughout, "fixed point" means the mean-field approximation to the single-site
marginals of the chain's stationary law, not the stationary law itself.
"""

from __future__ import annotations

from einops import einsum
from torch import Tensor

from neqnn import vmf


def housekeeping_entropy_production(
    couplings: Tensor, covariance_traces: Tensor, beta: float
) -> Tensor:
    """sigma_hk = (beta^2/2) sum_ij (J_ij - J_ji)^2 C*_ij.

    The irreversibility the system keeps paying once it has settled, carried
    entirely by the asymmetric part of the coupling rule.  It does not relax
    away: at the fixed point of an asymmetric J it is positive and stays there,
    which is the whole point of calling it housekeeping.  Symmetrizing J kills
    the ``(J_ij - J_ji)^2`` factor exactly and is the natural control, since
    that is the detailed-balance case where the dynamics genuinely stops.
    """
    asymmetry = couplings - couplings.transpose(-1, -2)
    return 0.5 * beta**2 * (asymmetry**2 * covariance_traces).sum((-2, -1))


def covariance_traces_from_samples(covariances: Tensor) -> Tensor:
    """C*_ij = Tr(Sigma_i Sigma_j) from sampled single-site covariances (N, D, D)."""
    return einsum(covariances, covariances, "... i d e, ... j e d -> ... i j")


def mismatch(field: Tensor, steady_field: Tensor, beta: float, exact: bool = False) -> Tensor:
    """Post-quench mismatch Delta = KL(p_k || p*), summed over sites.

    The relaxation coordinate: how far the current single-site laws sit from the
    ones at the fixed point of the frozen-drive recurrence.  It is zero exactly
    when the two fields agree, and penalizes norm and angular departure
    together, so one scalar tracks the whole approach.  That it shrinks
    monotonically in k is a hypothesis about the mean-field iteration, not a
    property of this function.

    ``exact`` swaps the large-D expansion for the Bessel-based KL, which is what
    tells us whether the cheap form is safe at a given D.
    """
    kl = vmf.kl if exact else vmf.kl_large_d
    return kl(field, steady_field, beta).sum(-1)
