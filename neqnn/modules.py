"""The spin-model transformer module.

A transformer block whose forward pass *is* the relaxation of a vector-spin
system.  Heads split the vector dimension, so each head carries its own
independent spin system of dimension ``dim // num_heads`` -- that head dimension,
not ``dim``, is the D the large-D approximation is controlled by.

Two independent binary choices span the design space:

                    | reset / amortized init   | carried init
    ----------------|--------------------------|---------------------------
    finite K        | stateless, transformer   | recurrent, stateful
    fixed point     | implicit, DEQ-like       | path-dependent branch choice

``num_steps`` picks the row (an int, or ``None`` for the fixed point) and
``init`` picks the column.  Nothing else changes between quadrants.

State is explicit: ``forward`` takes the previous state and returns the next one
rather than mutating anything, so which quadrant is running is the caller's
choice and batching stays honest.

A note on ``post_mix``.  Mixing head outputs is what a transformer does, but
here the outputs *are* magnetizations, and a linear map of them is no longer the
magnetization of anything -- it has left the mean-field state space.  So it is
applied to the readout only, never to the state that gets carried or fed to the
diagnostics.  Turning it on makes the module more transformer-like and its
output less physically interpretable, and that trade is the whole reason it is
a flag rather than a default.
"""

from __future__ import annotations

from functools import partial
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from neqnn import mean_field as mf, proxies, vmf


class MeanFieldState(NamedTuple):
    """Magnetizations carried across drive steps t, shaped (b, heads, n, dim_head)."""

    magnetizations: Tensor


class Readout(NamedTuple):
    magnetizations: Tensor
    state: MeanFieldState
    entropy_production: Tensor | None = None


class Relaxation(NamedTuple):
    """Diagnostics along the relaxation at frozen drive, leading axis k."""

    magnetizations: Tensor
    mismatch: Tensor
    entropy_production: Tensor
    residual: Tensor


class SpinModelTransformerModule(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int = 1,
        num_steps: int | None = 1,
        init: Literal["reset", "amortized", "carried"] = "amortized",
        beta: float = 1.0,
        causal: bool = False,
        max_iter: int = 40,
        tol: float = 1e-5,
        ffn: bool = True,
        pre_mix: bool = False,
        post_mix: bool = False,
        measure_entropy_production: bool = False,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")

        self.dim = dim
        self.dim_head = dim // num_heads
        self.num_heads = num_heads
        self.radius_head = vmf.radius(self.dim_head)

        self.num_steps = num_steps
        self.init = init
        self.beta = beta
        self.causal = causal
        self.max_iter = max_iter
        self.tol = tol
        self.measure_entropy_production = measure_entropy_production

        self.split_heads = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.merge_heads = Rearrange("b h n d -> b n (h d)")

        self.to_qk = nn.Linear(dim, 2 * dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Disabling drops the memory term from the drive entirely.  It cannot be
        # an nn.Identity: the drive is ``x + f_FFN(x)``, so an identity would
        # double the input rather than remove the term.
        self.ffn = (
            nn.Sequential(
                nn.Linear(dim, 4 * dim, bias=False),
                nn.GELU(),
                nn.Linear(4 * dim, dim, bias=False),
            )
            if ffn
            else None
        )
        self.pre_mix = nn.Linear(dim, dim, bias=False) if pre_mix else nn.Identity()
        self.post_mix = nn.Linear(dim, dim, bias=False) if post_mix else nn.Identity()

        self.register_buffer("causal_mask", None, persistent=False)

    #
    # Drive-dependent quantities.  These depend on X_t only, so they are shared
    # by the forward pass and the diagnostics and computed once.
    #

    def normalize(self, x: Tensor) -> Tensor:
        """Put every head's slice on its own sphere of radius R(dim_head).

        Normalizing per head rather than once over ``dim`` is what makes
        ``||x_head|| = R_head`` hold exactly, which is the scale the mean-field
        expressions assume.
        """
        x = self.split_heads(self.pre_mix(x))
        return self.merge_heads(self.radius_head * F.normalize(x, dim=-1))

    def drive_and_couplings(self, x: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor]:
        """The frozen-drive field ``x + f_FFN(x)`` and the coupling rule J(X_t)."""
        queries, keys = self.to_qk(x).chunk(2, dim=-1)
        drive = self.split_heads(x if self.ffn is None else x + self.ffn(x))
        queries, keys = map(lambda t: F.normalize(self.split_heads(t), dim=-1), (queries, keys))

        sim = self.radius_head * torch.einsum("bhid,bhjd->bhij", queries, keys)
        if mask is not None:
            sim = sim.masked_fill(~rearrange(mask, "b j -> b 1 1 j"), -torch.finfo(sim.dtype).max)
        if self.causal:
            sim = sim.masked_fill(
                rearrange(self.get_causal_mask(sim.shape[-1], sim.device), "i j -> 1 1 i j"),
                -torch.finfo(sim.dtype).max,
            )
        return drive, sim.softmax(dim=-1)

    def get_causal_mask(self, n: int, device) -> Tensor:
        if self.causal_mask is not None and self.causal_mask.shape[-1] >= n:
            return self.causal_mask[:n, :n]
        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("causal_mask", mask, persistent=False)
        return mask

    def initial(self, x: Tensor, state: MeanFieldState | None) -> Tensor:
        """M_{t,0}, the choice that picks a column of the table.

        - ``reset``     unmagnetized, M = 0.  The trivial start: carries nothing
          from the drive and nothing from history, so it is the control the
          other two are read against.
        - ``amortized`` M = X_t W_V, a learned guess at where relaxation ends up.
        - ``carried``   M = M_{t-1,K} from the previous drive step, falling back
          to the amortized guess when there is no history, so the reset and
          carried columns differ only from t=1 on.
        """
        if self.init == "reset":
            return self.split_heads(torch.zeros_like(x))
        if self.init == "carried" and state is not None:
            return state.magnetizations
        return self.split_heads(self.to_v(x))

    #
    # Relaxation
    #

    def settle(self, start: Tensor, drive: Tensor, couplings: Tensor) -> tuple[Tensor, Tensor]:
        """Relax to the horizon set by ``num_steps``, returning the last two iterates.

        The previous iterate is kept because the delayed correlations that feed
        the entropy production need two consecutive fields; at a fixed point the
        two coincide, which is exactly the steady-state expression.
        """
        step_fn = partial(mf.step_large_d, drive=drive, couplings=couplings, beta=self.beta)
        if self.num_steps is None:
            settled = mf.anderson(step_fn, start, max_iter=self.max_iter, tol=self.tol)
            return settled, settled
        trajectory = mf.relax_large_d(
            start, drive, couplings, self.beta, num_steps=self.num_steps
        )
        return trajectory[-1], trajectory[-2]

    def forward(
        self, x: Tensor, state: MeanFieldState | None = None, mask: Tensor | None = None
    ) -> Readout:
        x = self.normalize(x)
        drive, couplings = self.drive_and_couplings(x, mask)
        settled, previous = self.settle(self.initial(x, state), drive, couplings)

        entropy_production = None
        if self.measure_entropy_production:
            field = mf.effective_field(settled, drive, couplings)
            previous_field = mf.effective_field(previous, drive, couplings)
            entropy_production = proxies.housekeeping_entropy_production(
                couplings, mf.covariance_traces_large_d(field, previous_field, self.beta), self.beta
            )

        # The returned state is not detached: truncating the history is the
        # caller's decision, not something the module should make silently.
        return Readout(
            magnetizations=self.post_mix(self.merge_heads(settled)),
            state=MeanFieldState(magnetizations=settled),
            entropy_production=entropy_production,
        )

    @torch.no_grad()
    def relaxation(
        self,
        x: Tensor,
        state: MeanFieldState | None = None,
        mask: Tensor | None = None,
        num_steps: int = 64,
    ) -> Relaxation:
        """Trace the relaxation at frozen drive, for probing rather than for output.

        Always plain step-by-step iteration ``m_{k+1} = phi(x + J m_k)``, never
        the accelerated solver: here the path is the object of study, and
        Anderson's iterates are solver states that do not correspond to any
        physical k.  The fixed point the mismatch is measured against is solved
        separately, so this reports the approach to the steady state even when
        the module itself runs at finite K and never computes one.
        """
        x = self.normalize(x)
        drive, couplings = self.drive_and_couplings(x, mask)
        start = self.initial(x, state)

        trajectory = mf.relax_large_d(start, drive, couplings, self.beta, num_steps=num_steps)
        step_fn = partial(mf.step_large_d, drive=drive, couplings=couplings, beta=self.beta)
        steady = mf.anderson(step_fn, start, max_iter=self.max_iter, tol=self.tol)

        fields = mf.effective_field(trajectory, drive, couplings)
        steady_field = mf.effective_field(steady, drive, couplings)
        traces = mf.covariance_traces_large_d(fields[1:], fields[:-1], self.beta)

        return Relaxation(
            magnetizations=trajectory,
            mismatch=proxies.mismatch(fields, steady_field, self.beta),
            entropy_production=proxies.housekeeping_entropy_production(
                couplings, traces, self.beta
            ),
            residual=(trajectory[1:] - trajectory[:-1]).norm(dim=-1).amax(dim=-1),
        )
