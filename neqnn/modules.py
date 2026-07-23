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


def advance(
    state: MeanFieldState, *, drop: int = 1, add: int = 1, fill: Tensor | None = None
) -> MeanFieldState:
    """Realign carried magnetizations after the window moves along the stream.

    ``drop`` sites leave the front and ``add`` arrive at the back, so a sliding
    window is ``drop=add=1`` and a growing one is ``drop=0``.  Without this the
    carried state would be misaligned by one site per step and would describe the
    wrong tokens entirely.

    Arriving sites are unmagnetized by default, which is the per-site version of
    the reset initialization: a token that just entered the window genuinely has
    no relaxation history.  Pass ``fill`` to seed them with an amortized guess
    instead.
    """
    magnetizations = state.magnetizations[..., drop:, :]
    if add:
        tail = (
            fill
            if fill is not None
            else magnetizations.new_zeros(
                *magnetizations.shape[:-2], add, magnetizations.shape[-1]
            )
        )
        magnetizations = torch.cat([magnetizations, tail], dim=-2)
    return MeanFieldState(magnetizations=magnetizations)


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
        qk_norm: bool = True,
        qk_bias: bool = False,
        rope: bool = False,
        rope_base: float = 10_000.0,
        pre_mix: bool = False,
        post_mix: bool = False,
        measure_entropy_production: bool = False,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        if num_steps is not None and num_steps < 1:
            raise ValueError(f"num_steps must be a positive int or None, got {num_steps}")

        self.dim = dim
        self.dim_head = dim // num_heads
        self.num_heads = num_heads
        self.radius_head = vmf.radius(self.dim_head)

        self.num_steps = num_steps
        self.init = init
        self.beta = beta
        self.causal = causal
        self.qk_norm = qk_norm
        self.rope = rope
        self.rope_base = rope_base
        self.max_iter = max_iter
        self.tol = tol
        self.measure_entropy_production = measure_entropy_production

        self.split_heads = Rearrange("b n (h d) -> b h n d", h=num_heads)
        self.merge_heads = Rearrange("b h n d -> b n (h d)")

        # ``qk_bias`` adds a content-independent component to queries and keys.
        # With zero-mean embeddings and no bias, rope has no constant vector to
        # rotate, so *positional-only* attention lobes are unrepresentable and
        # single-layer induction measurably fails to form; the bias restores
        # that pathway (experiments README, single-pass induction result).
        self.to_qk = nn.Linear(dim, 2 * dim, bias=qk_bias)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Ordinary RMS norm for the learned branches, with the physics living
        # entirely in the init: a uniform gain of R / sqrt(dim_head) gives each
        # normalized head norm exactly R.  As in PaLM's parallel block, the
        # direct input drive bypasses this norm while Q/K/V and the FFN consume
        # the normalized stream.  RMS rather than layer norm because subtracting
        # the mean would remove the component along the all-ones vector, an
        # arbitrary coordinate axis with no meaning for spins on a sphere.
        self.drive_norm = nn.RMSNorm(self.dim_head, elementwise_affine=True)
        nn.init.constant_(self.drive_norm.weight, self.radius_head / self.dim_head**0.5)

        # Attention sharpness.  A fixed 1/sqrt(d) works when queries and keys are
        # unnormalized, because training sharpens attention by growing their
        # norms.  `qk_norm` pins those norms to 1 and removes that lever
        # entirely, so the scale has to come back as an explicit parameter --
        # the same pairing QK-norm architectures use.
        self.attn_temperature = nn.Parameter(torch.tensor(float(self.dim_head) ** 0.5))

        # Disabling drops the memory term from the drive entirely.  It cannot be
        # an nn.Identity: the drive is ``x + f_FFN(norm(x))``, so an identity
        # would add the normalized stream rather than remove the term.
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
        self.register_buffer("rope_cache", None, persistent=False)

    #
    # Drive-dependent quantities.  These depend on X_t only, so they are shared
    # by the forward pass and the diagnostics and computed once.
    #

    def normalize(self, x: Tensor) -> Tensor:
        """Normalize the input stream for Q/K/V and the FFN.

        R is the radius the *microscopic spins* live on, and magnetizations are
        bounded by it by construction.  The drive is under no such obligation.
        Spins are pure direction; fields are not, and their magnitude is
        physical -- ``kappa = beta R ||h||`` is how hard a site is pinned -- so
        forcing every drive onto one sphere throws that away.

        Plain layer norm also puts every vector on a sphere; the per-token norm
        variation in a transformer comes entirely from the learnable gain, which
        makes the radius depend on direction.  The gain is uniform at init, so
        norms all start at R, and training is free to spread them.
        """
        return self.merge_heads(self.drive_norm(self.split_heads(self.pre_mix(x))))

    def rotary(self, n: int, device, dtype) -> tuple[Tensor, Tensor]:
        """Rotary angles for relative positions, cached on the buffer.

        Rotating queries and keys by their absolute position makes the logit
        depend on ``j - i`` alone, which is exactly the invariance a sliding
        window needs: shift every site by one and the couplings between
        surviving pairs are unchanged.  Absolute site embeddings would break
        that and make carried state describe the wrong tokens.
        """
        if self.rope_cache is not None and self.rope_cache.shape[-2] >= n:
            cached = self.rope_cache[..., :n, :]
            return cached[0], cached[1]
        power = torch.arange(0, self.dim_head, 2, device=device, dtype=dtype) / self.dim_head
        angles = torch.outer(torch.arange(n, device=device, dtype=dtype), self.rope_base**-power)
        self.register_buffer("rope_cache", torch.stack([angles.cos(), angles.sin()]), persistent=False)
        return angles.cos(), angles.sin()

    def apply_rotary(self, t: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        even, odd = t[..., 0::2], t[..., 1::2]
        return torch.stack([even * cos - odd * sin, even * sin + odd * cos], dim=-1).flatten(-2)

    def drive_and_couplings(
        self, x: Tensor, mask: Tensor | None, *, normalized: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """The field ``x + f_FFN(norm(x))`` and coupling rule ``J(norm(x))``.

        With ``qk_norm`` the logit is ``R_head cos(q, k)``, bounded by the head
        radius whatever the weights do -- stable, but it also means query and key
        *magnitude* carries no information and the usual lever on attention
        sharpness is disconnected.  Without it the logit is the plain scaled dot
        product, so norms are free to grow and sharpen the couplings.

        ``normalized`` lets forward paths reuse their normalized input for the
        amortized V initialization.  Diagnostic callers can omit it.
        """
        normalized = self.normalize(x) if normalized is None else normalized
        queries, keys = self.to_qk(normalized).chunk(2, dim=-1)
        drive = self.split_heads(x if self.ffn is None else x + self.ffn(normalized))
        queries, keys = map(self.split_heads, (queries, keys))

        if self.qk_norm:
            queries, keys = map(lambda t: F.normalize(t, dim=-1), (queries, keys))
            scale = self.attn_temperature
        else:
            scale = self.attn_temperature / self.dim_head
        if self.rope:
            cos, sin = self.rotary(x.shape[-2], x.device, x.dtype)
            queries, keys = (self.apply_rotary(t, cos, sin) for t in (queries, keys))
        sim = scale * torch.einsum("bhid,bhjd->bhij", queries, keys)
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

        The fixed-point branch solves without grad and then re-attaches an
        exact implicit gradient.  Backpropagating through the solver would store
        every iterate and differentiate the path rather than the solution, and
        Anderson's ring buffer is written in place besides.
        """
        step_fn = partial(mf.step_large_d, drive=drive, couplings=couplings, beta=self.beta)
        if self.num_steps is None:
            with torch.no_grad():
                solved = mf.anderson(step_fn, start, max_iter=self.max_iter, tol=self.tol)
            settled = mf.implicit_grad(step_fn, solved, max_iter=self.max_iter)
            return settled, settled
        trajectory = mf.relax_large_d(
            start, drive, couplings, self.beta, num_steps=self.num_steps
        )
        return trajectory[-1], trajectory[-2]

    def forward(
        self, x: Tensor, state: MeanFieldState | None = None, mask: Tensor | None = None
    ) -> Readout:
        normalized = self.normalize(x)
        drive, couplings = self.drive_and_couplings(x, mask, normalized=normalized)
        settled, previous = self.settle(self.initial(normalized, state), drive, couplings)

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
        normalized = self.normalize(x)
        drive, couplings = self.drive_and_couplings(x, mask, normalized=normalized)
        start = self.initial(normalized, state)

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
