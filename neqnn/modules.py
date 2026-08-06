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

The input coordinate is explicit too.  ``input_mode="field"`` preserves the
original interface, where the incoming tensor contributes directly to the
external drive.  ``input_mode="magnetization"`` treats each input head as an
interior mean parameter and first lifts it to its conjugate field through the
inverse large-D response.  With no additional field and a reset initializer, a
one-step module in the latter mode is exactly the identity, including its input
Jacobian.

A note on ``post_mix``.  Mixing head outputs is what a transformer does, but
here the outputs *are* magnetizations, and a linear map of them is no longer the
magnetization of anything -- it has left the mean-field state space.  So it is
applied to the readout only, never to the state that gets carried or fed to the
diagnostics.  Turning it on makes the module more transformer-like and its
output less physically interpretable, and that trade is the whole reason it is
a flag rather than a default.
"""

from __future__ import annotations

import math
import warnings
from functools import partial
from typing import Literal, NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from neqnn import fixed_point as fp, mean_field as mf, proxies, vmf


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
    if not isinstance(drop, int) or isinstance(drop, bool) or drop < 0:
        raise ValueError(f"drop must be a non-negative integer, got {drop!r}")
    if not isinstance(add, int) or isinstance(add, bool) or add < 0:
        raise ValueError(f"add must be a non-negative integer, got {add!r}")
    if state.magnetizations.ndim < 2:
        raise ValueError(
            "state magnetizations must have shape (..., sites, dim), got "
            f"{tuple(state.magnetizations.shape)}"
        )
    sites = state.magnetizations.shape[-2]
    if drop > sites:
        raise ValueError(f"cannot drop {drop} sites from a state with {sites} sites")
    magnetizations = state.magnetizations[..., drop:, :]
    if add:
        expected = (*magnetizations.shape[:-2], add, magnetizations.shape[-1])
        if fill is not None and tuple(fill.shape) != expected:
            raise ValueError(
                f"fill must have shape {expected} for add={add}, got {tuple(fill.shape)}"
            )
        if fill is not None and (
            fill.dtype != magnetizations.dtype or fill.device != magnetizations.device
        ):
            raise ValueError("fill must share the state's dtype and device")
        tail = (
            fill
            if fill is not None
            else magnetizations.new_zeros(
                *magnetizations.shape[:-2], add, magnetizations.shape[-1]
            )
        )
        magnetizations = torch.cat([magnetizations, tail], dim=-2)
    elif fill is not None:
        raise ValueError("fill was provided but add=0")
    return MeanFieldState(magnetizations=magnetizations)


class Probe(NamedTuple):
    """Raw ingredients of one forward pass, for control-room instrumentation.

    Returned by ``forward(..., probe=True)`` so that diagnostics decompose the
    *same* pass the module ran, rather than re-implementing the forward outside
    it and silently drifting when it changes.  Everything is per head, shaped
    like the carried state; what to measure on these tensors is the
    instrument's business, not the module's.
    """

    x: Tensor  # the input stream, split into heads
    drive: Tensor  # carrier field + f_FFN(norm(x))
    couplings: Tensor  # J(norm(x)), (b, heads, n, n)
    initial: Tensor  # M_{t,0}, the initialization the relaxation started from


class Readout(NamedTuple):
    """Physical state, optional learned readout, and fixed-point evidence.

    ``magnetizations`` always remains inside the physical mean-field state
    space.  ``output`` is the possibly post-mixed tensor intended for downstream
    neural-network use; it is identical to ``magnetizations`` when
    ``post_mix=False``.
    """

    magnetizations: Tensor
    state: MeanFieldState
    entropy_production: Tensor | None = None
    probe: Probe | None = None
    output: Tensor | None = None
    fixed_point: fp.Solve | None = None


class Relaxation(NamedTuple):
    """Diagnostics along the relaxation at frozen drive.

    ``magnetizations[k]`` is the state m_k, including the initializer at k=0.
    ``mismatch[k]`` is evaluated at that state through its update field h(m_k);
    the vMF law induced by this field has mean m_{k+1}. ``entropy_production[k]``
    and ``residual[k]`` belong to the transition m_k -> m_{k+1}, so they have
    one fewer entry than the state and mismatch trajectories.
    """

    magnetizations: Tensor
    mismatch: Tensor
    entropy_production: Tensor
    residual: Tensor
    fixed_point: fp.Solve


class SpinModelTransformerModule(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int = 1,
        num_steps: int | None = 1,
        init: Literal["reset", "amortized", "carried"] = "amortized",
        input_mode: Literal["field", "magnetization"] = "field",
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
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
        if (
            not isinstance(num_heads, int)
            or isinstance(num_heads, bool)
            or num_heads <= 0
        ):
            raise ValueError(f"num_heads must be a positive integer, got {num_heads!r}")
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        dim_head = dim // num_heads
        if dim_head <= 2:
            raise ValueError(
                "the spin-radius convention requires dim / num_heads > 2, "
                f"got head dimension {dim_head}"
            )
        if init not in {"reset", "amortized", "carried"}:
            raise ValueError(
                f"init must be one of 'reset', 'amortized', or 'carried', got {init!r}"
            )
        if input_mode not in {"field", "magnetization"}:
            raise ValueError(
                "input_mode must be either 'field' or 'magnetization', "
                f"got {input_mode!r}"
            )
        if num_steps is not None and (
            not isinstance(num_steps, int)
            or isinstance(num_steps, bool)
            or num_steps < 1
        ):
            raise ValueError(
                f"num_steps must be a positive int or None, got {num_steps}"
            )
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError(f"beta must be finite and positive, got {beta}")
        if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter < 2:
            raise ValueError(f"max_iter must be an integer >= 2, got {max_iter!r}")
        if not math.isfinite(tol) or tol <= 0:
            raise ValueError(f"tol must be finite and positive, got {tol}")
        if not math.isfinite(rope_base) or rope_base <= 0:
            raise ValueError(f"rope_base must be finite and positive, got {rope_base}")
        if rope and dim_head % 2:
            raise ValueError(f"rope requires an even head dimension, got {dim_head}")
        if input_mode == "magnetization" and pre_mix:
            raise ValueError(
                "pre_mix is not supported for magnetization inputs because its "
                "unconstrained linear map does not preserve the physical ball"
            )

        self.dim = dim
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.radius_head = vmf.radius(self.dim_head)

        self.num_steps = num_steps
        self.init = init
        self.input_mode = input_mode
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
        self.to_qk = nn.Linear(dim, 2 * dim, bias=qk_bias)
        self.to_v = nn.Linear(dim, dim, bias=False)

        # Ordinary RMS norm for the learned branches, with the physics living
        # entirely in the init: a uniform gain of R / sqrt(dim_head) gives each
        # normalized head norm exactly R. RMS rather than layer norm because
        # subtracting the mean would remove the component along the all-ones vector,
        # an arbitrary coordinate axis with no meaning for spins on a sphere.
        self.drive_norm = nn.RMSNorm(self.dim_head, elementwise_affine=True)
        nn.init.constant_(self.drive_norm.weight, self.radius_head / self.dim_head**0.5)

        # Attention sharpness. With normalized Q/K the learned scalar is the
        # entire logit scale. Keep the established sqrt(D_head) operating point;
        # unlike the physical response scale this is a neural-network parameter,
        # not something bounded by R. Its magnitude is used in the forward pass
        # so optimization cannot silently turn similarity attention into
        # anti-similarity attention.
        self.attn_temperature = nn.Parameter(torch.tensor(float(self.dim_head) ** 0.5))

        # Disabling drops the memory term from the drive entirely.
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

        self.register_buffer("_causal_mask", None, persistent=False)
        self.register_buffer("_rope_cache", None, persistent=False)

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
        """
        return self.merge_heads(self.drive_norm(self.split_heads(x)))

    def rotary(self, n: int, device, dtype) -> tuple[Tensor, Tensor]:
        """Rotary angles for relative positions, cached on the buffer.

        Rotating queries and keys by their absolute position makes the logit
        depend on ``j - i`` alone, which is exactly the invariance a sliding
        window needs: shift every site by one and the couplings between
        surviving pairs are unchanged.  Absolute site embeddings would break
        that and make carried state describe the wrong tokens.
        """
        cache = self._rope_cache
        if (
            cache is not None
            and cache.shape[-2] >= n
            and cache.dtype == dtype
            and cache.device == device
        ):
            cached = cache[..., :n, :]
            return cached[0], cached[1]
        power = (
            torch.arange(0, self.dim_head, 2, device=device, dtype=dtype)
            / self.dim_head
        )
        angles = torch.outer(
            torch.arange(n, device=device, dtype=dtype), self.rope_base**-power
        )
        self.register_buffer(
            "_rope_cache", torch.stack([angles.cos(), angles.sin()]), persistent=False
        )
        return angles.cos(), angles.sin()

    def apply_rotary(self, t: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        even, odd = t[..., 0::2], t[..., 1::2]
        return torch.stack(
            [even * cos - odd * sin, even * sin + odd * cos], dim=-1
        ).flatten(-2)

    def drive_and_couplings(
        self, x: Tensor, mask: Tensor | None, *, normalized: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """The carrier plus learned field and coupling rule ``J(norm(x))``.

        Field inputs contribute ``x`` directly. Magnetization inputs contribute
        ``phi_beta_inverse(x)`` instead, cancelling the response's shrinking
        Jacobian on the otherwise idle one-step path. In both cases the FFN is
        an additive physical field increment.

        With ``qk_norm`` the logit is ``temperature * cos(q, k)``. The positive
        learned temperature starts at ``sqrt(D_head)``; query and key magnitude
        carries no information, so sharpness lives entirely in that scalar.
        Without normalization it scales the plain dot product by
        ``temperature / D``.

        ``normalized`` lets forward paths reuse their normalized input for the
        amortized V initialization.  Diagnostic callers can omit it.
        """
        normalized = self.normalize(x) if normalized is None else normalized
        queries, keys = self.to_qk(normalized).chunk(2, dim=-1)
        carrier = self.split_heads(x)
        if self.input_mode == "magnetization":
            carrier = vmf.inverse_response_large_d(carrier, self.beta)
        drive = (
            carrier
            if self.ffn is None
            else carrier + self.split_heads(self.ffn(normalized))
        )
        queries, keys = map(self.split_heads, (queries, keys))

        if self.qk_norm:
            queries, keys = map(lambda t: F.normalize(t, dim=-1), (queries, keys))
            scale = self.attn_temperature.abs()
        else:
            scale = self.attn_temperature.abs() / self.dim_head
        if self.rope:
            cos, sin = self.rotary(x.shape[-2], x.device, x.dtype)
            queries, keys = (self.apply_rotary(t, cos, sin) for t in (queries, keys))
        sim = scale * torch.einsum("bhid,bhjd->bhij", queries, keys)
        available = torch.ones(
            x.shape[0],
            1,
            x.shape[-2],
            x.shape[-2],
            dtype=torch.bool,
            device=x.device,
        )
        if mask is not None:
            available = available & rearrange(mask, "b j -> b 1 1 j")
        if self.causal:
            available = available & ~rearrange(
                self.causal_mask(sim.shape[-1], sim.device), "i j -> 1 1 i j"
            )
        if not bool(available.any(-1).all()):
            raise ValueError(
                "mask leaves at least one query with no available key; "
                "softmax over a fully masked row is undefined"
            )
        sim = sim.masked_fill(~available, -torch.finfo(sim.dtype).max)
        return drive, sim.softmax(dim=-1)

    def causal_mask(self, n: int, device) -> Tensor:
        if self._causal_mask is not None and self._causal_mask.shape[-1] >= n:
            return self._causal_mask[:n, :n]
        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("_causal_mask", mask, persistent=False)
        return mask

    def initial(self, x: Tensor, state: MeanFieldState | None) -> Tensor:
        """M_{t,0}, the choice that picks a column of the table.

        - ``reset``     unmagnetized, M = 0.  The trivial start: carries nothing
          from the drive and nothing from history, so it is the control the
          other two are read against.
        - ``amortized`` H_0 = X_t W_V and M = phi_beta(H_0), a learned
          initializer field mapped through the same physical response used by
          every relaxation step.
        - ``carried``   M = M_{t-1,K} from the previous drive step, falling back
          to the amortized guess when there is no history, so the reset and
          carried columns differ only from t=1 on.
        """
        if self.init == "reset":
            return self.split_heads(torch.zeros_like(x))
        if self.init == "carried" and state is not None:
            return state.magnetizations
        initializer_field = self.split_heads(self.to_v(x))
        return vmf.response_large_d(initializer_field, self.beta)

    #
    # Relaxation
    #

    def _settle_with_evidence(
        self, start: Tensor, drive: Tensor, couplings: Tensor
    ) -> tuple[Tensor, Tensor, fp.Solve | None]:
        """Relax to the horizon set by ``num_steps``, returning the last two iterates.

        The previous iterate is kept because the delayed correlations that feed
        the entropy production need two consecutive fields; at a fixed point the
        two coincide, which is exactly the steady-state expression.

        The fixed-point branch solves without grad and then re-attaches an
        exact implicit gradient. Backpropagating through the solver would store
        every iterate and differentiate the path rather than the solution, and
        Anderson's ring buffer is written in place besides.
        """
        step_fn = partial(
            mf.step_large_d, drive=drive, couplings=couplings, beta=self.beta
        )
        if self.num_steps is None:
            with torch.no_grad():
                solve = fp.anderson(
                    step_fn, start, max_iter=self.max_iter, tol=self.tol
                )
            if not solve.converged:
                warnings.warn(
                    f"fixed-point forward residual {solve.residual:.2e} exceeds "
                    f"tol {self.tol:.2e}; output and implicit gradient may be "
                    "untrustworthy",
                    RuntimeWarning,
                    stacklevel=3,
                )
            settled = fp.implicit_grad(step_fn, solve.solution, max_iter=self.max_iter)
            return settled, settled, solve
        previous, current = start, start
        for _ in range(self.num_steps):
            previous, current = current, step_fn(current)
        return current, previous, None

    def settle(
        self, start: Tensor, drive: Tensor, couplings: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Settle while preserving the historical two-tensor public interface."""
        settled, previous, _ = self._settle_with_evidence(start, drive, couplings)
        return settled, previous

    def _validate_inputs(
        self, x: Tensor, state: MeanFieldState | None, mask: Tensor | None
    ) -> None:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape (batch, sites, dim), got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError(f"x must be floating point, got {x.dtype}")
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"x has feature dimension {x.shape[-1]}, expected {self.dim}"
            )
        if x.shape[-2] < 1:
            raise ValueError("x must contain at least one site")
        if mask is not None:
            expected = x.shape[:2]
            if tuple(mask.shape) != expected:
                raise ValueError(
                    f"mask must have shape {tuple(expected)}, got {tuple(mask.shape)}"
                )
            if mask.dtype != torch.bool:
                raise TypeError(f"mask must be boolean, got {mask.dtype}")
            if mask.device != x.device:
                raise ValueError("mask and x must be on the same device")
        if state is not None:
            expected = (
                x.shape[0],
                self.num_heads,
                x.shape[1],
                self.dim_head,
            )
            if tuple(state.magnetizations.shape) != expected:
                raise ValueError(
                    "state magnetizations must have shape "
                    f"{expected}, got {tuple(state.magnetizations.shape)}"
                )
            if (
                state.magnetizations.dtype != x.dtype
                or state.magnetizations.device != x.device
            ):
                raise ValueError("state magnetizations must share x's dtype and device")

    def forward(
        self,
        x: Tensor,
        state: MeanFieldState | None = None,
        mask: Tensor | None = None,
        *,
        probe: bool = False,
        drive_offset: Tensor | None = None,
    ) -> Readout:
        self._validate_inputs(x, state, mask)
        if drive_offset is not None:
            if tuple(drive_offset.shape) != tuple(x.shape):
                raise ValueError(
                    "drive_offset must have the same shape as x, got "
                    f"{tuple(drive_offset.shape)} instead of {tuple(x.shape)}"
                )
            if not drive_offset.is_floating_point():
                raise TypeError(
                    f"drive_offset must be floating point, got {drive_offset.dtype}"
                )
            if drive_offset.dtype != x.dtype or drive_offset.device != x.device:
                raise ValueError("drive_offset must share x's dtype and device")
        x = self.pre_mix(x)
        normalized = self.normalize(x)
        drive, couplings = self.drive_and_couplings(x, mask, normalized=normalized)
        if drive_offset is not None:
            # Unlike ``x``, this is an external physical field: it changes the
            # relaxation drive without changing Q/K/V features or the initial
            # state.  This is useful for causal feedback or clamping phases.
            drive = drive + self.split_heads(drive_offset)
        initial = self.initial(normalized, state)
        settled, previous, solve = self._settle_with_evidence(initial, drive, couplings)

        entropy_production = None
        if self.measure_entropy_production:
            field = mf.effective_field(settled, drive, couplings)
            previous_field = mf.effective_field(previous, drive, couplings)
            entropy_production = proxies.housekeeping_entropy_production(
                couplings,
                mf.covariance_traces_large_d(field, previous_field, self.beta),
                self.beta,
            )

        # The returned state is not detached: truncating the history is the
        # caller's decision, not something the module should make silently.
        magnetizations = self.merge_heads(settled)
        return Readout(
            magnetizations=magnetizations,
            state=MeanFieldState(magnetizations=settled),
            entropy_production=entropy_production,
            probe=Probe(
                x=self.split_heads(x), drive=drive, couplings=couplings, initial=initial
            )
            if probe
            else None,
            output=self.post_mix(magnetizations),
            fixed_point=solve,
        )

    @torch.no_grad()
    def relaxation(
        self,
        x: Tensor,
        state: MeanFieldState | None = None,
        mask: Tensor | None = None,
        num_steps: int = 64,
        start: Tensor | None = None,
        reference: Tensor | None = None,
    ) -> Relaxation:
        """Trace the relaxation at frozen drive, for probing rather than for output.

        Always plain step-by-step iteration ``m_{k+1} = phi(x + J m_k)``, never
        the accelerated solver: here the path is the object of study, and
        Anderson's iterates are solver states that do not correspond to any
        physical k. The fixed point the mismatch is measured against is solved
        separately, so this reports the approach to the steady state even when
        the module itself runs at finite K and never computes one.

        ``start`` overrides the module's initializer for counterfactual paths.
        ``reference`` reuses a fixed-point tensor, letting several paths be
        compared against exactly the same target under the current dynamics.
        """
        self._validate_inputs(x, state, mask)
        expected = (x.shape[0], self.num_heads, x.shape[1], self.dim_head)
        for name, value in (("start", start), ("reference", reference)):
            if value is not None and tuple(value.shape) != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(value.shape)}"
                )
            if value is not None and (
                value.dtype != x.dtype or value.device != x.device
            ):
                raise ValueError(f"{name} must share x's dtype and device")
        x = self.pre_mix(x)
        normalized = self.normalize(x)
        drive, couplings = self.drive_and_couplings(x, mask, normalized=normalized)
        start = self.initial(normalized, state) if start is None else start

        trajectory = mf.relax_large_d(
            start, drive, couplings, self.beta, num_steps=num_steps
        )
        step_fn = partial(
            mf.step_large_d, drive=drive, couplings=couplings, beta=self.beta
        )
        if reference is None:
            solve = fp.anderson(step_fn, start, max_iter=self.max_iter, tol=self.tol)
        else:
            residual = fp.residual(step_fn(reference), reference)
            solve = fp.Solve(reference, residual, residual < self.tol)
        if not solve.converged:
            warnings.warn(
                f"diagnostic fixed-point residual {solve.residual:.2e} exceeds "
                f"tol {self.tol:.2e}; mismatch is measured against an "
                "unconverged reference",
                RuntimeWarning,
                stacklevel=2,
            )
        steady = solve.solution

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
            fixed_point=solve,
        )
