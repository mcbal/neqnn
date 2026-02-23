from functools import partial
from math import sqrt
from typing import NamedTuple

import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torchdeq import get_deq

from neqnn.systems import (
    entropy_production,
    m_plefka_t_1_t_naive_mf,
    td_corr_plefka_t_1_t_naive_mf,
    theta,
)


#
# spin-transformer module
#


def scale_from_dim(dim: int) -> float:
    return sqrt(dim / 2 - 1)


class MeanFieldState(NamedTuple):
    theta: torch.Tensor | None


class SpinTransformerModule(nn.Module):
    """
    Example of a transformer module wrapping around the mean-field dynamics of a
    vector-spin system. Implements parallel attention with focus on simplicity.
    Multiple heads are implemented by splitting the vector dim, leading to multiple
    parallel spin systems (head dim becomes another batch dim, not just for attention).

    TODO (mbal):
    - Add support for some kind of positional embedding (e.g., PoPE)
    """

    def __init__(
        self,
        *,
        dim,  # vector dimension
        num_heads,  # number of (dim // num_heads)-dimensional subspaces to split across
        pre_mix: bool = False,
        post_mix: bool = False,
        causal: bool = False,  # whether to impose causal mask on attention matrix
        num_steps: int | None = 1,  # time steps to take (None = time-evolution fp)
        fp_solver_max_iter: int | None = None,  # max_iter of fixed-point solver
        fp_solver_tol: float | None = None,  # tolerance of fixed-point solver
        beta: float = 1.0,  # inverse temperature of vector-spin system
        return_sigma: bool = False,  # add entropy production to output tuple
    ):
        super().__init__()

        self.dim = dim
        self.dim_head = dim // num_heads
        self.dim_inner = self.dim_head * num_heads
        self.num_heads = num_heads

        self.scale = scale_from_dim(self.dim)
        self.scale_head = scale_from_dim(self.dim_head)

        self.causal = causal
        self.register_buffer("causal_mask", None, persistent=False)

        self.beta = beta
        self.return_sigma = return_sigma

        # head management
        self.split_heads = Rearrange("b n (h d) -> b h n d", h=self.num_heads)
        self.merge_heads = Rearrange("b h n d -> b n (h d)")

        # attention params
        self.to_qk = nn.Linear(dim, 2 * self.dim_inner, bias=False)

        # memory params
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, 4 * self.dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * self.dim, self.dim, bias=False),
        )

        # input/output mixing
        self.pre_mix = (
            nn.Linear(self.dim, self.dim, bias=False) if pre_mix else nn.Identity()
        )
        self.post_mix = (
            nn.Linear(self.dim_inner, self.dim, bias=False)
            if post_mix
            else nn.Identity()
        )

        # time-delayed correlations
        self.td_corr = partial(
            td_corr_plefka_t_1_t_naive_mf, beta=self.beta, scale=self.scale
        )

        # magnetizations
        self.magnetizations = partial(
            m_plefka_t_1_t_naive_mf, beta=self.beta, scale=self.scale
        )

        # step (either take `num_steps` or find fixed point)
        if num_steps is not None:

            def _inner_loop(_m0, _x, _J):
                for i in range(num_steps):
                    state = theta(_x, _J, _m0)
                    m = self.magnetizations(state)
                    if i < num_steps - 1:
                        _m0 = m  # TODO: can also detach() here (inner iteration)
                return m, state

            self.step = _inner_loop
        else:

            def _fixed_point(_m0, _x, _J, *, _solver):
                z_out, _ = _solver(
                    lambda _m: self.magnetizations(theta(_x, _J, _m)),
                    self.magnetizations(theta(_x, _J, _m0)),
                )
                m = z_out[-1]
                return m, theta(_x, _J, m)

            self.step = partial(
                _fixed_point,
                _solver=get_deq(
                    f_solver="anderson",
                    f_max_iter=(
                        fp_solver_max_iter if fp_solver_max_iter is not None else 40
                    ),
                    f_tol=fp_solver_tol if fp_solver_tol is not None else 1e-4,
                ),
            )

        # init mean-field state
        self.prev_mf_state = MeanFieldState(theta=None)

    def get_causal_mask(self, n, device):
        if self.causal_mask is not None and self.causal_mask.shape[-1] >= n:
            return self.causal_mask[:n, :n]

        causal_mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        return causal_mask

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        # maybe pre-mix inputs
        x = self.pre_mix(x)

        # normalize inputs to sphere
        x = self.scale * F.normalize(x, dim=-1)

        # queries and keys from inputs
        q, k = self.to_qk(x).chunk(2, dim=-1)

        # ff from inputs
        ff = self.ffn(x)

        # head dim becomes batch dim
        x, q, k, ff = map(self.split_heads, (x, q, k, ff))

        # normalize queries and keys to unit sphere
        q, k = map(lambda t: F.normalize(t, dim=-1), (q, k))

        # overlap
        sim = self.scale_head * torch.einsum("b h i d, b h j d -> b h i j", q, k)

        # key mask
        if mask is not None:
            mask = rearrange(mask, "b j -> b 1 1 j")  # true -> keep
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # causal mask
        if self.causal:
            causal_mask = rearrange(
                self.get_causal_mask(x.size(-2), x.device), "i j -> 1 1 i j"
            )  # true -> mask
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # softmax attention
        attn = sim.softmax(dim=-1)  # (b h n n)

        # print(attn)

        # first fwd pass: init previous mean-field state with all-ones
        if self.prev_mf_state.theta is None:
            self.prev_mf_state = self.prev_mf_state._replace(theta=torch.ones(x.size()))
            # TODO (mbal): add padding logic to support shrinking / growing of `seq_len`

        # augment residual with ffn memory
        # print(torch.linalg.vector_norm(x, dim=-1), torch.linalg.vector_norm(ff, dim=-1))
        x = x + ff  # (b h n d)

        # update mean-field magnetizations
        m0 = self.magnetizations(self.prev_mf_state.theta)
        m, theta = self.step(m0, x, attn)  # (b h n d), (b h n d)

        # store updated mean-field state
        self.prev_mf_state = self.prev_mf_state._replace(theta=theta.detach())

        # rearrange head outputs
        m = self.merge_heads(m)

        # maybe post-mix outputs (but then `m` no longer corresponds to `theta` in mf state!)
        m = self.post_mix(m)

        # return magnetizations (and entropy production)
        out = (m,)
        if self.return_sigma:
            out += (
                entropy_production(
                    self.beta, attn, self.td_corr(theta, attn, self.prev_mf_state.theta)
                ),
            )
        return out


#
# spin-transformer model (stack of `SpinTransformerModule` layers)
#


class SpinTransformerModel(nn.Module):
    def __init__(
        self,
        num_layers: int,
        dim: int,
        num_heads: int,
        pre_mix: bool = False,
        post_mix: bool = False,
        causal: bool = False,
        num_steps: int | None = 1,
        fp_solver_max_iter: int | None = None,
        fp_solver_tol: float | None = None,
        beta: float = 1.0,
        return_sigmas: bool = False,
        should_detach: bool = False,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                SpinTransformerModule(
                    dim=dim,
                    num_heads=num_heads,
                    pre_mix=pre_mix,
                    post_mix=post_mix,
                    causal=causal,
                    beta=beta,
                    num_steps=num_steps,
                    fp_solver_max_iter=fp_solver_max_iter,
                    fp_solver_tol=fp_solver_tol,
                    return_sigma=return_sigmas,
                )
            )

        self.return_sigmas = return_sigmas
        self.should_detach = should_detach

    def forward(self, x):
        if self.return_sigmas:
            sigmas = []

        m = x
        for layer in self.layers:
            out = layer(m)
            if self.return_sigmas:
                m, sigma = out
                sigmas.append(sigma)
            else:
                m = out
            if self.should_detach:
                m = m.detach()

        if self.return_sigmas:
            return m, sigmas
        else:
            return m


if __name__ == "__main__":
    torch.manual_seed(1234)

    bsz, seq_len, dim = 1, 2048, 512

    model = SpinTransformerModel(
        num_layers=4,
        dim=dim,
        num_heads=4,
        pre_mix=True,
        num_steps=1,
        causal=False,
        return_sigmas=True,
        should_detach=True,
    )

    x = torch.randn((bsz, seq_len, dim))
    y, sigmas = model(x)

    (-sum(map(lambda x: x.mean(), sigmas))).backward()

    print(sigmas)

    for n, m in model.named_parameters():
        print((n, m.grad))
