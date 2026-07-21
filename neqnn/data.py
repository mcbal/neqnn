"""A small formal language with repeated motifs, and windows over it.

Structured enough that some drive steps are genuinely surprising and others are
not, which is the contrast the quench experiment groups by, and simple enough
that "surprising" is a label we construct rather than one we have to infer.

Token embeddings are frozen random unit vectors: at this stage the parameters
are fixed and we are probing the dynamics, not learning a representation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def motif_stream(
    num_blocks: int,
    *,
    motif_length: int = 4,
    num_symbols: int = 6,
    repeats: int = 3,
) -> tuple[Tensor, Tensor]:
    """Repeated-motif token stream, with a novelty flag per position.

    Each block draws a fresh motif and emits it ``repeats`` times, separated by a
    separator token.  The first emission is novel -- nothing earlier in the
    stream predicts it -- while later emissions are recoverable from that first
    occurrence, so a system with memory should find them cheap.  The separator is
    always predictable and never counts as novel.

    Returns ``(tokens, novel)``, both of length
    ``num_blocks * repeats * (motif_length + 1)``.  The vocabulary is
    ``num_symbols + 1``, with the separator last.
    """
    separator = num_symbols
    tokens: list[int] = []
    novel: list[bool] = []
    for _ in range(num_blocks):
        motif = torch.randint(num_symbols, (motif_length,)).tolist()
        for repeat in range(repeats):
            tokens.extend(motif)
            novel.extend([repeat == 0] * motif_length)
            tokens.append(separator)
            novel.append(False)
    return torch.tensor(tokens), torch.tensor(novel)


def token_embeddings(vocab_size: int, dim: int) -> Tensor:
    """Frozen random token embeddings, one unit vector per token.

    The module renormalizes each head's slice onto its own sphere anyway, so only
    the direction here matters.
    """
    return F.normalize(torch.randn(vocab_size, dim), dim=-1)


def windows(length: int, size: int, *, policy: str = "sliding"):
    """Yield ``(start, stop)`` index pairs, one per drive step t.

    Both policies begin on the same window and take the same number of steps, so
    they can be compared directly:

    - ``sliding``  fixed-size window advancing one token per step.  N is
      constant, which keeps sigma_hk and the mismatch commensurable across t.
    - ``growing``  nothing is ever evicted, the KV-cache shape of a conventional
      transformer.  N grows with t, and sigma_hk sums over N^2 pairs, so it
      drifts upwards for purely geometric reasons -- read it accordingly.
    """
    if size > length:
        raise ValueError(f"window {size} does not fit in a stream of {length}")
    if policy == "sliding":
        for start in range(length - size + 1):
            yield start, start + size
    elif policy == "growing":
        for stop in range(size, length + 1):
            yield 0, stop
    else:
        raise ValueError(f"unknown window policy {policy!r}")


def sites_dropped(policy: str) -> int:
    """How many sites leave the front of the window per step, for realigning state."""
    return 1 if policy == "sliding" else 0
