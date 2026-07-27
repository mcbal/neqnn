"""Shared fixtures-in-spirit for the per-module test files.

Everything in the suite is a cross-check between two independent routes to the
same number -- Bessel against sampling, large-D against exact, streaming against
stored, implicit gradient against unrolled -- rather than against stored
constants.  Monte Carlo tolerances are set from the standard error, not tuned.
"""

from __future__ import annotations

import torch

from neqnn import stochastic

DIMS = [16, 32, 64, 128, 256, 512]


def relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).norm() / expected.norm())


def slope_in_dim(errors: list[float]) -> float:
    """Fitted exponent of ``error ~ D^slope``; the large-D forms should give -1."""
    design = torch.stack(
        [torch.log(torch.tensor(DIMS, dtype=torch.float64)), torch.ones(len(DIMS))], -1
    )
    return float(
        torch.linalg.lstsq(design, torch.log(torch.tensor(errors))[:, None]).solution[0, 0]
    )


def random_problem(dim: int, sites: int = 8, seed: int = 1):
    torch.manual_seed(seed)
    drive = stochastic.random_state((sites,), dim)
    couplings = 0.9 * torch.softmax(torch.randn(sites, sites), -1)
    return drive, couplings
