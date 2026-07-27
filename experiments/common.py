"""Shared scaffolding for the experiment scripts: style, caching, console output.

Deliberately thin.  Anything that is physics belongs in ``neqnn``; anything
specific to one question belongs in that experiment's script.  What is left is
the stuff that would otherwise be copy-pasted four times.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

#
# Palette.  Three categorical slots for "which route to the number" and a
# four-step single-hue ramp for beta, which is ordered and so must not be
# categorical.  Both sets are validated for colour-vision deficiency against the
# light chart surface; the aqua slot sits below 3:1 contrast, which is why every
# series is also directly labelled rather than identified by colour alone.
#
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
INK_FAINT = "#a8a7a1"

SAMPLED = "#2a78d6"  # blue
EXACT = "#eb6834"  # orange
LARGE_D = "#1baf7a"  # aqua

SERIES = {"sampled": SAMPLED, "exact": EXACT, "large_d": LARGE_D}

# Ordered families (D, drive strength) get a single-hue ramp rather than
# categorical slots, because those variables are magnitudes and colour should
# carry that.  Five steps is the ceiling: the blue ramp cannot clear the 0.06
# adjacent-lightness gap over six, so a sixth level would read as a duplicate.
DIM_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
BETA_RAMP = DIM_RAMP


def use_style() -> None:
    """Blog-facing matplotlib defaults: recessive axes, thin marks, no chartjunk."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SOFT,
            "axes.edgecolor": INK_FAINT,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": INK_FAINT,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.6,
            "lines.markersize": 4.5,
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )


def save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path)
    fig.savefig(path.with_suffix(".pdf"))
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


#: When set, ``cached`` refuses to compute anything and reads only what is
#: already on disk.  Tuning a figure should not cost a sweep: do the run once,
#: then iterate on the drawing with this on.
PLOT_ONLY = False


def cached(key: str, compute: Callable[[], dict], *, refresh: bool = False) -> dict:
    """Run ``compute`` once and keep the result on disk under ``key``.

    Caching per cell rather than per sweep is what makes a long run resumable:
    an interrupted sweep loses at most the cell it was in, and re-plotting costs
    nothing.  ``refresh`` forces recomputation of a cell that is already there.

    Whatever varies the *result* has to be in ``key``, not just the identity of
    the cell -- a sampling budget that is not in the key means a smoke run and a
    full run collide on disk and the next full run silently reuses smoke numbers.
    """
    path = DATA / f"{key}.pt"
    if path.exists() and not refresh:
        return torch.load(path, weights_only=False)
    if PLOT_ONLY:
        raise FileNotFoundError(
            f"--plot-only, but {path.relative_to(ROOT)} is not cached; "
            "run without it first"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    result = compute()
    torch.save(result, path)
    return result


class Console:
    """Aligned live table, so a long sweep reads like an instrument rather than a log."""

    def __init__(self, columns: dict[str, int]):
        self.columns = columns
        self.start = time.time()

    def rule(self, title: str = "") -> None:
        width = sum(self.columns.values()) + 3 * (len(self.columns) - 1)
        print(f"\n{title}" if title else "")
        print("-" * width)

    def header(self) -> None:
        print("   ".join(f"{name:>{w}}" for name, w in self.columns.items()))
        print("-" * (sum(self.columns.values()) + 3 * (len(self.columns) - 1)))

    def row(self, *values) -> None:
        cells = []
        for value, width in zip(values, self.columns.values()):
            text = value if isinstance(value, str) else f"{value:.3g}"
            cells.append(f"{text:>{width}}")
        print("   ".join(cells), flush=True)

    def elapsed(self) -> str:
        seconds = time.time() - self.start
        return f"{int(seconds // 60)}m{int(seconds % 60):02d}s"


def relative(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative error in Frobenius norm, the comparison used throughout."""
    return float((actual - expected).norm() / expected.norm())


__all__ = [
    "BETA_RAMP",
    "DIM_RAMP",
    "Console",
    "DATA",
    "EXACT",
    "FIGURES",
    "INK",
    "INK_FAINT",
    "INK_SOFT",
    "LARGE_D",
    "SAMPLED",
    "SERIES",
    "SURFACE",
    "cached",
    "relative",
    "save",
    "use_style",
]
