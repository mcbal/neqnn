"""Small shared pieces for experiment I/O and figure styling."""

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
SURFACE, INK, MUTED = "#fbfaf7", "#171717", "#66635f"
BLUE, ORANGE, GREEN = "#2374c6", "#e66b35", "#15936f"


def numbers(value: str, cast=float) -> list:
    values = [cast(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected a comma-separated list")
    return values


def save_data(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"saved {path}")


def load_data(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#aaa6a0",
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.frameon": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig, name: str) -> None:
    import matplotlib.pyplot as plt

    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{suffix}")
    plt.close(fig)
    print(f"saved figures/{name}.png")
