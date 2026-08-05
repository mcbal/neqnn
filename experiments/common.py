"""Small shared pieces for experiment I/O and figure styling."""

from __future__ import annotations

from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "outputs"
FIGURES = ROOT / "figures"
SURFACE, INK, MUTED = "#faf9f6", "#29313d", "#707783"
BLUE, ORANGE, GREEN = "#6f9fc5", "#d89170", "#78a58f"
PURPLE, GRID = "#9a8fbd", "#d9d7d1"


def numbers(value: str, cast=float) -> list:
    values = [cast(item) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("expected a comma-separated list")
    return values


def save_data(payload: dict, path: Path) -> None:
    """Atomically replace an experiment artifact after it is fully serialized."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
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
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Inter",
                "Clear Sans",
                "Avenir Next",
                "Helvetica Neue",
                "DejaVu Sans",
            ],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": 500,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": GRID,
            "axes.labelcolor": MUTED,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.grid.which": "major",
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.45,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "legend.frameon": False,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig, name: str, directory: Path | None = None) -> None:
    import matplotlib.pyplot as plt

    directory = FIGURES if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}")
    plt.close(fig)
    print(f"saved {directory / f'{name}.png'}")
