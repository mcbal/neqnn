"""Render smooth phase diagrams from fidelity_compute.py output."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common

LABELS = {
    "magnetization": "magnetization",
    "delayed": "delayed correlation",
    "entropy": "entropy production",
}


def cmap(color: str):
    return LinearSegmentedColormap.from_list("", [common.SURFACE, color])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=common.DATA / "fidelity.pt")
    args = parser.parse_args()
    data = common.load_data(args.input)
    common.style()

    dims, us, betas = data["dims"], data["u"], data["beta"]
    fine_us, fine_betas = data["fine_u"], data["fine_beta"]
    levels = np.geomspace(1e-3, 2, 28)
    norm = LogNorm(levels[0], levels[-1])

    for name, label in LABELS.items():
        fig, axes = plt.subplots(
            2,
            len(dims),
            figsize=(3.45 * len(dims), 6.4),
            sharex="row",
            sharey="row",
            layout="constrained",
            squeeze=False,
        )
        for column, dim in enumerate(dims):
            measured = data["sampled_error"][name][column].numpy()
            floors = data["sampling_floor"][name][column].numpy()
            unresolved = ~np.isfinite(measured) | (measured <= floors)
            axes[0, column].set_facecolor("#dfddd8")
            top = axes[0, column].contourf(
                us,
                betas,
                np.ma.masked_where(unresolved, measured),
                levels=levels,
                norm=norm,
                cmap=cmap(common.BLUE),
                extend="both",
            )
            bottom = axes[1, column].contourf(
                fine_us,
                fine_betas,
                data["large_d_error"][name][column],
                levels=levels,
                norm=norm,
                cmap=cmap(common.ORANGE),
                extend="both",
            )
            grid_u, grid_beta = np.meshgrid(us, betas)
            axes[0, column].scatter(
                grid_u, grid_beta, s=5, color=common.INK, alpha=0.35, linewidth=0
            )
            multi = data["branches"][column].numpy() > 1
            failed = (data["branches"][column].numpy() == 0) | (
                data["residual"][column].numpy() > 1e-7
            )
            axes[0, column].scatter(
                grid_u[multi], grid_beta[multi], marker="x", color="white", s=35
            )
            axes[0, column].scatter(
                grid_u[failed], grid_beta[failed], marker="+", color="white", s=40
            )
            for row in range(2):
                ax = axes[row, column]
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xticks(us, [f"{value:g}" for value in us])
                ax.set_yticks(betas, [f"{value:g}" for value in betas])
                ax.minorticks_off()
                critical = 2 * dim / (dim - 2)
                if min(betas) <= critical <= max(betas):
                    ax.axhline(
                        critical,
                        color=common.INK,
                        ls=":",
                        lw=0.8,
                        alpha=0.6,
                    )
            axes[0, column].set_title(rf"$D={dim}$")
            axes[1, column].set_xlabel(r"pinning  $u=\beta\|x\|/R$")

        axes[0, 0].set_ylabel(r"coupling  $\beta$")
        axes[1, 0].set_ylabel(r"coupling  $\beta$")
        ticks = [1e-3, 1e-2, 1e-1, 1]
        fig.colorbar(
            top, ax=axes[0], shrink=0.86, ticks=ticks, label="mean field vs samples"
        )
        fig.colorbar(
            bottom, ax=axes[1], shrink=0.86, ticks=ticks, label="large D vs exact"
        )
        fig.legend(
            handles=[
                Patch(color="#dfddd8", label="below sampling resolution"),
                Line2D([], [], marker="x", color=common.MUTED, ls="", label="multistable"),
                Line2D([], [], marker="+", color=common.MUTED, ls="", label="solve failed"),
                Line2D([], [], color=common.INK, ls=":", lw=0.8, label=r"$\beta_c(D)$"),
            ],
            loc="outside lower center",
            ncol=4,
            fontsize=8,
        )
        fig.suptitle(f"Approximation fidelity — {label}", color=common.INK, fontsize=12)
        common.save_figure(fig, f"fidelity_{name}")


if __name__ == "__main__":
    main()
