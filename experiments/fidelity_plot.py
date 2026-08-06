"""Render fidelity phase diagrams from a saved compute artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import common
except ModuleNotFoundError:  # Allow importing helpers from the repository root.
    from experiments import common

LABELS = {
    "magnetization": "magnetization",
    "delayed": "delayed correlation",
    "entropy": "entropy production",
}
MISSING = "#deddd8"


def as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=float)


def sequential(name: str, color: str):
    palette = LinearSegmentedColormap.from_list(
        name, [common.SURFACE, color, common.INK]
    )
    return palette.with_extremes(under=common.SURFACE, bad=MISSING)


def crosses(value: np.ndarray, level: float) -> bool:
    finite = value[np.isfinite(value)]
    return bool(finite.size and finite.min() < level < finite.max())


def configure_axis(ax, us, betas, *, xlabel: bool) -> None:
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(us, [f"{value:g}" for value in us])
    ax.set_yticks(betas, [f"{value:g}" for value in betas])
    ax.minorticks_off()
    ax.grid(False)
    if xlabel:
        ax.set_xlabel(r"drive strength  $u=\beta\|x_i\|/R$")


def overlay_conditioning(ax, data: dict, column: int, dim: int) -> set[str]:
    shown = set()
    fine_contraction = data.get("fine_contraction")
    if fine_contraction is not None:
        contraction = as_numpy(fine_contraction[column])
        residual_data = data.get("fine_residual_exact")
        residual = (
            as_numpy(residual_data[column])
            if residual_data is not None
            else np.zeros_like(contraction)
        )
        residual_tol = float(data.get("config", {}).get("residual_tol", 1e-7))
        contraction = np.where(residual <= residual_tol, contraction, np.nan)
        if crosses(contraction, 1.0):
            ax.contour(
                data["fine_u"],
                data["fine_beta"],
                contraction,
                levels=[1.0],
                colors=[common.INK],
                linewidths=1.25,
            )
            shown.add("local")
    critical = 2 * dim / (dim - 2)
    if min(data["fine_beta"]) <= critical <= max(data["fine_beta"]):
        ax.axhline(critical, color=common.INK, ls=":", lw=0.9, alpha=0.65)
        shown.add("critical")
    return shown


def sampled_masks(data: dict, column: int) -> dict[str, np.ndarray]:
    residual = as_numpy(data["residual"][column])
    branches = as_numpy(data["branches"][column])
    residual_tol = float(data.get("config", {}).get("residual_tol", 1e-7))
    failed = (branches == 0) | ~np.isfinite(residual) | (residual > residual_tol)

    contraction = data.get("contraction")
    unstable = (
        as_numpy(contraction[column]) >= 1.0
        if contraction is not None
        else np.zeros_like(failed)
    )
    ergodicity = data.get("ergodicity")
    saturation = data.get("chain_saturation")
    broken = (
        (as_numpy(ergodicity[column]) > 1.4) & (as_numpy(saturation[column]) > 0.3)
        if ergodicity is not None and saturation is not None
        else np.zeros_like(failed)
    )
    multi = branches > 1
    return {
        "failed": failed,
        "unstable": unstable,
        "broken": broken,
        "multi": multi,
        "caveat": failed | unstable | broken | multi,
    }


def mark_sampled_cells(ax, data: dict, column: int, masks: dict) -> None:
    grid_u, grid_beta = np.meshgrid(data["u"], data["beta"])
    ax.scatter(
        grid_u,
        grid_beta,
        s=7,
        color=common.INK,
        alpha=0.22,
        linewidth=0,
        zorder=4,
    )
    for mask, marker, size in (
        (masks["multi"], "x", 34),
        (masks["failed"], "+", 42),
        (masks["broken"], "s", 32),
    ):
        if np.any(mask):
            style = (
                {"facecolors": "none", "edgecolors": common.INK}
                if marker == "s"
                else {"color": common.INK}
            )
            ax.scatter(
                grid_u[mask],
                grid_beta[mask],
                marker=marker,
                s=size,
                linewidth=1.05,
                zorder=5,
                **style,
            )


def plot_mean_field(
    data: dict,
    *,
    prefix: str,
    output_dir: Path | None,
) -> None:
    dims, us, betas = data["dims"], data["u"], data["beta"]
    fig, axes = plt.subplots(
        len(LABELS),
        len(dims),
        figsize=(3.55 * len(dims), 2.75 * len(LABELS)),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    norm = LogNorm(1e-3, 1.0)
    palette = sequential("soft-blue", common.BLUE)
    image = None
    shown = set()

    for row, (name, label) in enumerate(LABELS.items()):
        for column, dim in enumerate(dims):
            measured = as_numpy(data["sampled_error"][name][column])
            floor = as_numpy(data["sampling_floor"][name][column])
            quality = np.maximum(measured, floor)
            masks = sampled_masks(data, column)
            invalid = masks["caveat"] | ~np.isfinite(quality)
            plotted = np.ma.masked_where(invalid, quality)

            ax = axes[row, column]
            ax.set_facecolor(MISSING)
            image = ax.pcolormesh(
                us,
                betas,
                plotted,
                shading="nearest",
                cmap=palette,
                norm=norm,
            )
            mark_sampled_cells(ax, data, column, masks)
            shown.update(overlay_conditioning(ax, data, column, dim))
            for key in ("multi", "failed", "broken"):
                if np.any(masks[key]):
                    shown.add(key)
            if np.any(invalid):
                shown.add("missing")
            configure_axis(ax, us, betas, xlabel=row == len(LABELS) - 1)
            if row == 0:
                ax.set_title(rf"$D={dim}$")
            if column == 0:
                ax.set_ylabel(f"{label}\n" + r"inverse temperature  $\beta$")

    fig.colorbar(
        image,
        ax=axes,
        shrink=0.88,
        extend="both",
        label="conservative relative discrepancy: max(error estimate, Monte Carlo resolution)",
    )
    candidates = [
        (
            "local",
            Line2D(
                [],
                [],
                color=common.INK,
                lw=1.25,
                label=r"local contraction boundary  $\rho=1$",
            ),
        ),
        (
            "critical",
            Line2D(
                [],
                [],
                color=common.INK,
                ls=":",
                lw=0.9,
                label=r"zero-drive contraction threshold  $\beta_c(D)$",
            ),
        ),
        (
            "multi",
            Line2D([], [], marker="x", color=common.INK, ls="", label="multistable"),
        ),
        (
            "failed",
            Line2D([], [], marker="+", color=common.INK, ls="", label="solve failed"),
        ),
        (
            "broken",
            Line2D(
                [],
                [],
                marker="s",
                markerfacecolor="none",
                color=common.INK,
                ls="",
                label="ergodicity broken",
            ),
        ),
        ("missing", Patch(color=MISSING, label="comparison ill-posed")),
    ]
    handles = [handle for key, handle in candidates if key in shown]
    if handles:
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=min(3, len(handles)),
            fontsize=7.5,
            frameon=True,
            facecolor="#e9e9e6",
            edgecolor=common.GRID,
            framealpha=0.96,
        )
    common.save_figure(fig, f"{prefix}_mean_field", output_dir)


def plot_large_d(data: dict, *, prefix: str, output_dir: Path | None) -> None:
    dims = data["dims"]
    fine_us, fine_betas = data["fine_u"], data["fine_beta"]
    fig, axes = plt.subplots(
        len(LABELS),
        len(dims),
        figsize=(3.55 * len(dims), 2.75 * len(LABELS)),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    norm = LogNorm(1e-4, 2.0)
    palette = sequential("soft-coral", common.ORANGE)
    image = None

    for row, (name, label) in enumerate(LABELS.items()):
        for column, dim in enumerate(dims):
            values = as_numpy(data["large_d_error"][name][column])
            plotted = np.ma.masked_invalid(values)
            ax = axes[row, column]
            ax.set_facecolor(MISSING)
            image = ax.pcolormesh(
                fine_us,
                fine_betas,
                plotted,
                shading="nearest",
                cmap=palette,
                norm=norm,
            )
            overlay_conditioning(ax, data, column, dim)
            masks = sampled_masks(data, column)
            mark_sampled_cells(ax, data, column, masks)
            configure_axis(ax, data["u"], data["beta"], xlabel=row == len(LABELS) - 1)
            summary = values
            if data.get("fine_contraction") is not None:
                stable = as_numpy(data["fine_contraction"][column]) < 1.0
                summary = np.where(stable, summary, np.nan)
            finite = summary[np.isfinite(summary)]
            if finite.size:
                median = np.median(finite)
                ax.text(
                    0.97,
                    0.04,
                    f"stable median {median:.1%}",
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=7.5,
                    color=common.INK,
                    bbox={
                        "facecolor": common.SURFACE,
                        "edgecolor": "none",
                        "alpha": 0.78,
                        "pad": 2,
                    },
                )
            if row == 0:
                ax.set_title(rf"$D={dim}$")
            if column == 0:
                ax.set_ylabel(f"{label}\n" + r"inverse temperature  $\beta$")

    fig.colorbar(
        image,
        ax=axes,
        shrink=0.88,
        extend="both",
        label=r"relative gap  $\|\mathrm{large}\ D-\mathrm{exact}\|/\|\mathrm{exact}\|$",
    )
    common.save_figure(fig, f"{prefix}_large_d", output_dir)


def plot_projection(data: dict, *, prefix: str, output_dir: Path | None) -> None:
    """Explain which delayed-correlation errors entropy production can see."""
    if "delayed_error_fraction" not in data:
        print("artifact has no projection decomposition; skipping projection plot")
        return

    nearest = lambda values, target: min(
        range(len(values)), key=lambda index: abs(values[index] - target)
    )
    ui, bi = nearest(data["u"], 1.0), nearest(data["beta"], 1.0)
    dims = np.asarray(data["dims"])
    delayed = as_numpy(data["sampled_error"]["delayed"])[:, bi, ui]
    delayed_floor = as_numpy(data["sampling_floor"]["delayed"])[:, bi, ui]
    projected = as_numpy(data["projected_error"])[:, bi, ui]
    projected_floor = as_numpy(data["sampling_floor"]["entropy"])[:, bi, ui]
    resolved = as_numpy(data["projection_resolved"])[:, bi, ui].astype(bool)

    fig, (error_ax, fraction_ax) = plt.subplots(
        1, 2, figsize=(9.2, 3.7), layout="constrained"
    )
    error_ax.plot(
        dims,
        np.maximum(delayed, delayed_floor),
        color=common.ORANGE,
        marker="o",
        label=r"full $C^{del}$ error",
    )
    error_ax.plot(
        dims,
        np.maximum(projected, projected_floor),
        color=common.BLUE,
        marker="o",
        label=r"projection onto $J-J^\mathsf{T}$",
    )
    error_ax.set(
        xlabel="spin dimension $D$",
        ylabel="conservative relative discrepancy",
        yscale="log",
        title="Full error versus visible projection",
    )
    error_ax.set_xticks(dims)
    error_ax.legend()

    bottoms = np.zeros(len(dims))
    fractions = data["delayed_error_fraction"]
    for key, color, label in (
        ("symmetric", common.ORANGE, "symmetric · invisible"),
        (
            "antisymmetric_orthogonal",
            common.GREEN,
            "antisymmetric but orthogonal",
        ),
        ("parallel", common.BLUE, r"parallel to $J-J^\mathsf{T}$"),
    ):
        values = as_numpy(fractions[key])[:, bi, ui]
        values = np.where(resolved, values, 0.0)
        fraction_ax.bar(dims, values, bottom=bottoms, color=color, label=label)
        bottoms += values
    unresolved = ~resolved
    if np.any(unresolved):
        fraction_ax.scatter(
            dims[unresolved],
            np.full(unresolved.sum(), 0.5),
            marker="x",
            color=common.INK,
            zorder=4,
            label="MC floor-limited",
        )
    fraction_ax.set(
        xlabel="spin dimension $D$",
        ylabel="fraction of squared systematic error",
        ylim=(0, 1),
        title="Orthogonal error decomposition",
    )
    fraction_ax.set_xticks(dims)
    fraction_ax.legend(fontsize=7.5)
    fig.suptitle(
        rf"Delayed-correlation projection at $u={data['u'][ui]:g}$, "
        rf"$\beta={data['beta'][bi]:g}$",
        fontsize=12,
    )
    common.save_figure(fig, f"{prefix}_projection", output_dir)


def markdown_summary(data: dict, *, u: float = 1.0, beta: float = 1.0) -> str:
    """Return compact, blog-ready tables at the nearest sampled operating point."""

    def nearest(values, target):
        return min(range(len(values)), key=lambda index: abs(values[index] - target))

    def percent(value: float) -> str:
        if not np.isfinite(value):
            return "—"
        percentage = 100 * value
        return f"{percentage:.2f}%" if percentage < 1 else f"{percentage:.1f}%"

    ui, bi = nearest(data["u"], u), nearest(data["beta"], beta)
    fui = nearest(data["fine_u"], data["u"][ui])
    fbi = nearest(data["fine_beta"], data["beta"][bi])
    sampled_u, sampled_beta = data["u"][ui], data["beta"][bi]
    lines = [rf"At $u={sampled_u:g}$ and $\beta={sampled_beta:g}$.", ""]

    for name, label in LABELS.items():
        lines.extend(
            [
                f"### {label.capitalize()}",
                "",
                "| D | MF vs stochastic | MC floor | large-D vs exact MF |",
                "|---:|---:|---:|---:|",
            ]
        )
        for column, dim in enumerate(data["dims"]):
            measured = float(as_numpy(data["sampled_error"][name])[column, bi, ui])
            floor = float(as_numpy(data["sampling_floor"][name])[column, bi, ui])
            large_d = float(as_numpy(data["large_d_error"][name])[column, fbi, fui])
            estimate = "floor-limited" if measured <= floor else percent(measured)
            lines.append(
                f"| {dim} | {estimate} | {percent(floor)} | {percent(large_d)} |"
            )
        lines.append("")

    lines.append(
        "“Floor-limited” means the noise-corrected MF error did not exceed the Monte Carlo floor."
    )
    if "delayed_error_fraction" in data:
        lines.extend(
            [
                "",
                "### Delayed-error projection",
                "",
                (
                    "| D | symmetric and invisible | antisymmetric, orthogonal | "
                    r"parallel to $J-J^\mathsf{T}$ |"
                ),
                "|---:|---:|---:|---:|",
            ]
        )
        resolved = as_numpy(data["projection_resolved"])
        fractions = data["delayed_error_fraction"]
        for column, dim in enumerate(data["dims"]):
            if not bool(resolved[column, bi, ui]):
                lines.append(
                    f"| {dim} | floor-limited | floor-limited | floor-limited |"
                )
                continue
            values = [
                float(as_numpy(fractions[name])[column, bi, ui])
                for name in ("symmetric", "antisymmetric_orthogonal", "parallel")
            ]
            lines.append(
                f"| {dim} | " + " | ".join(percent(value) for value in values) + " |"
            )
    return "\n".join(lines) + "\n"


def plot(
    data: dict,
    *,
    prefix: str = "fidelity",
    output_dir: Path | None = None,
) -> None:
    common.style()
    plot_mean_field(
        data,
        prefix=prefix,
        output_dir=output_dir,
    )
    plot_large_d(data, prefix=prefix, output_dir=output_dir)
    plot_projection(data, prefix=prefix, output_dir=output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=common.DATA / "fidelity.pt")
    parser.add_argument("--output-dir", type=Path, default=common.FIGURES)
    parser.add_argument("--prefix", default="fidelity")
    args = parser.parse_args()
    data = common.load_data(args.input)
    plot(
        data,
        prefix=args.prefix,
        output_dir=args.output_dir,
    )
    summary = args.output_dir / f"{args.prefix}_summary.md"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(markdown_summary(data))
    print(f"saved {summary}")


if __name__ == "__main__":
    main()
