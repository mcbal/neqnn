"""Plot training diagnostics and samples from language_model_compute.py output."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation
from PIL import Image

try:
    import common
except ModuleNotFoundError:  # Allow importing helpers from the repository root.
    from experiments import common


def smooth(values, width=15):
    width = max(1, min(width, len(values)))
    return np.convolve(values, np.ones(width) / width, mode="valid"), width


def color_ramp(count: int, start: str, end: str) -> list:
    if count == 1:
        return [end]
    ramp = LinearSegmentedColormap.from_list("experiment-ramp", [start, end])
    return [ramp(value) for value in np.linspace(0.05, 0.95, count)]


def safe_ratio(values: np.ndarray, denominator: float) -> np.ndarray:
    return values / max(abs(float(denominator)), np.finfo(float).tiny)


def display_offset(ax, x_points: float = 0, y_points: float = 0):
    """Offset an artist in display space without changing its data coordinates."""
    return ax.transData + ScaledTranslation(
        x_points / 72, y_points / 72, ax.figure.dpi_scale_trans
    )


def path_arrow(
    ax,
    x,
    y,
    *,
    color,
    index: int,
    alpha: float = 1.0,
    linewidth: float = 1.0,
    transform=None,
) -> None:
    """Place one directional arrow on a polyline segment."""
    if len(x) < 2:
        return
    index = max(0, min(index, len(x) - 2))
    x_start = x[index] + 0.25 * (x[index + 1] - x[index])
    y_start = y[index] + 0.25 * (y[index + 1] - y[index])
    x_end = x[index] + 0.75 * (x[index + 1] - x[index])
    y_end = y[index] + 0.75 * (y[index + 1] - y[index])
    ax.annotate(
        "",
        xy=(x_end, y_end),
        xytext=(x_start, y_start),
        xycoords=transform or ax.transData,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "alpha": alpha,
            "lw": linewidth,
            "mutation_scale": 14,
            "shrinkA": 0,
            "shrinkB": 0,
        },
        zorder=5,
    )


def longest_visible_segment(x, y) -> int:
    """Choose a path segment after normalizing the two plotted coordinates."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_scale = max(float(np.ptp(x)), np.finfo(float).tiny)
    y_scale = max(float(np.ptp(y)), np.finfo(float).tiny)
    lengths = np.hypot(np.diff(x) / x_scale, np.diff(y) / y_scale)
    return int(np.argmax(lengths)) if len(lengths) else 0


def format_parameters(count: int) -> str:
    """Compact an exact parameter count in millions without trailing zeroes."""
    millions = f"{count / 1e6:.3f}".rstrip("0").rstrip(".")
    return f"{millions}M"


def plot_training(data: dict, prefix: str, output_dir: Path | None) -> None:
    runs = data["runs"]
    colors = color_ramp(len(runs), "#b8cde0", common.INK)
    fig, (loss_ax, grad_ax) = plt.subplots(
        1, 2, figsize=(9.5, 3.8), layout="constrained"
    )
    clip_grad = float(data.get("config", {}).get("clip_grad", 10.0))

    for index, (run, color) in enumerate(zip(runs, colors)):
        history = run["history"]
        rolling, width = smooth(
            history["train"], max(1, min(15, len(history["train"]) // 10))
        )
        loss_ax.plot(
            history["step"][width - 1 :],
            rolling,
            color=color,
            label=(
                f"{run['depth']} layers · {format_parameters(run['parameters'])} params"
            ),
        )
        loss_ax.scatter(
            history["eval_step"], history["valid"], color=color, s=18, zorder=3
        )
        probes = history["probe"]
        grad_ax.plot(
            [probe["step"] for probe in probes],
            [probe["grad_norm"] for probe in probes],
            color=color,
            marker="o",
            markersize=3,
            label="pre-clip" if index == 0 else None,
        )
        grad_ax.plot(
            [probe["step"] for probe in probes],
            [
                probe.get("grad_norm_post", min(probe["grad_norm"], clip_grad))
                for probe in probes
            ],
            color=color,
            ls="--",
            alpha=0.55,
            label="post-clip" if index == 0 else None,
        )
    last_step = max(run["history"]["step"][-1] for run in runs)
    for order, value in data.get("baselines", {}).items():
        loss_ax.axhline(value, color=common.MUTED, lw=0.8, ls=":", alpha=0.65)
        loss_ax.annotate(
            f"{order}-gram",
            (last_step, value),
            xytext=(-3, 2),
            textcoords="offset points",
            ha="right",
            color=common.MUTED,
            fontsize=7,
        )
    loss_ax.set(xlabel="optimizer step", ylabel="cross entropy (nats)")
    grad_ax.set(xlabel="optimizer step", ylabel="global gradient norm", yscale="log")
    grad_ax.axhline(clip_grad, color=common.MUTED, ls=":", lw=1, label="clip")
    loss_ax.set_title("character-level language modeling")
    grad_ax.set_title("optimization stability")
    loss_ax.legend()
    grad_ax.legend()
    common.save_figure(fig, f"{prefix}_training", output_dir)


def plot_timeline(data: dict, prefix: str, output_dir: Path | None) -> None:
    """Track the frozen-drive steady-state proxy on the training clock."""
    runs = data["runs"]
    has_fixed_point_entropy = all(
        "entropy_fixed_point_converged" in row
        for run in runs
        for probe in run["history"]["probe"]
        for row in probe["forward"]
    )
    metrics = []
    if has_fixed_point_entropy:
        metrics.append(("entropy", "frozen-drive steady-state irreversibility proxy"))
    if not metrics:
        print("artifact has no fixed-point proxy diagnostics; skipping timeline plot")
        return
    rows = len(metrics)
    fig, axes = plt.subplots(
        rows,
        len(runs),
        figsize=(max(5.0, 3.55 * len(runs)), 1.6 + 2.8 * rows),
        sharex="col",
        sharey="row",
        layout="constrained",
        squeeze=False,
    )
    layer_ramp = LinearSegmentedColormap.from_list(
        "layers", ["#cbd9e5", common.BLUE, common.INK]
    )

    for column, run in enumerate(runs):
        probes = run["history"]["probe"]
        probe_steps = [probe["step"] for probe in probes]
        layer_colors = [
            layer_ramp(value)
            for value in np.linspace(0.05, 0.95, max(run["depth"], 2))[: run["depth"]]
        ]
        for layer, color in enumerate(layer_colors):
            for row_index, (key, _) in enumerate(metrics):
                axes[row_index, column].plot(
                    probe_steps,
                    [probe["forward"][layer].get(key, np.nan) for probe in probes],
                    color=color,
                    marker=".",
                    ms=3,
                )

        axes[0, column].set_title(
            f"{run['depth']} layers · {format_parameters(run['parameters'])} parameters"
        )
        for row_index in range(rows):
            axes[row_index, column].set_yscale("log")
        axes[-1, column].set_xlabel("optimizer step")

    for row, (_, label) in enumerate(metrics):
        axes[row, 0].set_ylabel(label)
    fig.legend(
        handles=[
            Line2D([], [], color="#cbd9e5", label="shallow layer"),
            Line2D([], [], color=common.INK, label="deep layer"),
        ],
        loc="outside lower center",
        ncol=2,
        fontsize=8,
    )
    fig.suptitle(
        "Frozen-drive steady-state irreversibility during training", fontsize=12
    )
    common.save_figure(fig, f"{prefix}_timeline", output_dir)


def plot_signal(data: dict, prefix: str, output_dir: Path | None) -> None:
    runs = data["runs"]
    fig, axes = plt.subplots(
        3,
        len(runs),
        figsize=(3.5 * len(runs), 7.2),
        sharex="col",
        sharey="row",
        layout="constrained",
        squeeze=False,
    )
    for column, run in enumerate(runs):
        probes = run["history"]["probe"]
        for probe, style, alpha, label in (
            (probes[0], "--", 0.5, "initial"),
            (probes[-1], "-", 1.0, "trained"),
        ):
            layers = np.arange(1, run["depth"] + 1)
            fields = np.array([row["field"] for row in probe["forward"]])
            gradients = np.array(
                [
                    row["activation"] if isinstance(row, dict) else row
                    for row in probe["backward"]
                ]
            )
            saturation = np.array([row["saturation"] for row in probe["forward"]])
            axes[0, column].plot(
                layers,
                safe_ratio(fields, fields[0]),
                style,
                color=common.BLUE,
                alpha=alpha,
                label=label,
            )
            axes[1, column].plot(
                layers,
                safe_ratio(gradients, gradients[-1]),
                style,
                color=common.ORANGE,
                alpha=alpha,
            )
            axes[2, column].plot(
                layers, saturation, style, color=common.GREEN, alpha=alpha
            )
        axes[0, column].set_title(f"{run['depth']} layers")
        axes[2, column].set_xlabel("layer")
        axes[2, column].set_xticks(
            np.unique(np.linspace(1, run["depth"], min(6, run["depth"]), dtype=int))
        )
    axes[0, 0].set_ylabel(r"forward  $\|h_l\|/\|h_1\|$")
    axes[1, 0].set_ylabel(r"backward  $g_l/g_L$")
    axes[2, 0].set_ylabel(r"saturation  $\|m_l\|/R$")
    for ax in axes[1]:
        ax.set_yscale("log")
    for ax in axes[2]:
        ax.set_ylim(0, 1)
    axes[0, 0].legend()
    fig.suptitle(
        "Signal propagation across depth before and after training", fontsize=12
    )
    common.save_figure(fig, f"{prefix}_signal", output_dir)


def plot_fields(data: dict, prefix: str, output_dir: Path | None) -> None:
    runs = data["runs"]
    fig, axes = plt.subplots(
        1,
        len(runs),
        figsize=(3.5 * len(runs), 3.2),
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    for ax, run in zip(axes[0], runs):
        final = run["history"]["probe"][-1]["forward"]
        layers = np.arange(1, run["depth"] + 1)
        for values, color, label in (
            (
                [row.get("carrier", row["residual"]) for row in final],
                common.BLUE,
                "residual",
            ),
            ([row["ffn"] for row in final], common.ORANGE, "FFN"),
            ([row["coupling"] for row in final], common.GREEN, "attention"),
        ):
            ax.plot(layers, values, color=color, label=label)
        ax.set_title(f"{run['depth']} layers")
        ax.set_xlabel("layer")
    axes[0, 0].set_ylabel("mean per-site norm")
    axes[0, 0].legend()
    fig.suptitle("Final field decomposition", fontsize=12)
    common.save_figure(fig, f"{prefix}_fields", output_dir)


def plot_homeostasis(data: dict, prefix: str, output_dir: Path | None) -> None:
    """Compact cross-depth view of state and update geometry."""
    runs = data["runs"]
    required = {
        "saturation_p50",
        "saturation_p95",
        "increment_u",
        "increment_radial_u",
        "increment_transverse_u",
        "susceptibility_tangential",
        "susceptibility_radial",
        "direction_change",
        "direction_change_p95",
    }
    final_rows = [run["history"]["probe"][-1]["forward"] for run in runs]
    if any(required - row.keys() for final in final_rows for row in final):
        print("artifact has no homeostasis monitors; skipping homeostasis plot")
        return

    colors = color_ramp(len(runs), "#b8cde0", common.INK)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5), layout="constrained")

    for run, final, color in zip(runs, final_rows, colors):
        relative_depth = np.arange(1, run["depth"] + 1) / run["depth"]
        label = f"{run['depth']} layers"
        marker = "o" if run["depth"] <= 6 else None
        common_line = dict(color=color, marker=marker, ms=2.5)

        axes[0, 0].plot(
            relative_depth,
            [row["saturation_p50"] for row in final],
            label=label,
            **common_line,
        )
        axes[0, 0].plot(
            relative_depth,
            [row["saturation_p95"] for row in final],
            ls="--",
            alpha=0.55,
            **common_line,
        )

        axes[0, 1].plot(
            relative_depth,
            [
                row["susceptibility_radial"]
                / max(row["susceptibility_tangential"], np.finfo(float).tiny)
                for row in final
            ],
            label=label,
            **common_line,
        )

        axes[1, 0].plot(
            relative_depth,
            np.degrees([row["direction_change"] for row in final]),
            label=label,
            **common_line,
        )
        axes[1, 0].plot(
            relative_depth,
            np.degrees([row["direction_change_p95"] for row in final]),
            ls="--",
            alpha=0.55,
            **common_line,
        )

        increment = np.asarray([row["increment_u"] for row in final])
        denominator = np.maximum(increment, np.finfo(float).tiny)
        axes[1, 1].plot(
            relative_depth,
            np.asarray([row["increment_transverse_u"] for row in final]) / denominator,
            label=label,
            **common_line,
        )
        axes[1, 1].plot(
            relative_depth,
            np.asarray([row["increment_radial_u"] for row in final]) / denominator,
            ls="--",
            alpha=0.65,
            **common_line,
        )

    axes[0, 0].set(
        title="state saturation",
        ylabel=r"$\|m\|/R$",
        ylim=(0, 1),
    )
    axes[0, 0].axhspan(0.8, 1.0, color=common.ORANGE, alpha=0.08)
    axes[0, 1].set(
        title="response anisotropy",
        ylabel=r"$\chi_{\mathrm{radial}}/\chi_{\mathrm{tangential}}$",
        ylim=(0, 1),
    )
    axes[1, 0].set(
        title="successive reorientation",
        ylabel="field-direction change (degrees)",
    )
    axes[1, 1].set(
        title="update geometry",
        ylabel=r"component / $\|\Delta h\|$",
        ylim=(-0.75, 1.05),
    )
    axes[1, 1].axhline(0, color=common.MUTED, lw=0.7)
    for ax in axes[1]:
        ax.set_xlabel(r"relative layer depth  $l/L$")
    axes[0, 0].legend(
        handles=[
            Line2D([], [], color=common.INK, label="median"),
            Line2D(
                [], [], color=common.INK, ls="--", alpha=0.55, label="95th percentile"
            ),
        ],
        loc="lower right",
        fontsize=7.5,
    )
    axes[0, 1].legend(
        handles=[
            Line2D([], [], color=color, label=f"{run['depth']} layers")
            for run, color in zip(runs, colors)
        ],
        fontsize=8,
    )
    axes[1, 0].legend(
        handles=[
            Line2D([], [], color=common.INK, label="mean"),
            Line2D(
                [], [], color=common.INK, ls="--", alpha=0.55, label="95th percentile"
            ),
        ],
        fontsize=7.5,
    )
    axes[1, 1].legend(
        handles=[
            Line2D([], [], color=common.INK, label="transverse"),
            Line2D(
                [], [], color=common.INK, ls="--", alpha=0.65, label="signed radial"
            ),
        ],
        fontsize=7.5,
    )
    fig.suptitle("State and update geometry across depth", fontsize=12)
    common.save_figure(fig, f"{prefix}_homeostasis", output_dir)


def plot_layer_ablation(data: dict, prefix: str, output_dir: Path | None) -> None:
    """Paired held-out effects of skipping one residual update."""
    runs = [
        run
        for run in data["runs"]
        if (run["history"].get("layer_ablation") or {}).get("layers")
    ]
    if not runs:
        print("artifact has no layer-skip diagnostics; skipping layer-ablation plot")
        return

    colors = color_ramp(len(runs), "#b8cde0", common.INK)
    fig, loss_ax = plt.subplots(figsize=(6.3, 3.8), layout="constrained")
    for run, color in zip(runs, colors):
        rows = run["history"]["layer_ablation"]["layers"]
        layers = np.asarray([row["layer"] for row in rows]) / run["depth"]
        label = f"{run['depth']} layers"
        loss_ax.plot(
            layers,
            [max(row["loss_delta"], np.finfo(float).tiny) for row in rows],
            color=color,
            marker="o",
            ms=3,
            label=label,
        )
    loss_ax.set(
        xlabel=r"relative skipped-layer position  $l/L$",
        ylabel=r"held-out loss change  $\mathcal{L}_{skip}-\mathcal{L}_{full}$",
        yscale="log",
        title="task effect of removing one update",
    )
    loss_ax.legend()
    fig.suptitle("Individual layers become less indispensable with depth", fontsize=12)
    common.save_figure(fig, f"{prefix}_layer_ablation", output_dir)


def plot_relaxation_interventions(
    data: dict, prefix: str, output_dir: Path | None
) -> None:
    """Relate task degradation directly to approach toward the fixed point."""
    runs = [
        run for run in data["runs"] if run["history"].get("relaxation_interventions")
    ]
    if not runs:
        print("artifact has no K interventions; skipping intervention plot")
        return
    if len(runs) > 1:
        print("plotting the first run with K interventions")
    run = runs[0]
    records = run["history"]["relaxation_interventions"]
    final = records[-1]
    final_rows = final["layers"]
    available = final_rows[0]["loss_delta"]
    styles = {
        "1": (common.INK, r"$K=1$", "-", "o"),
        "2": (common.ORANGE, r"$K=2$", "-", "o"),
        "4": (common.GREEN, r"$K=4$", "-", "o"),
        "inf": (common.BLUE, r"$K=\infty$", "--", "o"),
    }
    conditions = [key for key in ("1", "2", "4", "inf") if key in available]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), layout="constrained")

    def paired_sem(row, horizon):
        values = np.asarray(row["loss_delta_batches"][horizon], dtype=float)
        finite = values[np.isfinite(values)]
        return finite.std(ddof=1) / finite.size**0.5 if finite.size > 1 else 0.0

    final_layer_rows = [
        next(row for row in record["layers"] if row["layer"] == run["depth"])
        for record in records
    ]
    dither = {
        "1": (0, 0),
        "2": (0, 0),
        "4": (2.0, 1.5),
        "inf": (-2.0, -1.5),
    }
    for condition in conditions:
        color, label, linestyle, marker = styles[condition]
        x = [row["rms"][condition] for row in final_layer_rows]
        y = [row["loss_delta"][condition] for row in final_layer_rows]
        transform = display_offset(axes[0], *dither[condition])
        axes[0].errorbar(
            x,
            y,
            yerr=[paired_sem(row, condition) for row in final_layer_rows],
            color=color,
            ls=linestyle,
            marker=marker,
            ms=3,
            capsize=1.5,
            label=label,
            transform=transform,
        )
        axes[0].scatter(
            [x[0]],
            [y[0]],
            s=32,
            facecolors=common.SURFACE,
            edgecolors=color,
            zorder=4,
            transform=transform,
        )
        axes[0].scatter(
            [x[-1]], [y[-1]], s=34, color=color, zorder=4, transform=transform
        )
        path_arrow(
            axes[0],
            x,
            y,
            color=color,
            index=longest_visible_segment(x, y),
            linewidth=1.5,
            transform=transform,
        )

    left_x = [
        row["rms"][condition] for row in final_layer_rows for condition in conditions
    ]
    left_y_low = [
        row["loss_delta"][condition] - paired_sem(row, condition)
        for row in final_layer_rows
        for condition in conditions
    ]
    left_y_high = [
        row["loss_delta"][condition] + paired_sem(row, condition)
        for row in final_layer_rows
        for condition in conditions
    ]
    x_span = max(left_x) - min(left_x)
    y_span = max(left_y_high) - min(left_y_low)
    axes[0].set_xlim(-0.025, max(left_x) + 0.06 * max(x_span, 1e-3))
    axes[0].set_ylim(
        min(left_y_low) - 0.06 * max(y_span, 1e-3),
        max(left_y_high) + 0.40 * max(y_span, 1e-3),
    )

    for row in final_rows:
        x = [row["rms"][condition] for condition in conditions]
        y = [row["loss_delta"][condition] for condition in conditions]
        line_color = common.BLUE
        alpha = 0.48
        axes[1].plot(
            x,
            y,
            color=line_color,
            lw=1.0,
            alpha=alpha,
            zorder=2,
        )
        axes[1].errorbar(
            x,
            y,
            yerr=[paired_sem(row, condition) for condition in conditions],
            fmt="none",
            ecolor=line_color,
            alpha=alpha,
            capsize=1.5,
            zorder=1,
        )
        for x_value, y_value, condition in zip(x, y, conditions):
            color, _, _, _ = styles[condition]
            is_baseline = condition == "1"
            axes[1].scatter(
                [x_value],
                [y_value],
                s=34 if is_baseline else 22,
                facecolors=common.INK if is_baseline else color,
                edgecolors=common.INK if is_baseline else color,
                linewidths=1.2 if is_baseline else 0.9,
                alpha=1.0 if is_baseline else 0.62,
                zorder=3,
            )
        path_arrow(
            axes[1],
            x,
            y,
            color=line_color,
            index=0,
            alpha=0.8,
            linewidth=1.1,
        )
        label_offsets = {
            1: (5, 7),
            2: (5, 6),
            3: (5, -10),
            4: (5, -10),
            5: (5, 5),
            6: (5, -10),
        }
        axes[1].annotate(
            f"L{row['layer']}",
            (x[1], y[1]),
            xytext=label_offsets[row["layer"]],
            textcoords="offset points",
            color=common.MUTED,
            fontsize=7.2,
        )

    for ax in axes:
        ax.axhline(0, color=common.MUTED, lw=0.7, ls=":")
        ax.set_xlabel(r"state RMS to fixed point  $\|m-m^\star\|$")
        ax.set_xlim(left=-0.025)
    axes[0].set(
        title="Final layer across training",
        ylabel=r"held-out $\mathrm{CE}-\mathrm{CE}(K=1)$",
    )
    axes[0].legend(ncol=1, loc="upper left", fontsize=8)
    axes[0].text(
        0.98,
        0.98,
        "open: initialization · filled: final · arrows: training time\n"
        "small display offsets separate overlapping paths",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
        color=common.MUTED,
        fontsize=7.5,
    )
    axes[1].set(
        title="Each layer follows its own relaxation path",
        ylabel=r"held-out $\mathrm{CE}-\mathrm{CE}(K=1)$",
    )
    axes[1].text(
        0.98,
        0.47,
        "arrows: increasing K",
        transform=axes[1].transAxes,
        ha="right",
        color=common.MUTED,
        fontsize=7.5,
    )

    if final.get("joint") is not None:
        joint = final["joint"]
        inset = axes[1].inset_axes([0.61, 0.61, 0.35, 0.31])
        positions = np.arange(len(conditions))
        joint_y = [joint["loss_delta"][condition] for condition in conditions]
        inset.plot(positions, joint_y, color=common.MUTED, lw=0.8, zorder=1)
        inset.errorbar(
            positions,
            joint_y,
            yerr=[paired_sem(joint, condition) for condition in conditions],
            fmt="none",
            ecolor=common.MUTED,
            capsize=1.5,
            zorder=1,
        )
        for position, value, condition in zip(positions, joint_y, conditions):
            inset.scatter(position, value, color=styles[condition][0], s=22, zorder=2)
        inset.axhline(0, color=common.MUTED, lw=0.6, ls=":")
        inset.set_xticks(positions, ["1", "2", "4", r"$\infty$"])
        inset.tick_params(labelsize=6.5)
        inset.set_title("all layers jointly", fontsize=7.5)
        inset.set_ylabel(r"$\Delta$CE", fontsize=7)

    fig.suptitle(
        f"Frozen-protocol relaxation interventions · depth {run['depth']}",
        fontsize=12,
    )
    common.save_figure(fig, f"{prefix}_relaxation_intervention", output_dir)


def plot_initializer_causal_test(
    data: dict, prefix: str, output_dir: Path | None
) -> None:
    """Test which physical launch state makes one frozen update useful."""
    runs = [
        run for run in data["runs"] if run["history"].get("relaxation_interventions")
    ]
    if not runs:
        return
    run = runs[0]
    records = run["history"]["relaxation_interventions"]
    final_rows = records[-1]["layers"]
    if not final_rows or "initializer_causal" not in final_rows[0]:
        print("artifact has no initializer causal test; skipping causal plot")
        return

    styles = {
        "actual": (common.INK, "learned values", "-", "o"),
        "carrier": (common.PURPLE, "residual", "-.", "^"),
        "zero": (common.MUTED, "zero", ":", "s"),
        "shuffled": (common.ORANGE, "shuffled values", "--", "x"),
    }
    conditions = list(styles)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), layout="constrained")

    def causal(row):
        return row["initializer_causal"]

    def paired_sem(row, condition):
        values = np.asarray(causal(row)["loss_delta_batches"][condition], dtype=float)
        finite = values[np.isfinite(values)]
        return finite.std(ddof=1) / finite.size**0.5 if finite.size > 1 else 0.0

    final_layer_rows = [
        next(row for row in record["layers"] if row["layer"] == run["depth"])
        for record in records
    ]
    for condition in conditions:
        color, label, linestyle, marker = styles[condition]
        x = [causal(row)["rms"][condition] for row in final_layer_rows]
        y = [causal(row)["loss_delta"][condition] for row in final_layer_rows]
        axes[0].errorbar(
            x,
            y,
            yerr=[paired_sem(row, condition) for row in final_layer_rows],
            color=color,
            ls=linestyle,
            marker=marker,
            ms=3,
            capsize=1.5,
            label=label,
        )
        axes[0].scatter(
            [x[0]],
            [y[0]],
            s=32,
            facecolors=common.SURFACE,
            edgecolors=color,
            zorder=4,
        )
        axes[0].scatter([x[-1]], [y[-1]], s=34, color=color, zorder=4)
        path_arrow(
            axes[0],
            x,
            y,
            color=color,
            index=longest_visible_segment(x, y),
            linewidth=1.5,
        )

        x = [causal(row)["rms"][condition] for row in final_rows]
        y = [causal(row)["loss_delta"][condition] for row in final_rows]
        axes[1].errorbar(
            x,
            y,
            yerr=[paired_sem(row, condition) for row in final_rows],
            fmt="none",
            ecolor=color,
            alpha=0.8,
            capsize=1.5,
        )
        axes[1].plot(
            x,
            y,
            color=color,
            ls=linestyle,
            label=label,
        )
        for layer_index, (x_value, y_value) in enumerate(zip(x, y)):
            is_first = layer_index == 0
            is_final = layer_index == len(x) - 1
            axes[1].scatter(
                [x_value],
                [y_value],
                marker="o",
                s=38 if is_first or is_final else 14,
                facecolors=common.SURFACE if is_first else color,
                edgecolors=color,
                linewidths=1.0,
                zorder=3,
            )
        path_arrow(
            axes[1],
            x,
            y,
            color=color,
            index=longest_visible_segment(x, y),
            linewidth=1.5,
        )

    for ax in axes:
        ax.axhline(0, color=common.MUTED, lw=0.7, ls=":")
        ax.set_xlabel(r"state RMS to fixed point  $\|m_1-m^\star\|$")
        ax.set_xlim(left=0)
    axes[0].set(
        title="Final layer across training",
        ylabel="held-out CE − learned-values baseline",
    )
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].text(
        0.02,
        0.98,
        "open: initialization · filled: final · arrows: training time",
        transform=axes[0].transAxes,
        va="top",
        color=common.MUTED,
        fontsize=7.5,
    )
    axes[1].set(
        title="Task alignment, not fixed-point proximity",
        ylabel="held-out CE − learned-values baseline",
    )
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].text(
        0.02,
        0.82,
        "large open: L1 · large filled: L6 · arrows: depth",
        transform=axes[1].transAxes,
        ha="left",
        color=common.MUTED,
        fontsize=7.5,
    )
    fig.suptitle(
        f"One frozen update from alternative starting points · depth {run['depth']}",
        fontsize=12,
    )
    common.save_figure(fig, f"{prefix}_initializer_causal", output_dir)


def wrapped_sample(text: str, width: int = 88) -> str:
    lines = []
    for line in text.expandtabs(4).splitlines() or [""]:
        lines.extend(
            textwrap.wrap(
                line,
                width=width,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            or [""]
        )
    return "\n".join(lines).replace("$", r"\$")


def plot_samples(
    data: dict,
    *,
    prefix: str = "language_model",
    output_dir: Path | None = None,
) -> Path | None:
    runs = [run for run in data["runs"] if run["history"].get("samples")]
    if not runs:
        print("no saved samples; skipping GIF")
        return None
    common.style()
    output_dir = common.FIGURES if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = sorted(
        {sample["step"] for run in runs for sample in run["history"]["samples"]}
    )
    fig, axes = plt.subplots(
        len(runs),
        1,
        figsize=(9.2, 1.9 + 1.75 * len(runs)),
        squeeze=False,
    )
    fig.subplots_adjust(left=0.035, right=0.985, bottom=0.05, top=0.84, hspace=0.42)
    prompt_cards, generated_cards = [], []
    for ax, run in zip(axes[:, 0], runs):
        ax.set_facecolor("#f1f3f2")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{run['depth']} layers", loc="left", color=common.INK)
        prompt_cards.append(
            ax.text(
                0.025,
                0.86,
                "",
                transform=ax.transAxes,
                va="top",
                ha="left",
                family="DejaVu Sans Mono",
                fontsize=8.2,
                color=common.INK,
            )
        )
        generated_cards.append(
            ax.text(
                0.025,
                0.65,
                "",
                transform=ax.transAxes,
                va="top",
                ha="left",
                family="DejaVu Sans Mono",
                fontweight="bold",
                fontsize=8.2,
                color=common.INK,
            )
        )
    title = fig.suptitle("", fontsize=11)
    prompt = data.get("config", {}).get("prompt", "saved prompt")

    def update(step):
        title.set_text(
            f"Generation during training · step {step} · "
            "normal prompt, bold continuation"
        )
        for prompt_card, generated_card, run in zip(
            prompt_cards, generated_cards, runs
        ):
            available = [
                sample for sample in run["history"]["samples"] if sample["step"] <= step
            ]
            chosen = available[-1] if available else run["history"]["samples"][0]
            text = chosen["text"]
            continuation = text[len(prompt) :] if text.startswith(prompt) else text
            prompt_card.set_text(wrapped_sample(prompt))
            generated_card.set_text(wrapped_sample(continuation))
        return [title, *prompt_cards, *generated_cards]

    destination = output_dir / f"{prefix}_samples.gif"
    frames = []
    for step in steps:
        update(step)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
        frames.append(Image.fromarray(rgba).convert("RGB"))
    frames[0].save(
        destination,
        save_all=True,
        append_images=frames[1:],
        duration=900,
        loop=0,
        disposal=2,
    )
    plt.close(fig)
    print(f"saved {destination}")
    return destination


def plot(
    data: dict,
    prefix: str = "language_model",
    output_dir: Path | None = None,
) -> None:
    if not data.get("runs"):
        raise ValueError("language-model artifact contains no completed runs")
    common.style()
    plot_training(data, prefix, output_dir)
    plot_timeline(data, prefix, output_dir)
    plot_signal(data, prefix, output_dir)
    plot_fields(data, prefix, output_dir)
    plot_homeostasis(data, prefix, output_dir)
    plot_layer_ablation(data, prefix, output_dir)
    plot_relaxation_interventions(data, prefix, output_dir)
    plot_initializer_causal_test(data, prefix, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=common.DATA / "language_model.pt")
    parser.add_argument("--output-dir", type=Path, default=common.FIGURES)
    parser.add_argument("--prefix", default="language_model")
    parser.add_argument("--no-gif", action="store_true")
    args = parser.parse_args()
    data = common.load_data(args.input)
    plot(data, args.prefix, args.output_dir)
    if not args.no_gif:
        plot_samples(data, prefix=args.prefix, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
