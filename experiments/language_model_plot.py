"""Plot training diagnostics and samples from language_model_compute.py output."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
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
                f"{run['depth']} layers · "
                f"{format_parameters(run['parameters'])} params"
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
    """Show the thermodynamic layer diagnostics on the training clock."""
    runs = data["runs"]
    has_fixed_point_entropy = all(
        "entropy_fixed_point_converged" in row
        for run in runs
        for probe in run["history"]["probe"]
        for row in probe["forward"]
    )
    has_relaxation_mismatch = all(
        "relaxation_mismatch" in row
        for run in runs
        for probe in run["history"]["probe"]
        for row in probe["forward"]
    )
    metrics = []
    if has_fixed_point_entropy:
        metrics.append(("entropy", "housekeeping entropy production"))
    if has_relaxation_mismatch:
        metrics.append(
            (
                "relaxation_mismatch",
                r"one-step mismatch  "
                r"$D_{KL}[p_{h(m_0)}\,\|\,p_{h(m_1)}]$",
            )
        )
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
            for value in np.linspace(0.05, 0.95, max(run["depth"], 2))[
                : run["depth"]
            ]
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
            f"{run['depth']} layers · "
            f"{format_parameters(run['parameters'])} parameters"
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
    fig.suptitle("Proxy diagnostics during training", fontsize=12)
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
    fig.suptitle("Signal propagation across depth", fontsize=12)
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
    """Final depth profiles for saturation and local response sensitivity."""
    runs = data["runs"]
    required = {
        "saturation_p50",
        "saturation_p95",
        "saturation_max",
        "effective_u_p50",
        "effective_u_p95",
        "effective_u_max",
        "increment_u",
        "increment_radial_u",
        "increment_radial_abs_u",
        "increment_transverse_u",
        "susceptibility_tangential",
        "susceptibility_radial",
        "susceptibility_radial_p05",
    }
    final_rows = [run["history"]["probe"][-1]["forward"] for run in runs]
    if any(required - row.keys() for final in final_rows for row in final):
        print("artifact has no homeostasis monitors; skipping homeostasis plot")
        return

    fig, axes = plt.subplots(
        4,
        len(runs),
        figsize=(3.55 * len(runs), 9.1),
        sharex="col",
        sharey="row",
        layout="constrained",
        squeeze=False,
    )
    for column, (run, final) in enumerate(zip(runs, final_rows)):
        layers = np.arange(1, run["depth"] + 1)
        axes[0, column].plot(
            layers,
            [row["saturation_p50"] for row in final],
            color=common.BLUE,
            label="median",
        )
        axes[0, column].plot(
            layers,
            [row["saturation_p95"] for row in final],
            color=common.ORANGE,
            label="95th percentile",
        )
        axes[0, column].plot(
            layers,
            [row["saturation_max"] for row in final],
            color=common.INK,
            ls=":",
            label="maximum",
        )
        axes[0, column].axhspan(0.7, 0.8, color=common.ORANGE, alpha=0.06)
        axes[0, column].axhspan(0.8, 1.0, color=common.ORANGE, alpha=0.12)

        for key, color, label, style in (
            ("effective_u_p50", common.BLUE, "median", "-"),
            ("effective_u_p95", common.ORANGE, "95th percentile", "-"),
            ("effective_u_max", common.INK, "maximum", ":"),
        ):
            axes[1, column].plot(
                layers,
                [row[key] for row in final],
                color=color,
                ls=style,
                label=label,
            )

        for key, color, label, style in (
            ("increment_u", common.INK, r"$\|\Delta h\|$", "-"),
            ("increment_transverse_u", common.BLUE, "transverse", "-"),
            ("increment_radial_u", common.ORANGE, "signed radial", "-"),
            ("increment_radial_abs_u", common.ORANGE, "absolute radial", ":"),
        ):
            axes[2, column].plot(
                layers,
                [row[key] for row in final],
                color=color,
                ls=style,
                label=label,
            )
        axes[2, column].axhline(0, color=common.MUTED, lw=0.7)

        for key, color, label, style in (
            ("susceptibility_tangential", common.BLUE, "tangential mean", "-"),
            ("susceptibility_radial", common.ORANGE, "radial mean", "-"),
            ("susceptibility_radial_p05", common.INK, "radial 5th percentile", ":"),
        ):
            axes[3, column].plot(
                layers,
                [row[key] for row in final],
                color=color,
                ls=style,
                label=label,
            )

        axes[0, column].set_title(f"{run['depth']} layers")
        axes[0, column].set_ylim(0, 1)
        axes[3, column].set_xlabel("layer")

    for row, label in enumerate(
        (
            r"output saturation  $\|m\|/R$",
            r"effective field  $u=\beta\|h\|/R$",
            r"field increment in $u$ units",
            "local susceptibility",
        )
    ):
        axes[row, 0].set_ylabel(label)
        axes[row, 0].legend(fontsize=7.2)
    fig.suptitle("Module-local homeostasis at the final checkpoint", fontsize=12)
    common.save_figure(fig, f"{prefix}_homeostasis", output_dir)


def plot_effective_depth(data: dict, prefix: str, output_dir: Path | None) -> None:
    """Show whether successive residual updates still rotate the representation."""
    runs = data["runs"]
    required = {
        "direction_change",
        "direction_change_p95",
        "increment_relative",
        "increment_radial_relative",
        "increment_transverse_relative",
        "ffn_increment_radial_u",
        "coupling_increment_radial_u",
    }
    final_rows = [run["history"]["probe"][-1]["forward"] for run in runs]
    if any(required - row.keys() for final in final_rows for row in final):
        print("artifact has no effective-depth monitors; skipping effective-depth plot")
        return

    fig, axes = plt.subplots(
        3,
        len(runs),
        figsize=(3.55 * len(runs), 7.4),
        sharex="col",
        sharey="row",
        layout="constrained",
        squeeze=False,
    )
    for column, (run, final) in enumerate(zip(runs, final_rows)):
        layers = np.arange(1, run["depth"] + 1)
        axes[0, column].plot(
            layers,
            np.degrees([row["direction_change"] for row in final]),
            color=common.BLUE,
            label="mean",
        )
        axes[0, column].plot(
            layers,
            np.degrees([row["direction_change_p95"] for row in final]),
            color=common.ORANGE,
            label="95th percentile",
        )

        for key, color, label in (
            ("increment_relative", common.INK, "total"),
            ("increment_transverse_relative", common.BLUE, "transverse"),
            ("increment_radial_relative", common.ORANGE, "signed radial"),
        ):
            axes[1, column].plot(
                layers, [row[key] for row in final], color=color, label=label
            )
        axes[1, column].axhline(0, color=common.MUTED, lw=0.7)

        for key, color, label in (
            ("increment_radial_u", common.INK, "total"),
            ("ffn_increment_radial_u", common.ORANGE, "FFN"),
            ("coupling_increment_radial_u", common.GREEN, "attention"),
        ):
            axes[2, column].plot(
                layers, [row[key] for row in final], color=color, label=label
            )
        axes[2, column].axhline(0, color=common.MUTED, lw=0.7)

        axes[0, column].set_title(f"{run['depth']} layers")
        axes[2, column].set_xlabel("layer")

    for row, label in enumerate(
        (
            "field-direction change (degrees)",
            r"increment relative to carrier  $\Delta h/h$",
            r"signed radial increment in $u$ units",
        )
    ):
        axes[row, 0].set_ylabel(label)
        axes[row, 0].legend(fontsize=7.2)
    fig.suptitle("Effective depth in conjugate-field coordinates", fontsize=12)
    common.save_figure(fig, f"{prefix}_effective_depth", output_dir)


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
    fig, (loss_ax, kl_ax) = plt.subplots(
        1, 2, figsize=(9.5, 3.7), layout="constrained"
    )
    for run, color in zip(runs, colors):
        rows = run["history"]["layer_ablation"]["layers"]
        layers = [row["layer"] for row in rows]
        label = f"{run['depth']} layers"
        loss_ax.plot(
            layers,
            [row["loss_delta"] for row in rows],
            color=color,
            marker="o",
            ms=3,
            label=label,
        )
        kl_ax.plot(
            layers,
            [max(row["kl_from_full"], np.finfo(float).tiny) for row in rows],
            color=color,
            marker="o",
            ms=3,
            label=label,
        )
    loss_ax.axhline(0, color=common.MUTED, lw=0.8)
    loss_ax.set(
        xlabel="skipped layer",
        ylabel=r"held-out loss change  $\mathcal{L}_{skip}-\mathcal{L}_{full}$",
        title="task effect",
    )
    kl_ax.set(
        xlabel="skipped layer",
        ylabel=r"$D_{KL}(p_{full}\,\|\,p_{skip})$",
        yscale="log",
        title="output-distribution effect",
    )
    loss_ax.legend()
    kl_ax.legend()
    common.save_figure(fig, f"{prefix}_layer_ablation", output_dir)


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
        {
            sample["step"]
            for run in runs
            for sample in run["history"]["samples"]
        }
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
                sample
                for sample in run["history"]["samples"]
                if sample["step"] <= step
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
    plot_effective_depth(data, prefix, output_dir)
    plot_layer_ablation(data, prefix, output_dir)


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
