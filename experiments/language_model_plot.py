"""Plot loss and signal propagation from language_model_compute.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import common


def smooth(values, width=15):
    width = min(width, len(values))
    return np.convolve(values, np.ones(width) / width, mode="valid"), width


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=common.DATA / "language_model.pt")
    args = parser.parse_args()
    data = common.load_data(args.input)
    runs = data["runs"]
    common.style()
    colors = ["#76a9df", common.BLUE, "#183d6b"]

    fig, (loss_ax, grad_ax) = plt.subplots(
        1, 2, figsize=(9.4, 3.8), layout="constrained"
    )
    for index, (run, color) in enumerate(zip(runs, colors)):
        history = run["history"]
        rolling, width = smooth(history["train"])
        loss_ax.plot(
            history["step"][width - 1 :],
            rolling,
            color=color,
            label=f"{run['depth']} layers · {run['parameters'] / 1e3:.0f}k params",
        )
        loss_ax.scatter(history["eval_step"], history["valid"], color=color, s=18)
        grad_ax.plot(
            [probe["step"] for probe in history["probe"]],
            [probe["grad_norm"] for probe in history["probe"]],
            color=color,
            marker="o",
            markersize=3,
            label="pre-clip" if index == 0 else None,
        )
        grad_ax.plot(
            [probe["step"] for probe in history["probe"]],
            [
                probe.get(
                    "grad_norm_post",
                    min(probe["grad_norm"], data["config"]["clip_grad"]),
                )
                for probe in history["probe"]
            ],
            color=color,
            ls="--",
            alpha=0.55,
            label="post-clip" if index == 0 else None,
        )
    for order, value in data.get("baselines", {}).items():
        loss_ax.axhline(value, color=common.MUTED, lw=0.8, ls=":", alpha=0.65)
        loss_ax.annotate(
            f"{order}-gram",
            (runs[0]["history"]["step"][-1], value),
            xytext=(-3, 2),
            textcoords="offset points",
            ha="right",
            color=common.MUTED,
            fontsize=7,
        )
    loss_ax.set(xlabel="optimizer step", ylabel="cross entropy (nats)")
    grad_ax.set(xlabel="optimizer step", ylabel="global gradient norm", yscale="log")
    grad_ax.axhline(
        data["config"]["clip_grad"], color=common.MUTED, ls=":", lw=1, label="clip"
    )
    loss_ax.set_title("character-level Dostoevsky")
    grad_ax.set_title("optimization stability")
    loss_ax.legend()
    grad_ax.legend()
    common.save_figure(fig, "language_model_training")

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
                layers, fields / fields[0], style, color=common.BLUE, alpha=alpha, label=label
            )
            axes[1, column].plot(
                layers,
                gradients / gradients[-1],
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
    common.save_figure(fig, "language_model_signal")

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
        for key, color, label in (
            ("residual", common.BLUE, "residual"),
            ("ffn", common.ORANGE, "FFN"),
            ("coupling", common.GREEN, "coupling"),
        ):
            ax.plot(layers, [row[key] for row in final], color=color, label=label)
        ax.set_title(f"{run['depth']} layers")
        ax.set_xlabel("layer")
    axes[0, 0].set_ylabel("mean per-site norm")
    axes[0, 0].legend()
    fig.suptitle("Final field decomposition", fontsize=12)
    common.save_figure(fig, "language_model_fields")


if __name__ == "__main__":
    main()
