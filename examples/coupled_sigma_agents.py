#!/usr/bin/env python
"""Two spin agents coupled through detached interfaces.

Agent A consumes agent B's previous output, agent B consumes agent A's previous
output, and each agent locally optimizes only its own entropy-production loss.
The experiment logs whether the closed loop settles into a low-drift regime,
keeps wandering, or becomes numerically pathological.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib-neqnn"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from neqnn.models import SpinTransformerModel


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument(
        "--ffn-mode",
        choices=["full", "per_head"],
        default="full",
        help="Use dense channel-mixing memory or shared head-local memory.",
    )
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--init-scale", type=float, default=1.0)
    parser.add_argument("--drive-noise", type=float, default=0.01)
    parser.add_argument("--interface-mix", type=float, default=1.0)
    parser.add_argument(
        "--update-mode",
        choices=["parallel", "sequential"],
        default="sequential",
        help="Sequential lets B consume A's fresh output; parallel reads stale interfaces.",
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--objective",
        choices=["maximize", "minimize"],
        default="maximize",
        help="Maximize sigma with loss=-sigma, or minimize it with loss=sigma.",
    )
    parser.add_argument(
        "--pre-mix",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use learned input mixing inside each spin agent.",
    )
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use causal attention masks inside each spin agent.",
    )
    parser.add_argument(
        "--settle-delta",
        type=float,
        default=1e-3,
        help="Mean closed-loop RMS drift below this is called equilibrium-like.",
    )
    parser.add_argument(
        "--settle-slope",
        type=float,
        default=1e-4,
        help="Absolute sigma slope below this is called equilibrium-like.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Directory for CSV, config, summary, and plots.",
    )
    return parser


def build_agent(args: argparse.Namespace) -> SpinTransformerModel:
    return SpinTransformerModel(
        num_layers=args.num_layers,
        dim=args.dim,
        num_heads=args.num_heads,
        pre_mix=args.pre_mix,
        post_mix=False,
        causal=args.causal,
        num_steps=args.num_steps,
        ffn_mode=args.ffn_mode,
        beta=args.beta,
        return_sigmas=True,
        should_detach=False,
    ).to(args.device)


def sigma_mean(sigmas: list[torch.Tensor]) -> torch.Tensor:
    return sum(sigma.mean() for sigma in sigmas)


def local_loss(
    sigmas: list[torch.Tensor], objective: str
) -> tuple[torch.Tensor, torch.Tensor]:
    sigma = sigma_mean(sigmas)
    if objective == "maximize":
        return -sigma, sigma
    if objective == "minimize":
        return sigma, sigma
    raise ValueError(f"unknown objective: {objective}")


def rms(tensor: torch.Tensor) -> float:
    return float(tensor.detach().pow(2).mean().sqrt().cpu().item())


def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.detach().reshape(-1, a.size(-1))
    b_flat = b.detach().reshape(-1, b.size(-1))
    return float(F.cosine_similarity(a_flat, b_flat, dim=-1).mean().cpu().item())


def vector_norm_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    a_norm = torch.linalg.vector_norm(a.detach(), dim=-1)
    b_norm = torch.linalg.vector_norm(b.detach(), dim=-1)
    return rms(a_norm - b_norm) / math.sqrt(a.size(-1))


def direction_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    a_dir = F.normalize(a.detach(), dim=-1)
    b_dir = F.normalize(b.detach(), dim=-1)
    return rms(a_dir - b_dir)


def grad_norm(parameters) -> float:
    grad_sq = 0.0
    for param in parameters:
        if param.grad is not None:
            grad_sq += float(param.grad.detach().pow(2).sum().cpu().item())
    return math.sqrt(grad_sq)


def param_norm(parameters) -> float:
    param_sq = 0.0
    for param in parameters:
        param_sq += float(param.detach().pow(2).sum().cpu().item())
    return math.sqrt(param_sq)


def finite_row(row: dict[str, float | int | str]) -> bool:
    for value in row.values():
        if isinstance(value, float) and not math.isfinite(value):
            return False
    return True


def slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    return float(np.polyfit(x, y, deg=1)[0])


def classify_run(
    logs: list[dict[str, float | int | str]],
    *,
    settle_delta: float,
    settle_slope: float,
) -> dict[str, float | str]:
    if not logs:
        return {"status": "empty", "reason": "no steps were recorded"}
    if not all(finite_row(row) for row in logs):
        return {"status": "pathological", "reason": "non-finite value encountered"}

    window = min(100, max(10, len(logs) // 5))
    tail = logs[-window:]
    loop_delta = float(np.mean([row["loop_delta"] for row in tail]))
    two_step_delta = float(np.mean([row["two_step_loop_delta"] for row in tail]))
    sigma_total = [row["sigma_a"] + row["sigma_b"] for row in tail]
    sigma_abs = float(np.mean([abs(value) for value in sigma_total]))
    sigma_slope = slope(sigma_total)
    max_grad = max(float(row["grad_norm_a"] + row["grad_norm_b"]) for row in tail)

    thermo_quiet = sigma_abs < settle_slope and abs(sigma_slope) < settle_slope
    fixed_point = loop_delta < settle_delta
    two_cycle = two_step_delta < settle_delta <= loop_delta

    if thermo_quiet and fixed_point:
        status = "fixed-point-equilibrium"
        reason = "low entropy production and low one-step interface drift"
    elif thermo_quiet and two_cycle:
        status = "two-cycle-equilibrium"
        reason = "low entropy production with low two-step interface drift"
    elif thermo_quiet:
        status = "thermodynamically-quiet"
        reason = "low entropy production while interface state is still moving"
    elif max_grad > 1e4:
        status = "pathological"
        reason = "very large gradients in the tail window"
    else:
        status = "drifting"
        reason = "entropy production or interface state is still moving"

    return {
        "status": status,
        "reason": reason,
        "tail_window": window,
        "tail_loop_delta_mean": loop_delta,
        "tail_two_step_loop_delta_mean": two_step_delta,
        "tail_sigma_abs_mean": sigma_abs,
        "tail_sigma_slope": sigma_slope,
        "tail_max_grad_norm_sum": max_grad,
    }


def save_plots(run_dir: Path, logs: list[dict[str, float | int | str]]):
    steps = [row["step"] for row in logs]

    plt.figure(figsize=(10, 4))
    plt.plot(steps, [row["sigma_a"] for row in logs], label="sigma A")
    plt.plot(steps, [row["sigma_b"] for row in logs], label="sigma B")
    plt.plot(
        steps,
        [row["sigma_a"] + row["sigma_b"] for row in logs],
        label="sigma total",
        alpha=0.7,
    )
    plt.xlabel("step")
    plt.ylabel("mean sigma")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "sigmas.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(steps, [row["loss_a"] for row in logs], label="loss A")
    plt.plot(steps, [row["loss_b"] for row in logs], label="loss B")
    plt.xlabel("step")
    plt.ylabel("local loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "losses.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(steps, [row["loop_delta"] for row in logs], label="one-step drift")
    plt.plot(
        steps,
        [row["two_step_loop_delta"] for row in logs],
        label="two-step drift",
        alpha=0.85,
    )
    plt.plot(
        steps, [row["a_input_output_delta"] for row in logs], label="A input/output"
    )
    plt.plot(
        steps, [row["b_input_output_delta"] for row in logs], label="B input/output"
    )
    plt.xlabel("step")
    plt.ylabel("RMS delta")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "interface_deltas.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(
        steps,
        [row["one_step_direction_delta"] for row in logs],
        label="one-step direction",
    )
    plt.plot(
        steps,
        [row["one_step_norm_delta"] for row in logs],
        label="one-step norm",
    )
    plt.plot(
        steps,
        [row["two_step_direction_delta"] for row in logs],
        label="two-step direction",
        alpha=0.85,
    )
    plt.plot(
        steps,
        [row["two_step_norm_delta"] for row in logs],
        label="two-step norm",
        alpha=0.85,
    )
    plt.xlabel("step")
    plt.ylabel("RMS delta")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "interface_delta_decomposition.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(
        steps,
        [row["interface_alignment"] for row in logs],
        label="A->B vs B->A",
    )
    plt.plot(
        steps,
        [row["one_step_interface_alignment"] for row in logs],
        label="one-step self",
    )
    plt.plot(
        steps,
        [row["two_step_interface_alignment"] for row in logs],
        label="two-step self",
    )
    plt.xlabel("step")
    plt.ylabel("cosine alignment")
    plt.ylim(-1.05, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "interface_alignment.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(steps, [row["param_norm_a"] for row in logs], label="param norm A")
    plt.plot(steps, [row["param_norm_b"] for row in logs], label="param norm B")
    plt.plot(
        steps, [row["grad_norm_a"] for row in logs], label="grad norm A", alpha=0.7
    )
    plt.plot(
        steps, [row["grad_norm_b"] for row in logs], label="grad norm B", alpha=0.7
    )
    plt.xlabel("step")
    plt.ylabel("norm")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "norms.png", dpi=160)
    plt.close()


def write_csv(path: Path, logs: list[dict[str, float | int | str]]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(logs[0].keys()))
        writer.writeheader()
        writer.writerows(logs)


def validate_args(args: argparse.Namespace):
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be positive")
    if args.dim % args.num_heads != 0:
        raise ValueError("--dim must be divisible by --num-heads")
    if args.dim // args.num_heads <= 2:
        raise ValueError("--dim / --num-heads must be greater than 2")
    if not 0.0 < args.interface_mix <= 1.0:
        raise ValueError("--interface-mix must be in (0, 1]")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.grad_clip < 0.0:
        raise ValueError("--grad-clip must be non-negative")


def main():
    args = build_arg_parser().parse_args()
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    run_dir = args.run_dir or Path("/tmp") / "neqnn-coupled-agents" / timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)

    agent_a = build_agent(args)
    agent_b = build_agent(args)
    opt_a = torch.optim.AdamW(
        agent_a.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    opt_b = torch.optim.AdamW(
        agent_b.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    shape = (args.batch_size, args.seq_len, args.dim)
    a_to_b = args.init_scale * torch.randn(shape, device=args.device)
    b_to_a = args.init_scale * torch.randn(shape, device=args.device)
    prev_a_to_b = None
    prev_b_to_a = None
    logs = []
    start_time = time.time()

    print(f"writing run artifacts to {run_dir}")
    print(
        f"objective={args.objective}, steps={args.steps}, "
        f"shape={shape}, beta={args.beta}, lr={args.lr}, "
        f"update_mode={args.update_mode}"
    )

    def agent_step(agent, optimizer, agent_input):
        optimizer.zero_grad(set_to_none=True)
        out, sigmas = agent(agent_input)
        loss, sigma = local_loss(sigmas, args.objective)
        loss.backward()
        grad = grad_norm(agent.parameters())
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(agent.parameters(), args.grad_clip)
        optimizer.step()
        return out, loss, sigma, grad

    for step in range(args.steps):
        a_in = b_to_a.detach()
        if args.drive_noise:
            a_in = a_in + args.drive_noise * torch.randn_like(a_in)
        a_out, loss_a, sigma_a, grad_norm_a = agent_step(agent_a, opt_a, a_in)
        next_a_to_b = torch.lerp(a_to_b, a_out.detach(), args.interface_mix)

        if args.update_mode == "parallel":
            b_in = a_to_b.detach()
        else:
            b_in = next_a_to_b.detach()
        if args.drive_noise:
            b_in = b_in + args.drive_noise * torch.randn_like(b_in)
        b_out, loss_b, sigma_b, grad_norm_b = agent_step(agent_b, opt_b, b_in)

        with torch.no_grad():
            next_b_to_a = torch.lerp(b_to_a, b_out.detach(), args.interface_mix)
            delta_a_to_b = rms(next_a_to_b - a_to_b)
            delta_b_to_a = rms(next_b_to_a - b_to_a)
            loop_delta = 0.5 * (delta_a_to_b + delta_b_to_a)
            one_step_alignment = 0.5 * (
                cosine_mean(next_a_to_b, a_to_b)
                + cosine_mean(next_b_to_a, b_to_a)
            )
            if prev_a_to_b is None or prev_b_to_a is None:
                two_step_delta_a_to_b = loop_delta
                two_step_delta_b_to_a = loop_delta
                two_step_loop_delta = loop_delta
                two_step_alignment = one_step_alignment
                two_step_norm_delta = 0.5 * (
                    vector_norm_delta(next_a_to_b, a_to_b)
                    + vector_norm_delta(next_b_to_a, b_to_a)
                )
                two_step_direction_delta = 0.5 * (
                    direction_delta(next_a_to_b, a_to_b)
                    + direction_delta(next_b_to_a, b_to_a)
                )
            else:
                two_step_delta_a_to_b = rms(next_a_to_b - prev_a_to_b)
                two_step_delta_b_to_a = rms(next_b_to_a - prev_b_to_a)
                two_step_loop_delta = 0.5 * (
                    two_step_delta_a_to_b + two_step_delta_b_to_a
                )
                two_step_alignment = 0.5 * (
                    cosine_mean(next_a_to_b, prev_a_to_b)
                    + cosine_mean(next_b_to_a, prev_b_to_a)
                )
                two_step_norm_delta = 0.5 * (
                    vector_norm_delta(next_a_to_b, prev_a_to_b)
                    + vector_norm_delta(next_b_to_a, prev_b_to_a)
                )
                two_step_direction_delta = 0.5 * (
                    direction_delta(next_a_to_b, prev_a_to_b)
                    + direction_delta(next_b_to_a, prev_b_to_a)
                )
            sigma_total = float((sigma_a + sigma_b).detach().cpu().item())
            one_step_norm_delta = 0.5 * (
                vector_norm_delta(next_a_to_b, a_to_b)
                + vector_norm_delta(next_b_to_a, b_to_a)
            )
            one_step_direction_delta = 0.5 * (
                direction_delta(next_a_to_b, a_to_b)
                + direction_delta(next_b_to_a, b_to_a)
            )

            row = {
                "step": step,
                "update_mode": args.update_mode,
                "loss_a": float(loss_a.detach().cpu().item()),
                "loss_b": float(loss_b.detach().cpu().item()),
                "sigma_a": float(sigma_a.detach().cpu().item()),
                "sigma_b": float(sigma_b.detach().cpu().item()),
                "sigma_total": sigma_total,
                "loop_delta": loop_delta,
                "delta_a_to_b": delta_a_to_b,
                "delta_b_to_a": delta_b_to_a,
                "two_step_loop_delta": two_step_loop_delta,
                "two_step_delta_a_to_b": two_step_delta_a_to_b,
                "two_step_delta_b_to_a": two_step_delta_b_to_a,
                "a_input_output_delta": rms(a_out - a_in),
                "b_input_output_delta": rms(b_out - b_in),
                "a_out_rms": rms(a_out),
                "b_out_rms": rms(b_out),
                "interface_rms": 0.5 * (rms(next_a_to_b) + rms(next_b_to_a)),
                "one_step_norm_delta": one_step_norm_delta,
                "one_step_direction_delta": one_step_direction_delta,
                "two_step_norm_delta": two_step_norm_delta,
                "two_step_direction_delta": two_step_direction_delta,
                "interface_alignment": cosine_mean(next_a_to_b, next_b_to_a),
                "one_step_interface_alignment": one_step_alignment,
                "two_step_interface_alignment": two_step_alignment,
                "grad_norm_a": grad_norm_a,
                "grad_norm_b": grad_norm_b,
                "param_norm_a": param_norm(agent_a.parameters()),
                "param_norm_b": param_norm(agent_b.parameters()),
            }
            logs.append(row)
            prev_a_to_b = a_to_b.detach()
            prev_b_to_a = b_to_a.detach()
            a_to_b = next_a_to_b.detach()
            b_to_a = next_b_to_a.detach()

        if step == 0 or (step + 1) % args.log_every == 0:
            print(
                f"step {step + 1:5d}/{args.steps}: "
                f"sigma=({row['sigma_a']:.4f}, {row['sigma_b']:.4f}) "
                f"d1={row['loop_delta']:.4e} "
                f"d2={row['two_step_loop_delta']:.4e} "
                f"dir={row['one_step_direction_delta']:.2e} "
                f"norm={row['one_step_norm_delta']:.2e} "
                f"align=({row['interface_alignment']:.3f}, "
                f"{row['one_step_interface_alignment']:.3f}, "
                f"{row['two_step_interface_alignment']:.3f}) "
                f"grad=({row['grad_norm_a']:.2e}, {row['grad_norm_b']:.2e})"
            )

        if not finite_row(row):
            print(f"stopping early at step {step}: non-finite metric encountered")
            break

    summary = classify_run(
        logs,
        settle_delta=args.settle_delta,
        settle_slope=args.settle_slope,
    )
    summary["elapsed_sec"] = time.time() - start_time
    summary["steps_recorded"] = len(logs)

    write_csv(run_dir / "metrics.csv", logs)
    save_plots(run_dir, logs)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (run_dir / "config.json").write_text(
        json.dumps(
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            indent=2,
        )
    )

    print(
        f"summary: {summary['status']} ({summary['reason']}); "
        f"tail_delta={summary.get('tail_loop_delta_mean', float('nan')):.4e}, "
        f"tail_delta_2={summary.get('tail_two_step_loop_delta_mean', float('nan')):.4e}, "
        f"tail_sigma_slope={summary.get('tail_sigma_slope', float('nan')):.4e}"
    )
    print(f"saved metrics, summary, config, and plots to {run_dir}")


if __name__ == "__main__":
    main()
