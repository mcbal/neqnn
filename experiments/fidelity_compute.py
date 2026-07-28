"""Compute the sampled/exact/large-D fidelity phase diagrams."""

from __future__ import annotations

import argparse
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

import common
from neqnn import fixed_point as fp
from neqnn import mean_field as mf
from neqnn import proxies, stochastic

OBSERVABLES = ("magnetization", "delayed", "entropy")


def problem(sites: int, dim: int, seed: int):
    torch.manual_seed(seed)
    couplings = torch.softmax(torch.randn(sites, sites), -1)
    torch.manual_seed(seed + 1)
    direction = stochastic.random_state((sites,), dim)
    return direction, couplings


def predict(drive, couplings, beta: float, *, large_d: bool) -> dict:
    step = mf.step_large_d if large_d else mf.step
    delayed = mf.delayed_correlations_large_d if large_d else mf.delayed_correlations
    step_fn = lambda m: step(m, drive, couplings, beta)
    solved = fp.anderson(step_fn, torch.zeros_like(drive), max_iter=80, tol=1e-10)
    field = mf.effective_field(solved.solution, drive, couplings)
    lagged = delayed(field, field, couplings, beta)
    return {
        "magnetization": solved.solution,
        "delayed": lagged,
        "entropy": proxies.entropy_production(couplings, lagged, beta),
        "residual": solved.residual,
    }


def relative(actual: torch.Tensor, reference: torch.Tensor) -> float:
    scale = reference.norm()
    return float((actual - reference).norm() / scale) if scale > 0 else math.nan


def pair_mean(values: torch.Tensor) -> torch.Tensor:
    pairs = itertools.combinations(range(len(values)), 2)
    return torch.stack([(values[i] * values[j]).sum() for i, j in pairs]).mean()


def sampled_errors(reference: dict, estimates, couplings, beta: float):
    entropy = torch.stack(
        [
            proxies.entropy_production(couplings, lagged, beta)
            for lagged in estimates.delayed_correlations
        ]
    )
    replicates = {
        "magnetization": estimates.magnetizations,
        "delayed": estimates.delayed_correlations,
        "entropy": entropy,
    }
    errors, floors = {}, {}
    for name, values in replicates.items():
        denominator = pair_mean(values)
        numerator = pair_mean(reference[name] - values)
        errors[name] = (
            float((numerator.clamp_min(0) / denominator).sqrt())
            if denominator > 0
            else math.nan
        )
        pooled = values.mean(0)
        spread = (values - pooled).reshape(len(values), -1).norm(dim=-1)
        floors[name] = float(
            spread.square().sum().div(len(values) - 1).sqrt()
            / len(values) ** 0.5
            / pooled.norm().clamp_min(torch.finfo(values.dtype).tiny)
        )
    return errors, floors


def branch_count(step_fn, template, starts: int) -> int:
    guesses = [torch.zeros_like(template)]
    for index in range(1, starts):
        torch.manual_seed(10_000 + index)
        guesses.append(
            0.5 * stochastic.random_state(template.shape[:-1], template.shape[-1])
        )
    return len(fp.distinct_fixed_points(step_fn, guesses, max_iter=120, tol=1e-11))


def empty(shape):
    return {name: torch.full(shape, math.nan) for name in OBSERVABLES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=common.DATA / "fidelity.pt")
    parser.add_argument("--sites", type=int, default=32)
    parser.add_argument("--dims", default="3,16,64")
    parser.add_argument("--u", default="0.25,0.5,1,2,4")
    parser.add_argument("--beta", default="0.5,1,2,4,8")
    parser.add_argument("--fine", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chains", type=int, default=8)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--burn-in", type=int, default=24)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dims = common.numbers(args.dims, int)
    us, betas = common.numbers(args.u), common.numbers(args.beta)
    if args.repeats < 3:
        parser.error("--repeats must be at least 3 for noise correction")
    if min(dims) <= 2 or min(us + betas) <= 0:
        parser.error("dimensions must exceed 2; u and beta must be positive")

    torch.set_default_dtype(torch.float64)
    sample_shape = (len(dims), len(betas), len(us))
    fine_us = np.geomspace(min(us), max(us), args.fine).tolist()
    fine_betas = np.geomspace(min(betas), max(betas), args.fine).tolist()
    fine_shape = (len(dims), args.fine, args.fine)
    sampled = empty(sample_shape)
    floor = empty(sample_shape)
    large_d = empty(fine_shape)
    residual = torch.full(sample_shape, math.nan)
    branches = torch.zeros(sample_shape, dtype=torch.int64)
    started = time.time()

    for d, dim in enumerate(dims):
        direction, couplings = problem(args.sites, dim, args.seed)
        for b, beta in enumerate(betas):
            for u_index, u in enumerate(us):
                drive = (u / beta) * direction
                exact = predict(drive, couplings, beta, large_d=False)
                torch.manual_seed(args.seed + 1_000_000 * d + 1_000 * b + u_index)
                estimates = stochastic.estimate_replicates(
                    drive,
                    couplings,
                    beta,
                    num_repeats=args.repeats,
                    num_chains=args.chains,
                    num_steps=args.steps,
                    burn_in=args.burn_in,
                )
                errors, floors = sampled_errors(exact, estimates, couplings, beta)
                for name in OBSERVABLES:
                    sampled[name][d, b, u_index] = errors[name]
                    floor[name][d, b, u_index] = floors[name]
                residual[d, b, u_index] = exact["residual"]
                step_fn = lambda m: mf.step(m, drive, couplings, beta)
                branches[d, b, u_index] = branch_count(step_fn, drive, args.starts)
                print(
                    f"D {dim:>2}  beta {beta:>4g}  u {u:>4g}  "
                    f"m {errors['magnetization']:.2e}/{floors['magnetization']:.2e}  "
                    f"C {errors['delayed']:.2e}/{floors['delayed']:.2e}  "
                    f"sigma {errors['entropy']:.2e}/{floors['entropy']:.2e}  "
                    f"branches {branches[d, b, u_index]}",
                    flush=True,
                )

        for b, beta in enumerate(fine_betas):
            for u_index, u in enumerate(fine_us):
                drive = (u / beta) * direction
                exact = predict(drive, couplings, beta, large_d=False)
                approximate = predict(drive, couplings, beta, large_d=True)
                if max(exact["residual"], approximate["residual"]) < 1e-7:
                    for name in OBSERVABLES:
                        large_d[name][d, b, u_index] = relative(
                            approximate[name], exact[name]
                        )
        print(f"finished D={dim} in {(time.time() - started) / 60:.1f} min")

    common.save_data(
        {
            "config": vars(args) | {"output": str(args.output)},
            "dims": dims,
            "u": us,
            "beta": betas,
            "fine_u": fine_us,
            "fine_beta": fine_betas,
            "sampled_error": sampled,
            "sampling_floor": floor,
            "large_d_error": large_d,
            "residual": residual,
            "branches": branches,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
