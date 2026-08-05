"""Compute the sampled/exact/large-D fidelity phase diagrams."""

from __future__ import annotations

import argparse
import itertools
import math
import time
from pathlib import Path

import numpy as np
import torch

try:
    import common
except ModuleNotFoundError:  # Allow importing helpers from the repository root.
    from experiments import common
from neqnn import fixed_point as fp
from neqnn import mean_field as mf
from neqnn import proxies, stochastic, vmf

OBSERVABLES = ("magnetization", "delayed", "entropy")


def problem(sites: int, dim: int, seed: int, device: torch.device):
    torch.manual_seed(seed)
    couplings = torch.softmax(torch.randn(sites, sites, device=device), -1)
    torch.manual_seed(seed + 1)
    direction = stochastic.random_state((sites,), dim, device=device)
    return direction, couplings


def predict(
    drive,
    couplings,
    beta: float,
    *,
    large_d: bool,
    max_iter: int = 80,
    tol: float = 1e-10,
) -> dict:
    step = mf.step_large_d if large_d else mf.step
    delayed = mf.delayed_correlations_large_d if large_d else mf.delayed_correlations
    step_fn = lambda m: step(m, drive, couplings, beta)
    solved = fp.anderson(
        step_fn, torch.zeros_like(drive), max_iter=max_iter, tol=tol
    )
    field = mf.effective_field(solved.solution, drive, couplings)
    lagged = delayed(field, field, couplings, beta)
    return {
        "magnetization": solved.solution,
        "delayed": lagged,
        "entropy": proxies.entropy_production(couplings, lagged, beta),
        "solution": solved.solution,
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


def sampled_diagnostics(estimates, dim: int) -> tuple[float, float]:
    """Return directional chain disagreement and per-chain polarization."""
    pooled = estimates.magnetizations.mean(0)
    pooled_magnitude = pooled.norm(dim=-1).mean()
    chain_magnitude = estimates.chain_magnetizations.mean()
    tiny = torch.finfo(pooled_magnitude.dtype).tiny
    ergodicity = float(chain_magnitude / pooled_magnitude.clamp_min(tiny))
    saturation = float(chain_magnitude) / vmf.radius(dim)
    return ergodicity, saturation


def branch_count(
    step_fn,
    template,
    starts: int,
    *,
    max_iter: int,
    tol: float,
    residual_tol: float,
) -> int:
    guesses = [torch.zeros_like(template)]
    for index in range(1, starts):
        torch.manual_seed(10_000 + index)
        guesses.append(
            0.5
            * stochastic.random_state(
                template.shape[:-1],
                template.shape[-1],
                dtype=template.dtype,
                device=template.device,
            )
        )
    return len(
        fp.distinct_fixed_points(
            step_fn,
            guesses,
            max_iter=max_iter,
            tol=tol,
            residual_tol=residual_tol,
        )
    )


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
    parser.add_argument("--solve-iterations", type=int, default=80)
    parser.add_argument("--branch-iterations", type=int, default=120)
    parser.add_argument("--power-iterations", type=int, default=30)
    parser.add_argument("--solve-tol", type=float, default=1e-10)
    parser.add_argument("--residual-tol", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.device = torch.device(args.device)

    try:
        dims = sorted(set(common.numbers(args.dims, int)))
        us = sorted(set(common.numbers(args.u)))
        betas = sorted(set(common.numbers(args.beta)))
    except ValueError as error:
        parser.error(str(error))
    if args.repeats < 3:
        parser.error("--repeats must be at least 3 for noise correction")
    if (
        min(dims) <= 2
        or any(not math.isfinite(value) or value <= 0 for value in us + betas)
    ):
        parser.error("dimensions must exceed 2; u and beta must be positive")
    if args.sites < 2:
        parser.error("--sites must be at least 2")
    if args.fine < 2:
        parser.error("--fine must be at least 2")
    if args.chains < 2 or args.steps < 2 or args.burn_in < 0:
        parser.error("--chains and --steps must be at least 2; --burn-in must be non-negative")
    if args.starts < 1:
        parser.error("--starts must be positive")
    if min(args.solve_iterations, args.branch_iterations, args.power_iterations) < 2:
        parser.error("solver and power iteration counts must be at least 2")
    if args.solve_tol <= 0 or args.residual_tol <= 0:
        parser.error("solver tolerances must be positive")

    torch.set_default_dtype(torch.float64)
    sample_shape = (len(dims), len(betas), len(us))
    fine_us = np.geomspace(min(us), max(us), args.fine).tolist()
    fine_betas = np.geomspace(min(betas), max(betas), args.fine).tolist()
    fine_shape = (len(dims), args.fine, args.fine)
    sampled = empty(sample_shape)
    floor = empty(sample_shape)
    large_d = empty(fine_shape)
    residual = torch.full(sample_shape, math.nan)
    contraction = torch.full(sample_shape, math.nan)
    branches = torch.zeros(sample_shape, dtype=torch.int64)
    ergodicity = torch.full(sample_shape, math.nan)
    chain_saturation = torch.full(sample_shape, math.nan)
    fine_residual_exact = torch.full(fine_shape, math.nan)
    fine_residual_large_d = torch.full(fine_shape, math.nan)
    fine_contraction = torch.full(fine_shape, math.nan)
    config = {
        key: str(value) if isinstance(value, (Path, torch.device)) else value
        for key, value in vars(args).items()
    }

    def payload(completed_dims: int) -> dict:
        return {
            "schema_version": 2,
            "config": config,
            "dims": dims,
            "u": us,
            "beta": betas,
            "fine_u": fine_us,
            "fine_beta": fine_betas,
            "sampled_error": sampled,
            "sampling_floor": floor,
            "large_d_error": large_d,
            "residual": residual,
            "contraction": contraction,
            "branches": branches,
            "ergodicity": ergodicity,
            "chain_saturation": chain_saturation,
            "fine_residual_exact": fine_residual_exact,
            "fine_residual_large_d": fine_residual_large_d,
            "fine_contraction": fine_contraction,
            "completed_dims": dims[:completed_dims],
            "complete": completed_dims == len(dims),
        }

    started = time.time()

    for d, dim in enumerate(dims):
        direction, couplings = problem(args.sites, dim, args.seed, args.device)
        exact_cache = {}

        def exact_prediction(beta: float, u: float):
            key = (f"{beta:.12g}", f"{u:.12g}")
            if key not in exact_cache:
                exact_cache[key] = predict(
                    (u / beta) * direction,
                    couplings,
                    beta,
                    large_d=False,
                    max_iter=args.solve_iterations,
                    tol=args.solve_tol,
                )
            return exact_cache[key]

        for b, beta in enumerate(betas):
            for u_index, u in enumerate(us):
                drive = (u / beta) * direction
                exact = exact_prediction(beta, u)
                residual[d, b, u_index] = exact["residual"]
                step_fn = lambda m: mf.step(m, drive, couplings, beta)
                branches[d, b, u_index] = branch_count(
                    step_fn,
                    drive,
                    args.starts,
                    max_iter=args.branch_iterations,
                    tol=args.solve_tol,
                    residual_tol=args.residual_tol,
                )
                if exact["residual"] <= args.residual_tol:
                    contraction[d, b, u_index] = fp.local_contraction(
                        step_fn,
                        exact["solution"],
                        iterations=args.power_iterations,
                    )
                else:
                    print(
                        f"D {dim:>2}  beta {beta:>4g}  u {u:>4g}  "
                        f"fixed-point solve failed ({exact['residual']:.2e}); "
                        "sampling skipped",
                        flush=True,
                    )
                    continue

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
                ergodicity[d, b, u_index], chain_saturation[d, b, u_index] = (
                    sampled_diagnostics(estimates, dim)
                )
                print(
                    f"D {dim:>2}  beta {beta:>4g}  u {u:>4g}  "
                    f"m {errors['magnetization']:.2e}/{floors['magnetization']:.2e}  "
                    f"C {errors['delayed']:.2e}/{floors['delayed']:.2e}  "
                    f"sigma {errors['entropy']:.2e}/{floors['entropy']:.2e}  "
                    f"rho {contraction[d, b, u_index]:.2f}  "
                    f"branches {branches[d, b, u_index]}  "
                    f"erg {ergodicity[d, b, u_index]:.2f}",
                    flush=True,
                )

        for b, beta in enumerate(fine_betas):
            for u_index, u in enumerate(fine_us):
                drive = (u / beta) * direction
                exact = exact_prediction(beta, u)
                approximate = predict(
                    drive,
                    couplings,
                    beta,
                    large_d=True,
                    max_iter=args.solve_iterations,
                    tol=args.solve_tol,
                )
                fine_residual_exact[d, b, u_index] = exact["residual"]
                fine_residual_large_d[d, b, u_index] = approximate["residual"]
                if exact["residual"] <= args.residual_tol:
                    step_fn = lambda m: mf.step(m, drive, couplings, beta)
                    fine_contraction[d, b, u_index] = fp.local_contraction(
                        step_fn,
                        exact["solution"],
                        iterations=args.power_iterations,
                    )
                if (
                    max(exact["residual"], approximate["residual"])
                    <= args.residual_tol
                ):
                    for name in OBSERVABLES:
                        large_d[name][d, b, u_index] = relative(
                            approximate[name], exact[name]
                        )
        elapsed = (time.time() - started) / 60
        print(f"finished D={dim} in {elapsed:.1f} min")
        common.save_data(payload(d + 1), args.output)


if __name__ == "__main__":
    main()
