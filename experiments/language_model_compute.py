"""Train small depth-swept spin language models on Dostoevsky."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import shutil
import time
import urllib.request
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import common
except ModuleNotFoundError:  # Allow reuse from importable experiment modules.
    from experiments import common
from neqnn import SpinModelTransformerModule
from neqnn import fixed_point as fp
from neqnn import mean_field as mf
from neqnn import proxies as physical_proxies
from neqnn import vmf

COMPONENTS = ("to_qk", "to_v", "ffn", "drive_norm", "attn_temperature")
INTERVENTION_HORIZONS = ("0", "1", "2", "4", "inf")
INTERVENTION_CONDITIONS = ("carrier", *INTERVENTION_HORIZONS)
INITIALIZER_CAUSAL_CONDITIONS = ("actual", "carrier", "zero", "shuffled")
MAX_FINITE_INTERVENTION_HORIZON = 4
DEFAULT_CORPUS_URL = "https://www.gutenberg.org/cache/epub/28054/pg28054.txt"


def clean_gutenberg(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    start = re.search(r"\*\*\* START OF .+? \*\*\*\n", text)
    end = re.search(r"\n\*\*\* END OF .+? \*\*\*", text)
    if start and end:
        text = text[start.end() : end.start()]
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def ensure_corpus(path: Path, url: str) -> None:
    """Download the corpus atomically when it is not already available."""
    if path.is_file():
        return
    if not url:
        raise FileNotFoundError(
            f"corpus {path} does not exist and automatic download is disabled"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.download")
    request = urllib.request.Request(
        url, headers={"User-Agent": "neqnn-language-model-experiment/1.0"}
    )
    print(f"downloading corpus from {url}", flush=True)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as destination:
                shutil.copyfileobj(response, destination)
        if temporary.stat().st_size == 0:
            raise RuntimeError("downloaded corpus is empty")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(f"saved corpus to {path}", flush=True)


def corpus(path: Path, holdout: float, url: str = DEFAULT_CORPUS_URL):
    ensure_corpus(path, url)
    text = clean_gutenberg(path.read_text(encoding="utf-8"))
    vocabulary = sorted(set(text))
    encode = {character: index for index, character in enumerate(vocabulary)}
    tokens = torch.tensor([encode[character] for character in text], dtype=torch.long)
    split = int((1 - holdout) * len(tokens))
    digest = hashlib.sha256(text.encode()).hexdigest()
    return (
        tokens[:split],
        tokens[split:],
        vocabulary,
        digest,
        text[:split],
        text[split:],
    )


def ngram_baselines(train: str, valid: str, alpha: float = 0.1) -> dict:
    vocabulary = len(set(train) | set(valid))
    scores = {}
    for order in (1, 2, 3):
        width = order - 1
        contexts = Counter(train[i - width : i] for i in range(width, len(train)))
        pairs = Counter(
            (train[i - width : i], train[i]) for i in range(width, len(train))
        )
        loss = 0.0
        for i in range(width, len(valid)):
            context = valid[i - width : i]
            probability = (pairs[context, valid[i]] + alpha) / (
                contexts[context] + alpha * vocabulary
            )
            loss -= math.log(probability)
        scores[order] = loss / (len(valid) - width)
    return scores


def batch(tokens: Tensor, size: int, length: int, generator, device):
    available = len(tokens) - length
    if available <= 0:
        raise ValueError(
            f"token split must contain more than {length} characters, got {len(tokens)}"
        )
    starts = torch.randint(available, (size,), generator=generator)
    offsets = torch.arange(length + 1)
    windows = tokens[starts[:, None] + offsets]
    return windows[:, :-1].to(device), windows[:, 1:].to(device)


class LanguageModel(nn.Module):
    def __init__(
        self,
        vocab: int,
        dim: int,
        depth: int,
        heads: int,
        input_mode: str = "field",
    ):
        super().__init__()
        if input_mode not in {"field", "magnetization"}:
            raise ValueError(
                "input_mode must be either 'field' or 'magnetization', "
                f"got {input_mode!r}"
            )
        self.embedding = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            SpinModelTransformerModule(
                dim=dim,
                num_heads=heads,
                num_steps=1,
                init="amortized",
                # Token embeddings are physical fields.  Once the first module
                # responds, subsequent layer inputs are magnetizations and may
                # use the conjugate-field carrier.
                input_mode="field" if layer_index == 0 else input_mode,
                beta=1.0,
                causal=True,
                qk_bias=True,
                rope=True,
            )
            for layer_index in range(depth)
        )
        self.readout = nn.Sequential(nn.RMSNorm(dim), nn.Linear(dim, vocab, bias=False))
        with torch.no_grad():
            vectors = self.embedding.weight.view(vocab, heads, dim // heads)
            vectors.copy_(self.layers[0].radius_head * F.normalize(vectors, dim=-1))

    def forward(
        self,
        token_ids: Tensor,
        *,
        inspect: bool = False,
        skip_layer: int | None = None,
    ):
        if skip_layer is not None and not 0 <= skip_layer < len(self.layers):
            raise ValueError(
                f"skip_layer must lie in [0, {len(self.layers)}), got {skip_layer}"
            )
        x, activations, probes = self.embedding(token_ids), [], []
        for layer_index, layer in enumerate(self.layers):
            if layer_index == skip_layer:
                continue
            result = layer(x, probe=inspect)
            x = result.magnetizations
            if inspect:
                x.retain_grad()
                for tensor in (
                    result.probe.drive,
                    result.probe.couplings,
                    result.probe.initial,
                ):
                    tensor.retain_grad()
                activations.append(x)
                probes.append(result.probe)
        return self.readout(x), activations, probes


def gradient_rms(parameters) -> float:
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    count = sum(parameter.numel() for parameter in parameters)
    squared = sum(
        float(parameter.grad.float().square().sum()) for parameter in parameters
    )
    return (squared / count) ** 0.5 if count else float("nan")


def gradient_norm(parameters) -> float:
    squared = sum(
        float(parameter.grad.float().square().sum())
        for parameter in parameters
        if parameter.grad is not None
    )
    return squared**0.5


def distribution(values: Tensor) -> dict[str, float]:
    """Small detached summary for per-site physical diagnostics."""
    flattened = values.detach().float().reshape(-1)
    quantiles = torch.quantile(flattened, flattened.new_tensor([0.05, 0.5, 0.95]))
    return {
        "mean": float(flattened.mean()),
        "p05": float(quantiles[0]),
        "p50": float(quantiles[1]),
        "p95": float(quantiles[2]),
        "max": float(flattened.max()),
    }


def signals(model, activations, probes):
    forward, backward = [], []
    for layer, activation, probe in zip(model.layers, activations, probes):
        carrier = (
            probe.x
            if layer.input_mode == "field"
            else vmf.inverse_response_large_d(probe.x, layer.beta)
        )
        coupling = torch.einsum("bhij,bhjd->bhid", probe.couplings, probe.initial)
        norm = lambda value: float(value.detach().float().norm(dim=-1).mean())
        with torch.no_grad():
            settled = layer.split_heads(activation)
            update_field = probe.drive + coupling
            initializer_field = vmf.inverse_response_large_d(probe.initial, layer.beta)
            settled_field = vmf.inverse_response_large_d(settled, layer.beta)
            # Compare exactly the two endpoint magnetizations, expressed in
            # the conjugate-field coordinates that parameterize their vMF laws:
            # h_0 = phi^-1(m_0), h_1 = phi^-1(m_1).
            relaxation_mismatch = physical_proxies.mismatch(
                initializer_field, settled_field, layer.beta
            )
            increment = update_field - carrier
            ffn_increment = probe.drive - carrier
            carrier_norm = carrier.norm(dim=-1, keepdim=True)
            carrier_direction = carrier / carrier_norm.clamp_min(
                torch.finfo(carrier.dtype).tiny
            )

            def components(value):
                radial = (value * carrier_direction).sum(-1)
                transverse = (value.pow(2).sum(-1) - radial.pow(2)).clamp_min(0).sqrt()
                return value.norm(dim=-1), radial, transverse

            increment_norm, increment_radial, increment_transverse = components(
                increment
            )
            ffn_norm, ffn_radial, ffn_transverse = components(ffn_increment)
            coupling_norm, coupling_radial, coupling_transverse = components(coupling)
            update_norm = update_field.norm(dim=-1, keepdim=True)
            cosine = (carrier * update_field).sum(-1) / (
                carrier_norm * update_norm
            ).squeeze(-1).clamp_min(torch.finfo(carrier.dtype).tiny)
            direction_change = cosine.clamp(-1, 1).acos()
            relative_scale = carrier_norm.squeeze(-1).clamp_min(
                torch.finfo(carrier.dtype).tiny
            )

            saturation = settled.norm(dim=-1) / layer.radius_head
            effective_u = layer.beta * update_norm.squeeze(-1) / layer.radius_head
            increment_u = layer.beta * increment_norm / layer.radius_head
            radial_u = layer.beta * increment_radial / layer.radius_head
            transverse_u = layer.beta * increment_transverse / layer.radius_head
            ffn_u = layer.beta * ffn_norm / layer.radius_head
            ffn_radial_u = layer.beta * ffn_radial / layer.radius_head
            ffn_transverse_u = layer.beta * ffn_transverse / layer.radius_head
            coupling_u = layer.beta * coupling_norm / layer.radius_head
            coupling_radial_u = layer.beta * coupling_radial / layer.radius_head
            coupling_transverse_u = layer.beta * coupling_transverse / layer.radius_head
            stiffness = vmf.gamma(update_field, layer.beta).squeeze(-1)
            susceptibility_tangential = layer.beta / (1 + stiffness)
            susceptibility_radial = layer.beta / (stiffness * (1 + stiffness))

            saturation_stats = distribution(saturation)
            effective_u_stats = distribution(effective_u)
            radial_susceptibility_stats = distribution(susceptibility_radial)
            direction_change_stats = distribution(direction_change)

            step_fn = lambda magnetizations: mf.step_large_d(
                magnetizations,
                probe.drive,
                probe.couplings,
                layer.beta,
            )
            fixed_point = fp.anderson(
                step_fn,
                settled,
                max_iter=layer.max_iter,
                tol=layer.tol,
            )
            if fixed_point.converged:
                steady_field = mf.effective_field(
                    fixed_point.solution, probe.drive, probe.couplings
                )
                covariance_traces = mf.covariance_traces_large_d(
                    steady_field, steady_field, layer.beta
                )
                entropy = float(
                    physical_proxies.housekeeping_entropy_production(
                        probe.couplings, covariance_traces, layer.beta
                    )
                    .float()
                    .mean()
                )
            else:
                entropy = math.nan
        forward.append(
            {
                "input": norm(probe.x),
                "carrier": norm(carrier),
                # Kept as an artifact-schema alias for older plotting code.
                "residual": norm(carrier),
                "ffn": norm(probe.drive - carrier),
                "coupling": norm(coupling),
                "field": norm(update_field),
                "saturation": saturation_stats["mean"],
                "saturation_p50": saturation_stats["p50"],
                "saturation_p95": saturation_stats["p95"],
                "saturation_max": saturation_stats["max"],
                "effective_u": effective_u_stats["mean"],
                "effective_u_p50": effective_u_stats["p50"],
                "effective_u_p95": effective_u_stats["p95"],
                "effective_u_max": effective_u_stats["max"],
                "increment_u": float(increment_u.mean()),
                "increment_radial_u": float(radial_u.mean()),
                "increment_radial_abs_u": float(radial_u.abs().mean()),
                "increment_transverse_u": float(transverse_u.mean()),
                "increment_relative": float((increment_norm / relative_scale).mean()),
                "increment_radial_relative": float(
                    (increment_radial / relative_scale).mean()
                ),
                "increment_transverse_relative": float(
                    (increment_transverse / relative_scale).mean()
                ),
                "direction_change": direction_change_stats["mean"],
                "direction_change_p50": direction_change_stats["p50"],
                "direction_change_p95": direction_change_stats["p95"],
                "ffn_increment_u": float(ffn_u.mean()),
                "ffn_increment_radial_u": float(ffn_radial_u.mean()),
                "ffn_increment_transverse_u": float(ffn_transverse_u.mean()),
                "coupling_increment_u": float(coupling_u.mean()),
                "coupling_increment_radial_u": float(coupling_radial_u.mean()),
                "coupling_increment_transverse_u": float(coupling_transverse_u.mean()),
                "susceptibility_tangential": float(susceptibility_tangential.mean()),
                "susceptibility_radial": radial_susceptibility_stats["mean"],
                "susceptibility_radial_p05": radial_susceptibility_stats["p05"],
                "entropy": entropy,
                "entropy_fixed_point_residual": fixed_point.residual,
                "entropy_fixed_point_converged": fixed_point.converged,
                # One-step distributional change from the amortized value guess
                # m_0 to the module output m_1. Unlike the fixed-point mismatch
                # used by Relaxation, this needs no additional solve.
                "relaxation_mismatch": float(relaxation_mismatch.float().mean()),
            }
        )
        row = {
            "activation": gradient_rms([activation]),
            "field": gradient_rms([probe.drive]),
            "couplings": gradient_rms([probe.couplings]),
            "initial": gradient_rms([probe.initial]),
        }
        for component in COMPONENTS:
            row[component] = gradient_rms(
                parameter
                for name, parameter in layer.named_parameters()
                if name.startswith(component)
            )
        backward.append(row)
    return forward, backward


@torch.no_grad()
def evaluate(model, tokens, args, seed):
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(args.eval_batches):
        x, y = batch(tokens, args.batch_size, args.seq_len, generator, args.device)
        logits, _, _ = model(x)
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()))
    model.train(was_training)
    return float(torch.stack(losses).mean())


@torch.no_grad()
def evaluate_layer_ablation(model, tokens, args, seed):
    """Paired final-checkpoint effect of omitting each residual update.

    Layer 1 is the field-to-magnetization adapter and is intentionally retained:
    skipping it would change coordinates rather than merely remove one residual
    update. All remaining variants use exactly the same held-out token batches.
    """
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    batches = [
        batch(tokens, args.batch_size, args.seq_len, generator, args.device)
        for _ in range(args.ablation_batches)
    ]
    full = []
    baseline_losses = []
    for x, y in batches:
        logits, _, _ = model(x)
        full.append(logits)
        baseline_losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()))
    baseline = torch.stack(baseline_losses).mean()

    rows = []
    for layer_index in range(1, len(model.layers)):
        skipped_losses, divergences = [], []
        for (x, y), full_logits in zip(batches, full):
            skipped_logits, _, _ = model(x, skip_layer=layer_index)
            skipped_losses.append(
                F.cross_entropy(skipped_logits.flatten(0, 1), y.flatten())
            )
            full_log_prob = full_logits.log_softmax(-1)
            skipped_log_prob = skipped_logits.log_softmax(-1)
            divergences.append(
                F.kl_div(
                    skipped_log_prob,
                    full_log_prob,
                    reduction="none",
                    log_target=True,
                )
                .sum(-1)
                .mean()
            )
        skipped = torch.stack(skipped_losses).mean()
        rows.append(
            {
                "layer": layer_index + 1,
                "loss": float(skipped),
                "loss_delta": float(skipped - baseline),
                "kl_from_full": float(torch.stack(divergences).mean()),
            }
        )
    model.train(was_training)
    return {"baseline_loss": float(baseline), "layers": rows}


def _intervention_state(layer, trace, horizon: str):
    if horizon == "inf":
        return trace.fixed_point.solution if trace.fixed_point.converged else None
    return trace.magnetizations[int(horizon)]


def _carrier_state(layer, probe):
    """Physical pass-through state in the module's declared input coordinate.

    A field input must first respond through ``phi`` to become a magnetization;
    a magnetization input is already a physical state. This excludes the FFN,
    attention interaction, and amortized value initializer.
    """
    if layer.input_mode == "field":
        return vmf.response_large_d(probe.x, layer.beta)
    return probe.x


def _state_mismatch(trace, horizon: str) -> tuple[float, float]:
    if not trace.fixed_point.converged:
        return math.nan, math.nan
    if horizon == "inf":
        return 0.0, 0.0
    index = int(horizon)
    state = trace.magnetizations[index]
    rms = (state - trace.fixed_point.solution).square().sum(-1).mean(-1).sqrt().mean()
    return float(rms), float(trace.mismatch[index].mean())


def _counterfactual_mismatch(layer, state, trace, probe) -> tuple[float, float]:
    if not trace.fixed_point.converged:
        return math.nan, math.nan
    rms = (state - trace.fixed_point.solution).square().sum(-1).mean(-1).sqrt().mean()
    field = mf.effective_field(state, probe.drive, probe.couplings)
    steady_field = mf.effective_field(
        trace.fixed_point.solution, probe.drive, probe.couplings
    )
    field_kl = physical_proxies.mismatch(field, steady_field, layer.beta).mean()
    return float(rms), float(field_kl)


def _initializer_start(layer, probe, condition: str, permutation):
    """Choose a physical start while leaving the frozen protocol untouched.

    The shuffled control permutes whole examples, not sites, so a causal model
    never receives a future-position value at an earlier sequence position.
    """
    if condition == "actual":
        return probe.initial
    if condition == "carrier":
        return _carrier_state(layer, probe)
    if condition == "zero":
        return torch.zeros_like(probe.initial)
    if condition == "shuffled":
        return probe.initial.index_select(0, permutation)
    raise ValueError(f"unknown initializer causal condition {condition!r}")


def _one_step_from_start(layer, probe, start):
    return mf.step_large_d(start, probe.drive, probe.couplings, layer.beta)


def _joint_horizon_output(layer, x, horizon: str):
    """Advance one layer without paying for an unused finite-K fixed point."""
    ordinary = layer(x, probe=True)
    if horizon == "carrier":
        return layer.merge_heads(_carrier_state(layer, ordinary.probe)), True
    if horizon == "0":
        return layer.merge_heads(ordinary.probe.initial), True
    if horizon == "1":
        return ordinary.magnetizations, True
    if horizon == "inf":
        trace = layer.relaxation(x, num_steps=1)
        if not trace.fixed_point.converged:
            return None, False
        return layer.merge_heads(trace.fixed_point.solution), True
    trajectory = mf.relax_large_d(
        ordinary.probe.initial,
        ordinary.probe.drive,
        ordinary.probe.couplings,
        layer.beta,
        num_steps=int(horizon),
    )
    return layer.merge_heads(trajectory[-1]), True


def _mean(values) -> float:
    return float(torch.tensor(values, dtype=torch.float64).mean())


@torch.no_grad()
def evaluate_relaxation_interventions(
    model,
    tokens,
    args,
    seed: int,
    *,
    all_layers: bool,
):
    """Apply frozen-protocol K changes locally, then run the suffix at K=1.

    Intermediate-layer states are never decoded directly. An altered state is
    fed through every later trained module in the ordinary way, so its cross
    entropy is an end-to-end causal effect. The optional joint profile is a
    cumulative systems intervention and is kept separate from the layer-local
    attribution.
    """
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    batches = [
        batch(tokens, args.batch_size, args.seq_len, generator, args.device)
        for _ in range(args.intervention_batches)
    ]
    shuffle_generator = torch.Generator().manual_seed(seed + 1)
    permutations = []
    identity = torch.arange(args.batch_size)
    for _ in batches:
        permutation = torch.randperm(args.batch_size, generator=shuffle_generator)
        if torch.equal(permutation, identity):
            permutation = permutation.roll(1)
        permutations.append(permutation.to(args.device))
    selected_layers = (
        list(range(len(model.layers))) if all_layers else [len(model.layers) - 1]
    )
    records = {
        layer_index: {
            "loss_batches": {condition: [] for condition in INTERVENTION_CONDITIONS},
            "rms_batches": {condition: [] for condition in INTERVENTION_CONDITIONS},
            "field_kl_batches": {
                condition: [] for condition in INTERVENTION_CONDITIONS
            },
            "initializer_loss_batches": {
                condition: [] for condition in INITIALIZER_CAUSAL_CONDITIONS
            },
            "initializer_rms_batches": {
                condition: [] for condition in INITIALIZER_CAUSAL_CONDITIONS
            },
            "initializer_field_kl_batches": {
                condition: [] for condition in INITIALIZER_CAUSAL_CONDITIONS
            },
            "converged": [],
            "fixed_point_residual": [],
        }
        for layer_index in selected_layers
    }
    baseline_batches = []
    joint_batches = (
        {condition: [] for condition in INTERVENTION_CONDITIONS} if all_layers else None
    )

    try:
        for batch_index, (token_ids, targets) in enumerate(batches):
            x = model.embedding(token_ids)
            layer_inputs = []
            for layer in model.layers:
                layer_inputs.append(x)
                x = layer(x).magnetizations
            baseline_logits = model.readout(x)
            baseline_loss = F.cross_entropy(
                baseline_logits.flatten(0, 1), targets.flatten()
            )
            baseline_batches.append(float(baseline_loss))

            for layer_index in selected_layers:
                layer = model.layers[layer_index]
                ordinary = layer(layer_inputs[layer_index], probe=True)
                trace = layer.relaxation(
                    layer_inputs[layer_index],
                    num_steps=MAX_FINITE_INTERVENTION_HORIZON,
                )
                row = records[layer_index]
                row["converged"].append(trace.fixed_point.converged)
                row["fixed_point_residual"].append(trace.fixed_point.residual)
                for condition in INTERVENTION_CONDITIONS:
                    state = (
                        _carrier_state(layer, ordinary.probe)
                        if condition == "carrier"
                        else _intervention_state(layer, trace, condition)
                    )
                    if state is None:
                        loss = math.nan
                    else:
                        altered = layer.merge_heads(state)
                        for suffix in model.layers[layer_index + 1 :]:
                            altered = suffix(altered).magnetizations
                        logits = model.readout(altered)
                        loss = float(
                            F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                        )
                    rms, field_kl = (
                        _counterfactual_mismatch(layer, state, trace, ordinary.probe)
                        if condition == "carrier"
                        else _state_mismatch(trace, condition)
                    )
                    row["loss_batches"][condition].append(loss)
                    row["rms_batches"][condition].append(rms)
                    row["field_kl_batches"][condition].append(field_kl)

                for condition in INITIALIZER_CAUSAL_CONDITIONS:
                    start = _initializer_start(
                        layer,
                        ordinary.probe,
                        condition,
                        permutations[batch_index],
                    )
                    state = _one_step_from_start(layer, ordinary.probe, start)
                    altered = layer.merge_heads(state)
                    for suffix in model.layers[layer_index + 1 :]:
                        altered = suffix(altered).magnetizations
                    logits = model.readout(altered)
                    loss = float(
                        F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                    )
                    rms, field_kl = _counterfactual_mismatch(
                        layer, state, trace, ordinary.probe
                    )
                    row["initializer_loss_batches"][condition].append(loss)
                    row["initializer_rms_batches"][condition].append(rms)
                    row["initializer_field_kl_batches"][condition].append(field_kl)

            if joint_batches is not None:
                for condition in INTERVENTION_CONDITIONS:
                    altered = model.embedding(token_ids)
                    converged = True
                    for layer in model.layers:
                        altered, converged = _joint_horizon_output(
                            layer, altered, condition
                        )
                        if not converged:
                            converged = False
                            break
                    if converged:
                        logits = model.readout(altered)
                        loss = float(
                            F.cross_entropy(logits.flatten(0, 1), targets.flatten())
                        )
                    else:
                        loss = math.nan
                    joint_batches[condition].append(loss)
    finally:
        model.train(was_training)

    baseline = _mean(baseline_batches)
    layers = []
    for layer_index in selected_layers:
        row = records[layer_index]
        loss = {
            condition: _mean(row["loss_batches"][condition])
            for condition in INTERVENTION_CONDITIONS
        }
        loss_delta_batches = {
            condition: [
                altered - ordinary
                for altered, ordinary in zip(
                    row["loss_batches"][condition], baseline_batches
                )
            ]
            for condition in INTERVENTION_CONDITIONS
        }
        initializer_loss = {
            condition: _mean(row["initializer_loss_batches"][condition])
            for condition in INITIALIZER_CAUSAL_CONDITIONS
        }
        initializer_loss_delta_batches = {
            condition: [
                altered - ordinary
                for altered, ordinary in zip(
                    row["initializer_loss_batches"][condition], baseline_batches
                )
            ]
            for condition in INITIALIZER_CAUSAL_CONDITIONS
        }
        layers.append(
            {
                "layer": layer_index + 1,
                "loss": loss,
                "loss_delta": {
                    horizon: _mean(values)
                    for horizon, values in loss_delta_batches.items()
                },
                "loss_delta_batches": loss_delta_batches,
                "rms": {
                    condition: _mean(row["rms_batches"][condition])
                    for condition in INTERVENTION_CONDITIONS
                },
                "field_kl": {
                    condition: _mean(row["field_kl_batches"][condition])
                    for condition in INTERVENTION_CONDITIONS
                },
                "initializer_causal": {
                    "loss": initializer_loss,
                    "loss_delta": {
                        condition: _mean(values)
                        for condition, values in initializer_loss_delta_batches.items()
                    },
                    "loss_delta_batches": initializer_loss_delta_batches,
                    "rms": {
                        condition: _mean(row["initializer_rms_batches"][condition])
                        for condition in INITIALIZER_CAUSAL_CONDITIONS
                    },
                    "field_kl": {
                        condition: _mean(row["initializer_field_kl_batches"][condition])
                        for condition in INITIALIZER_CAUSAL_CONDITIONS
                    },
                },
                "fixed_point_converged_fraction": sum(row["converged"])
                / len(row["converged"]),
                "fixed_point_residual_batches": row["fixed_point_residual"],
                "fixed_point_residual_max": max(row["fixed_point_residual"]),
            }
        )

    joint = None
    if joint_batches is not None:
        joint = {
            "loss": {
                condition: _mean(joint_batches[condition])
                for condition in INTERVENTION_CONDITIONS
            },
            "loss_delta": {
                condition: _mean(
                    [
                        altered - ordinary
                        for altered, ordinary in zip(
                            joint_batches[condition], baseline_batches
                        )
                    ]
                )
                for condition in INTERVENTION_CONDITIONS
            },
            "loss_delta_batches": {
                condition: [
                    altered - ordinary
                    for altered, ordinary in zip(
                        joint_batches[condition], baseline_batches
                    )
                ]
                for condition in INTERVENTION_CONDITIONS
            },
        }
    return {"baseline_loss": baseline, "layers": layers, "joint": joint}


@torch.no_grad()
def generate(
    model,
    vocabulary,
    prompt: str,
    length: int,
    context: int,
    seed: int,
    device,
    temperature: float = 0.8,
    top_k: int = 0,
):
    if not prompt:
        raise ValueError("generation prompt must not be empty")
    to_id = {character: index for index, character in enumerate(vocabulary)}
    unknown = sorted(set(prompt) - set(to_id))
    if unknown:
        raise ValueError(
            f"prompt contains characters outside the corpus vocabulary: {unknown!r}"
        )
    was_training = model.training
    model.eval()
    ids = torch.tensor(
        [to_id[character] for character in prompt], dtype=torch.long, device=device
    )[None]
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(length):
        logits, _, _ = model(ids[:, -context:])
        next_logits = logits[:, -1] / temperature
        if 0 < top_k < len(vocabulary):
            cutoff = next_logits.topk(top_k, dim=-1).values[:, -1, None]
            next_logits = next_logits.masked_fill(next_logits < cutoff, -torch.inf)
        probabilities = next_logits.softmax(-1)
        next_id = torch.multinomial(probabilities, 1, generator=generator)
        ids = torch.cat([ids, next_id], -1)
    model.train(was_training)
    return "".join(vocabulary[index] for index in ids[0].tolist())


def train(depth, train_tokens, valid_tokens, vocabulary, args):
    torch.manual_seed(args.seed)
    model = LanguageModel(
        len(vocabulary), args.dim, depth, args.heads, args.input_mode
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    generator = torch.Generator().manual_seed(args.seed)
    history = {
        "step": [],
        "train": [],
        "eval_step": [],
        "valid": [],
        "probe": [],
        "samples": [],
        "layer_ablation": None,
        "relaxation_interventions": [],
    }
    intervention_enabled = depth in args.intervention_depths
    intervention_steps = set(args.intervention_steps) if intervention_enabled else set()
    intervention_checkpoints = {}
    started = time.time()
    print(f"\ndepth {depth}: {parameters:,} parameters on {args.device}")

    validation_seed = args.seed + 20_000
    sample_seed = args.seed + 10_000 + depth
    initial_valid = evaluate(model, valid_tokens, args, validation_seed)
    history["eval_step"].append(0)
    history["valid"].append(initial_valid)
    initial_sample = generate(
        model,
        vocabulary,
        args.prompt,
        args.sample_tokens,
        args.seq_len,
        sample_seed,
        args.device,
        args.temperature,
        args.top_k,
    )
    history["samples"].append({"step": 0, "text": initial_sample})
    print(
        f"initial valid {initial_valid:.3f}\n\n"
        f"Sample, depth {depth}, step 0:\n{initial_sample}\n"
    )
    if 0 in intervention_steps:
        measured = evaluate_relaxation_interventions(
            model,
            valid_tokens,
            args,
            args.seed + 30_000,
            all_layers=False,
        )
        history["relaxation_interventions"].append({"step": 0} | measured)
        if args.save_intervention_checkpoints:
            intervention_checkpoints["0"] = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    for step in range(1, args.steps + 1):
        inspect = step == 1 or step == args.steps or step % args.log_every == 0
        x, y = batch(
            train_tokens, args.batch_size, args.seq_len, generator, args.device
        )
        logits, activations, probes = model(x, inspect=inspect)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at depth {depth}, step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if inspect:
            forward, backward = signals(model, activations, probes)
        pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        post_clip = gradient_norm(model.parameters())
        if not math.isfinite(float(pre_clip)) or not math.isfinite(post_clip):
            raise FloatingPointError(
                f"non-finite gradient at depth {depth}, step {step}"
            )
        clip_scale = min(1.0, args.clip_grad / max(float(pre_clip), 1e-30))
        optimizer.step()
        history["step"].append(step)
        history["train"].append(float(loss.detach()))

        if step in intervention_steps:
            measured = evaluate_relaxation_interventions(
                model,
                valid_tokens,
                args,
                args.seed + 30_000,
                all_layers=step == args.steps,
            )
            history["relaxation_interventions"].append({"step": step} | measured)
            if args.save_intervention_checkpoints:
                intervention_checkpoints[str(step)] = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
            final_layer = measured["layers"][-1]
            print(
                "\nrelaxation intervention "
                f"at depth {depth}, optimizer step {step}: "
                + "  ".join(
                    f"{condition} {final_layer['loss_delta'][condition]:+.3f}"
                    for condition in INTERVENTION_CONDITIONS
                )
            )

        if inspect:
            valid = evaluate(model, valid_tokens, args, validation_seed)
            history["eval_step"].append(step)
            history["valid"].append(valid)
            history["probe"].append(
                {
                    "step": step,
                    "forward": forward,
                    "backward": backward,
                    "grad_norm": float(pre_clip),
                    "grad_norm_post": post_clip,
                    "clip_scale": clip_scale,
                }
            )
            print(
                f"\ndepth {depth:>2}  step {step:>4}/{args.steps}  "
                f"train {loss:.3f}  valid {valid:.3f}  "
                f"grad {pre_clip:.2e}->{post_clip:.2e} (x{clip_scale:.2f})"
            )
            print("raw gradient RMS by layer (before clipping)")
            print(
                "layer    |h|  |m|/R    dL/dm    dL/dh     dL/dJ   dL/dm0  |"
                "       QK        V      FFN     norm     temp"
            )
            for index, (field, gradients) in enumerate(zip(forward, backward)):
                print(
                    f"{index + 1:>5}  {field['field']:>5.2f}  "
                    f"{field['saturation']:>6.3f}  "
                    f"{gradients['activation']:>8.1e}  "
                    f"{gradients['field']:>8.1e}  "
                    f"{gradients['couplings']:>8.1e}  "
                    f"{gradients['initial']:>8.1e}  |"
                    f" {gradients['to_qk']:>8.1e}"
                    f" {gradients['to_v']:>8.1e}"
                    f" {gradients['ffn']:>8.1e}"
                    f" {gradients['drive_norm']:>8.1e}"
                    f" {gradients['attn_temperature']:>8.1e}"
                )
        sample_now = step == args.steps or step % args.sample_every == 0
        if sample_now:
            sample = generate(
                model,
                vocabulary,
                args.prompt,
                args.sample_tokens,
                args.seq_len,
                sample_seed,
                args.device,
                args.temperature,
                args.top_k,
            )
            history["samples"].append({"step": step, "text": sample})
            print(f"\nSample, depth {depth}, step {step}:\n{sample}\n")

    history["layer_ablation"] = evaluate_layer_ablation(
        model, valid_tokens, args, validation_seed
    )
    ablation_rows = history["layer_ablation"]["layers"]
    if ablation_rows:
        strongest = max(ablation_rows, key=lambda row: row["kl_from_full"])
        weakest = min(ablation_rows, key=lambda row: row["kl_from_full"])
        print(
            "paired layer skips: "
            f"largest output effect layer {strongest['layer']} "
            f"(KL {strongest['kl_from_full']:.3e}, "
            f"delta CE {strongest['loss_delta']:+.3e}); "
            f"smallest layer {weakest['layer']} "
            f"(KL {weakest['kl_from_full']:.3e}, "
            f"delta CE {weakest['loss_delta']:+.3e})"
        )

    return {
        "depth": depth,
        "parameters": parameters,
        "seconds": time.time() - started,
        "history": history,
        "intervention_checkpoints": intervention_checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus", type=Path, default=common.ROOT / "data" / "pg28054.txt"
    )
    parser.add_argument(
        "--corpus-url",
        default=DEFAULT_CORPUS_URL,
        help="download source used only when --corpus does not exist; pass an empty value to disable",
    )
    parser.add_argument(
        "--output", type=Path, default=common.DATA / "language_model.pt"
    )
    parser.add_argument("--depths", default="3,6,12,24")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--input-mode",
        choices=("field", "magnetization"),
        default="magnetization",
        help=(
            "inter-layer coordinate: 'field' preserves the legacy stack; "
            "'magnetization' keeps layer 1 as a field adapter and uses "
            "conjugate-field transport thereafter"
        ),
    )
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=10.0,
        help="emergency global-norm cap; pre/post values are recorded",
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument(
        "--ablation-batches",
        type=int,
        default=32,
        help="paired held-out batches for final per-layer skip diagnostics",
    )
    parser.add_argument(
        "--intervention-depths",
        default="6",
        help="model depths receiving K-intervention checkpoints, or 'none'",
    )
    parser.add_argument(
        "--intervention-steps",
        default="0,1,50,500,1500,3000",
        help="post-update checkpoints; the final optimizer step is always added",
    )
    parser.add_argument(
        "--intervention-batches",
        type=int,
        default=32,
        help="fixed paired held-out batches used by every K intervention",
    )
    parser.add_argument(
        "--save-intervention-checkpoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="persist model states at intervention checkpoints for offline expansion",
    )
    parser.add_argument("--sample-tokens", type=int, default=80)
    parser.add_argument("--sample-every", type=int, default=40)
    parser.add_argument("--prompt", default="Aloysha remembers")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.device = torch.device(args.device)

    try:
        depths = list(dict.fromkeys(common.numbers(args.depths, int)))
        args.intervention_depths = (
            []
            if args.intervention_depths.strip().lower() == "none"
            else list(dict.fromkeys(common.numbers(args.intervention_depths, int)))
        )
        requested_intervention_steps = common.numbers(args.intervention_steps, int)
    except ValueError as error:
        parser.error(str(error))
    if not depths or min(depths) < 1:
        parser.error("--depths must contain positive integers")
    if args.dim % args.heads or (args.dim // args.heads) % 2:
        parser.error("--dim / --heads must be an even integer")
    if args.steps < 1 or args.batch_size < 1 or args.seq_len < 1:
        parser.error("--steps, --batch-size, and --seq-len must be positive")
    if args.intervention_depths and min(args.intervention_depths) < 1:
        parser.error("--intervention-depths must contain positive integers")
    if min(requested_intervention_steps) < 0:
        parser.error("--intervention-steps must be non-negative")
    args.intervention_steps = sorted(
        {step for step in requested_intervention_steps if step <= args.steps}
        | {args.steps}
    )
    if not 0 < args.holdout < 1:
        parser.error("--holdout must lie strictly between 0 and 1")
    if args.lr <= 0 or args.clip_grad <= 0:
        parser.error("--lr and --clip-grad must be positive")
    if (
        min(
            args.log_every,
            args.eval_batches,
            args.ablation_batches,
            args.intervention_batches,
            args.sample_tokens,
            args.sample_every,
        )
        < 1
    ):
        parser.error("logging, evaluation, and sampling counts must be positive")
    if args.temperature <= 0 or args.top_k < 0:
        parser.error("--temperature must be positive and --top-k non-negative")
    train_tokens, valid_tokens, vocabulary, digest, train_text, valid_text = corpus(
        args.corpus, args.holdout, args.corpus_url
    )
    unknown = sorted(set(args.prompt) - set(vocabulary))
    if not args.prompt:
        parser.error("--prompt must not be empty")
    if unknown:
        parser.error(f"--prompt contains characters outside the corpus: {unknown!r}")
    if min(len(train_tokens), len(valid_tokens)) <= args.seq_len:
        parser.error("both corpus splits must contain more characters than --seq-len")
    print(
        f"{args.corpus.name}: {len(train_tokens):,} train / "
        f"{len(valid_tokens):,} valid characters, vocab {len(vocabulary)}"
    )
    baselines = ngram_baselines(train_text, valid_text)
    print(
        "baselines: " + "  ".join(f"{n}-gram {ce:.3f}" for n, ce in baselines.items())
    )
    config = {
        key: str(value) if isinstance(value, (Path, torch.device)) else value
        for key, value in vars(args).items()
    } | {"corpus_sha256": digest}
    runs = []
    for depth in depths:
        runs.append(train(depth, train_tokens, valid_tokens, vocabulary, args))
        common.save_data(
            {
                "schema_version": 7,
                "config": config,
                "vocabulary": vocabulary,
                "baselines": baselines,
                "runs": runs,
                "complete": len(runs) == len(depths),
            },
            args.output,
        )


if __name__ == "__main__":
    main()
