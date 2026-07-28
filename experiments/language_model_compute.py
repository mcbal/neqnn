"""Train small depth-swept spin language models on Dostoevsky."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import common
from neqnn import SpinModelTransformerModule

COMPONENTS = ("to_qk", "to_v", "ffn", "drive_norm", "attn_temperature")


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


def corpus(path: Path, holdout: float):
    text = clean_gutenberg(path.read_text(encoding="utf-8"))
    vocabulary = sorted(set(text))
    encode = {character: index for index, character in enumerate(vocabulary)}
    tokens = torch.tensor([encode[character] for character in text], dtype=torch.long)
    split = int((1 - holdout) * len(tokens))
    digest = hashlib.sha256(text.encode()).hexdigest()
    return tokens[:split], tokens[split:], vocabulary, digest, text[:split], text[split:]


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
    starts = torch.randint(len(tokens) - length - 1, (size,), generator=generator)
    offsets = torch.arange(length + 1)
    windows = tokens[starts[:, None] + offsets]
    return windows[:, :-1].to(device), windows[:, 1:].to(device)


class LanguageModel(nn.Module):
    def __init__(self, vocab: int, dim: int, depth: int, heads: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList(
            SpinModelTransformerModule(
                dim=dim,
                num_heads=heads,
                num_steps=1,
                init="amortized",
                beta=1.0,
                causal=True,
                qk_bias=True,
                rope=True,
            )
            for _ in range(depth)
        )
        self.readout = nn.Sequential(nn.RMSNorm(dim), nn.Linear(dim, vocab, bias=False))
        with torch.no_grad():
            vectors = self.embedding.weight.view(vocab, heads, dim // heads)
            vectors.copy_(self.layers[0].radius_head * F.normalize(vectors, dim=-1))

    def forward(self, token_ids: Tensor, *, inspect: bool = False):
        x, activations, probes = self.embedding(token_ids), [], []
        for layer in self.layers:
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
    parameters = [
        parameter for parameter in parameters if parameter.grad is not None
    ]
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


def signals(model, activations, probes):
    forward, backward = [], []
    for layer, activation, probe in zip(model.layers, activations, probes):
        coupling = torch.einsum("bhij,bhjd->bhid", probe.couplings, probe.initial)
        norm = lambda value: float(value.detach().float().norm(dim=-1).mean())
        forward.append(
            {
                "residual": norm(probe.x),
                "ffn": norm(probe.drive - probe.x),
                "coupling": norm(coupling),
                "field": norm(probe.drive + coupling),
                "saturation": norm(layer.split_heads(activation)) / layer.radius_head,
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
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(args.eval_batches):
        x, y = batch(tokens, args.batch_size, args.seq_len, generator, args.device)
        logits, _, _ = model(x)
        losses.append(F.cross_entropy(logits.flatten(0, 1), y.flatten()))
    model.train()
    return float(torch.stack(losses).mean())


@torch.no_grad()
def generate(model, vocabulary, prompt: str, length: int, context: int, seed: int, device):
    model.eval()
    to_id = {character: index for index, character in enumerate(vocabulary)}
    ids = torch.tensor([to_id[c] for c in prompt if c in to_id], device=device)[None]
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(length):
        logits, _, _ = model(ids[:, -context:])
        probabilities = (logits[:, -1] / 0.8).softmax(-1)
        next_id = torch.multinomial(probabilities, 1, generator=generator)
        ids = torch.cat([ids, next_id], -1)
    model.train()
    return "".join(vocabulary[index] for index in ids[0].tolist())


def train(depth, train_tokens, valid_tokens, vocabulary, args):
    torch.manual_seed(args.seed)
    model = LanguageModel(len(vocabulary), args.dim, depth, args.heads).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.01
    )
    parameters = sum(parameter.numel() for parameter in model.parameters())
    generator = torch.Generator().manual_seed(args.seed)
    history = {"step": [], "train": [], "eval_step": [], "valid": [], "probe": []}
    started = time.time()
    print(f"\ndepth {depth}: {parameters:,} parameters on {args.device}")

    for step in range(1, args.steps + 1):
        inspect = step == 1 or step == args.steps or step % args.log_every == 0
        x, y = batch(
            train_tokens, args.batch_size, args.seq_len, generator, args.device
        )
        logits, activations, probes = model(x, inspect=inspect)
        loss = F.cross_entropy(logits.flatten(0, 1), y.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if inspect:
            forward, backward = signals(model, activations, probes)
        pre_clip = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        post_clip = gradient_norm(model.parameters())
        clip_scale = min(1.0, args.clip_grad / max(float(pre_clip), 1e-30))
        optimizer.step()
        history["step"].append(step)
        history["train"].append(float(loss.detach()))

        if inspect:
            valid = evaluate(model, valid_tokens, args, args.seed + step)
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
            sample = generate(
                model,
                vocabulary,
                "Alyosha remembers",
                args.sample_tokens,
                args.seq_len,
                args.seed + depth + step,
                args.device,
            )
            print(f"\nSample, depth {depth}:\n{sample}\n")

    return {
        "depth": depth,
        "parameters": parameters,
        "seconds": time.time() - started,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=common.ROOT.parent / "pg28054.txt")
    parser.add_argument("--output", type=Path, default=common.DATA / "language_model.pt")
    parser.add_argument("--depths", default="3,6,12")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=10.0,
        help="emergency global-norm cap; pre/post values are recorded",
    )
    parser.add_argument("--log-every", type=int, default=40)
    parser.add_argument("--eval-batches", type=int, default=3)
    parser.add_argument("--sample-tokens", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.device = torch.device(args.device)

    depths = common.numbers(args.depths, int)
    if args.dim % args.heads or (args.dim // args.heads) % 2:
        parser.error("--dim / --heads must be an even integer")
    train_tokens, valid_tokens, vocabulary, digest, train_text, valid_text = corpus(
        args.corpus, args.holdout
    )
    print(
        f"{args.corpus.name}: {len(train_tokens):,} train / "
        f"{len(valid_tokens):,} valid characters, vocab {len(vocabulary)}"
    )
    baselines = ngram_baselines(train_text, valid_text)
    print("baselines: " + "  ".join(f"{n}-gram {ce:.3f}" for n, ce in baselines.items()))
    runs = [
        train(depth, train_tokens, valid_tokens, vocabulary, args) for depth in depths
    ]
    config = {
        key: str(value) if isinstance(value, (Path, torch.device)) else value
        for key, value in vars(args).items()
    } | {"corpus_sha256": digest}
    common.save_data(
        {
            "config": config,
            "vocabulary": vocabulary,
            "baselines": baselines,
            "runs": runs,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
