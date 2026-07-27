"""Experiment 02 -- is a stack of these things an ordinary trainable network?

Experiment 01 asked whether the physics is right.  This one asks whether the
physics is *usable*: put a plain character-level next-token objective on a stack
of spin-model modules and check that the forward signal does not blow up or
collapse with depth, that the backward signal reaches every parameter, and that
the loss goes somewhere an n-gram baseline does not.  Nothing here is meant to
be a good language model.  It is meant to be unremarkable, which is the point --
the framework should not need special handling to train.

This is the transformer-like quadrant of the design table: finite horizon K=1,
amortized initialization, no carried state.  Layers compose by feeding one
layer's magnetizations in as the next layer's drive, with no residual or
projection in between, so what is under test is the composition itself.

Four settings are corrections that were paid for in round 1 and are not free
parameters:

- **module-default attention temperature.**  The probe operating point of the
  earlier work (8R) saturates the softmax and kills attention gradients.
- **``qk_bias=True``.**  Without it a content-independent (positional) component
  of the logit is unrepresentable and single-layer induction never forms.
- **random windows over a large corpus, never a small repeated set.**  Epoch
  training on a small set memorizes; the signature is train CE below the unigram
  entropy with held-out CE rising.
- **``dim_head`` is the D that controls the large-D approximation**, not ``dim``.
  The default is 64 heads-wide, where experiment 01 puts the large-D error near
  1%.  Raising ``--heads`` at fixed ``--dim`` makes the physics worse, not just
  the model narrower.

Run with ``uv run python experiments/02_text_trainability.py``; ``--steps 30``
for a wiring check.  The whole run -- history, baselines, and the probe
couplings -- is cached under ``data/02/<run-tag>.pt``, so tuning a figure never
costs a training run: ``--plot-only`` (with the same hyperparameter flags)
redraws from cache, ``--refresh`` retrains.  The cached file is also what
experiment 01 reads with ``--real`` to test the entropy-production alignment on
trained rather than synthetic couplings.
"""

from __future__ import annotations

import argparse
import math
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import Tensor, nn

import common
from neqnn.modules import SpinModelTransformerModule

CORPUS = Path(__file__).resolve().parents[1] / "input.txt"


class SpinLanguageModel(nn.Module):
    """Embeddings -> a stack of spin modules -> vocabulary logits."""

    def __init__(
        self,
        vocab_size: int,
        *,
        dim: int,
        depth: int,
        num_heads: int,
        beta: float,
        qk_bias: bool = True,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList(
            SpinModelTransformerModule(
                dim=dim,
                num_heads=num_heads,
                num_steps=1,
                init="amortized",
                beta=beta,
                causal=True,
                qk_bias=qk_bias,
                rope=True,
            )
            for _ in range(depth)
        )
        self.to_logits = nn.Sequential(
            nn.RMSNorm(dim), nn.Linear(dim, vocab_size, bias=False)
        )

        # Start every embedding head exactly on its sphere of radius R.  The
        # drive enters the field directly as a residual, so handing the first
        # module a unit-norm stream alongside an R-norm normalized one would put
        # the two terms on different scales before training even starts.
        with torch.no_grad():
            heads = self.token_embedding.weight.view(
                vocab_size, num_heads, dim // num_heads
            )
            heads.copy_(self.layers[0].radius_head * F.normalize(heads, dim=-1))
        nn.init.normal_(self.to_logits[1].weight, std=dim**-0.5)

    def forward(self, token_ids: Tensor, *, retain: bool = False):
        x = self.token_embedding(token_ids)
        activations = []
        for layer in self.layers:
            x = layer(x).magnetizations
            if retain:
                x.retain_grad()
            activations.append(x)
        return self.to_logits(x), activations


#
# Data
#


def load_corpus(path: Path, *, holdout: float = 0.1):
    text = path.read_text(encoding="utf-8")
    vocabulary = sorted(set(text))
    to_id = {character: index for index, character in enumerate(vocabulary)}
    tokens = torch.tensor([to_id[c] for c in text], dtype=torch.long)
    split = int(len(tokens) * (1 - holdout))
    return tokens[:split], tokens[split:], vocabulary, text[:split], text[split:]


def sample_batch(tokens: Tensor, *, batch_size: int, seq_len: int, generator):
    starts = torch.randint(
        len(tokens) - seq_len - 1, (batch_size,), generator=generator
    )
    windows = torch.stack([tokens[s : s + seq_len + 1] for s in starts])
    return windows[:, :-1], windows[:, 1:]


def ngram_baselines(train_text: str, valid_text: str, *, orders=(1, 2, 3), alpha=0.1):
    """Smoothed n-gram CE on the held-out split, the bar the model has to clear.

    Without these the loss curve has no scale: a number going down is not
    evidence of learning until it passes what counting can do.
    """
    vocab_size = len(set(train_text) | set(valid_text))
    baselines = {}
    for order in orders:
        context_size = order - 1
        contexts = Counter(
            train_text[i - context_size : i]
            for i in range(context_size, len(train_text))
        )
        joint = Counter(
            (train_text[i - context_size : i], train_text[i])
            for i in range(context_size, len(train_text))
        )
        total = 0.0
        for i in range(context_size, len(valid_text)):
            context = valid_text[i - context_size : i]
            probability = (joint[(context, valid_text[i])] + alpha) / (
                contexts[context] + alpha * vocab_size
            )
            total -= math.log(probability)
        baselines[order] = total / (len(valid_text) - context_size)
    return baselines


#
# Instrumentation -- the control room
#


@torch.no_grad()
def field_balance(model: SpinLanguageModel, token_ids: Tensor) -> list[dict]:
    """Decompose each layer's K=1 field into residual, FFN and attention terms.

    ``h = x + f_FFN(norm(x)) + J m_0``.  Which of the three dominates is the
    single most informative thing about whether the stack is doing anything:
    if the attention term is negligible the layer is an MLP with extra steps,
    and if the residual is negligible the drive has been forgotten.

    Reads the module's own ``probe`` rather than re-running its internals here,
    so this measures the pass the model actually makes and cannot drift from it.
    """
    model.eval()
    x = model.token_embedding(token_ids)
    rows = []
    for layer in model.layers:
        readout = layer(x, probe=True)
        probe = readout.probe
        attention = torch.einsum("bhij,bhjd->bhid", probe.couplings, probe.initial)
        settled = readout.state.magnetizations

        norm = lambda t: float(t.double().norm(dim=-1).mean())
        entropy = -(probe.couplings.clamp_min(1e-30).log() * probe.couplings).sum(-1)
        terms = {
            "x": norm(probe.x),
            "ffn": norm(probe.drive - probe.x),
            "attention": norm(attention),
        }
        rows.append(
            {
                **terms,
                "field": norm(probe.drive + attention),
                "output": norm(settled),
                "saturation": norm(settled) / layer.radius_head,
                "attn_entropy": float(entropy.mean()),
                "dominant": max(terms, key=terms.get),
            }
        )
        x = readout.magnetizations
    model.train()
    return rows


@torch.no_grad()
def export_probe(model: SpinLanguageModel, token_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Per-layer trained couplings and drives on one probe sequence, for 01 --real.

    Experiment 01's alignment result rests on structureless ``softmax(randn)``
    couplings; whether it survives couplings with trained structure is its open
    risk, and these tensors -- shaped (depth, heads, n, n) and (depth, heads, n,
    dim_head) -- are what that check runs on.
    """
    model.eval()
    x = model.token_embedding(token_ids)
    couplings, drives = [], []
    for layer in model.layers:
        readout = layer(x, probe=True)
        couplings.append(readout.probe.couplings[0])
        drives.append(readout.probe.drive[0])
        x = readout.magnetizations
    model.train()
    return torch.stack(couplings), torch.stack(drives)


@torch.no_grad()
def embedding_geometry(model: SpinLanguageModel) -> dict:
    """Are the token embeddings spread out, or collapsing onto one direction?

    Anisotropy is the usual failure: everything drifts into a narrow cone and
    cosine similarity stops carrying information.  Here it would also break the
    physics, since the drive directions are what the couplings compare.
    """
    weight = F.normalize(model.token_embedding.weight.double(), dim=-1)
    cosine = weight @ weight.T
    off_diagonal = cosine[~torch.eye(len(weight), dtype=torch.bool)]
    singular = torch.linalg.svdvals(weight - weight.mean(0))
    spectrum = singular / singular.sum()
    return {
        "mean_abs_cos": float(off_diagonal.abs().mean()),
        "max_cos": float(off_diagonal.max()),
        # Participation ratio of the singular spectrum: how many directions the
        # vocabulary actually uses, as a fraction of how many it could.
        "effective_rank": float(torch.exp(-(spectrum * spectrum.log()).sum()))
        / len(singular),
    }


def gradient_report(model: SpinLanguageModel, activations: list[Tensor]) -> dict:
    """Backward signal per layer, plus a global audit that nothing died."""
    components = ("to_qk", "to_v", "ffn", "drive_norm", "attn_temperature")
    per_layer = []
    for index, layer in enumerate(model.layers):
        entry = {"activation": float(activations[index].grad.norm())}
        for component in components:
            total = sum(
                float(parameter.grad.norm()) ** 2
                for name, parameter in layer.named_parameters()
                if name.startswith(component) and parameter.grad is not None
            )
            entry[component] = total**0.5
        per_layer.append(entry)

    missing = [
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all()
    ]
    return {"per_layer": per_layer, "missing": missing}


@torch.no_grad()
def evaluate(model, tokens, *, batch_size, seq_len, batches, seed) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        inputs, targets = sample_batch(
            tokens, batch_size=batch_size, seq_len=seq_len, generator=generator
        )
        logits, _ = model(inputs)
        losses.append(F.cross_entropy(logits.flatten(0, 1), targets.flatten()))
    model.train()
    return float(torch.stack(losses).mean())


def print_control_room(step, fields, gradients, geometry) -> None:
    print(f"\n  forward / backward by layer @ step {step}")
    print(
        f"    {'layer':>5} {'|x|':>7} {'|ffn|':>7} {'|Jm|':>7} {'|h|':>7} "
        f"{'|m|/R':>7} {'attnH':>7} {'largest':>10} | "
        f"{'d|act|':>9} {'qk':>9} {'v':>9} {'ffn':>9} {'temp':>9}"
    )
    for index, (field, grad) in enumerate(zip(fields, gradients["per_layer"])):
        print(
            f"    {index:5d} {field['x']:7.3f} {field['ffn']:7.3f} "
            f"{field['attention']:7.3f} {field['field']:7.3f} "
            f"{field['saturation']:7.3f} {field['attn_entropy']:7.3f} "
            f"{field['dominant']:>10} | "
            f"{grad['activation']:9.2e} {grad['to_qk']:9.2e} {grad['to_v']:9.2e} "
            f"{grad['ffn']:9.2e} {grad['attn_temperature']:9.2e}"
        )
    print(
        f"    embeddings: mean|cos| {geometry['mean_abs_cos']:.3f}  "
        f"max cos {geometry['max_cos']:.3f}  "
        f"effective rank {geometry['effective_rank']:.3f}"
    )
    if gradients["missing"]:
        print(f"    !! no finite gradient: {gradients['missing']}")


#
# Figures
#


def figure_training(history: dict, baselines: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(history["step"], history["train"], color=common.SAMPLED, label="train")
    ax.plot(
        history["eval_step"], history["valid"], color=common.EXACT, marker="o",
        label="held out",
    )
    for order, value in baselines.items():
        ax.axhline(value, color=common.INK_FAINT, lw=0.9, ls=":")
        ax.annotate(
            f"{order}-gram",
            xy=(history["step"][-1], value),
            xytext=(-4, 3),
            textcoords="offset points",
            ha="right",
            fontsize=7.5,
            color=common.INK_SOFT,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("cross entropy (nats)")
    ax.legend(loc="upper right")
    ax.set_title("Character-level next-token loss", color=common.INK)
    fig.tight_layout()
    common.save(fig, "02_training")
    plt.close(fig)


def figure_signal(history: dict, depth: int) -> None:
    """Forward and backward magnitude against depth, at several checkpoints.

    This is the actual claim of the experiment: both stay flat across layers.
    A stack that multiplies its signal by a constant per layer shows up here as
    a straight line on the log axis, and there is none.
    """
    checkpoints = history["probe_step"]
    picks = [0, len(checkpoints) // 2, len(checkpoints) - 1]
    colors = common.DIM_RAMP[:: max(1, len(common.DIM_RAMP) // len(picks))][: len(picks)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    layers = list(range(depth))

    for pick, color in zip(picks, colors):
        label = f"step {checkpoints[pick]}"
        fields = history["fields"][pick]
        grads = history["gradients"][pick]["per_layer"]
        axes[0].plot(
            layers, [f["field"] for f in fields], color=color, marker="o", label=label
        )
        axes[1].plot(
            layers,
            [g["activation"] for g in grads],
            color=color,
            marker="o",
            label=label,
        )
        axes[2].plot(
            layers,
            [f["saturation"] for f in fields],
            color=color,
            marker="o",
            label=label,
        )

    axes[0].set_ylabel(r"$\|h\|$  (forward field)")
    axes[0].set_title("forward signal", color=common.INK, fontsize=9.5)
    axes[1].set_ylabel(r"$\|\partial L/\partial m\|$")
    axes[1].set_yscale("log")
    axes[1].set_title("backward signal", color=common.INK, fontsize=9.5)
    axes[2].set_ylabel(r"$\|m\| / R$")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("saturation against the sphere", color=common.INK, fontsize=9.5)
    for ax in axes:
        ax.set_xlabel("layer")
        ax.set_xticks(layers)
        ax.legend(fontsize=7.5)
    fig.suptitle(
        "Signal propagation through the stack", color=common.INK, fontsize=11
    )
    fig.tight_layout()
    common.save(fig, "02_signal")
    plt.close(fig)


def figure_field_balance(history: dict, depth: int) -> None:
    """Which of the three terms in h = x + FFN + J m actually carries the field."""
    fields = history["fields"][-1]
    layers = list(range(depth))
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for name, color, label in (
        ("x", common.SAMPLED, r"residual  $x$"),
        ("ffn", common.EXACT, r"memory  $f_{\mathrm{FFN}}(\mathrm{norm}\,x)$"),
        ("attention", common.LARGE_D, r"coupling  $J m_0$"),
    ):
        ax.plot(layers, [f[name] for f in fields], color=color, marker="o", label=label)
    ax.set_xlabel("layer")
    ax.set_ylabel("mean per-site norm")
    ax.set_xticks(layers)
    ax.legend()
    ax.set_title(
        f"Field decomposition at step {history['probe_step'][-1]}", color=common.INK
    )
    fig.tight_layout()
    common.save(fig, "02_field_balance")
    plt.close(fig)


def run_tag(args) -> str:
    """Everything that changes the run's numbers, folded into the cache key."""
    return (
        f"{args.corpus.stem}_s{args.steps}_d{args.dim}x{args.depth}h{args.heads}"
        f"_beta{args.beta:g}_seq{args.seq_len}_b{args.batch_size}_lr{args.lr:g}"
        f"_c{args.clip_grad:g}_{args.dtype}_seed{args.seed}"
    )


def train(args) -> dict:
    torch.set_default_dtype(getattr(torch, args.dtype))
    torch.manual_seed(args.seed)

    train_tokens, valid_tokens, vocabulary, train_text, valid_text = load_corpus(
        args.corpus
    )
    model = SpinLanguageModel(
        len(vocabulary),
        dim=args.dim,
        depth=args.depth,
        num_heads=args.heads,
        beta=args.beta,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    parameters = sum(p.numel() for p in model.parameters())

    print(f"corpus {args.corpus.name}: {len(train_tokens):,} train / "
          f"{len(valid_tokens):,} held out, vocab {len(vocabulary)}")
    print(f"model: dim {args.dim}, depth {args.depth}, heads {args.heads}, "
          f"dim_head {args.dim // args.heads}, beta {args.beta}, {parameters:,} parameters")
    print("computing n-gram baselines ...", flush=True)
    baselines = ngram_baselines(train_text, valid_text)
    print("  " + "  ".join(f"{k}-gram {v:.3f}" for k, v in baselines.items()))

    console = common.Console(
        {"step": 6, "train CE": 9, "held out": 9, "grad": 9, "tok/s": 8, "elapsed": 8}
    )
    console.rule("experiment 02 -- trainability of a spin-model stack")
    console.header()

    generator = torch.Generator().manual_seed(args.seed)
    probe_inputs, _ = sample_batch(
        valid_tokens, batch_size=4, seq_len=args.seq_len, generator=generator
    )
    history = {
        "step": [], "train": [], "eval_step": [], "valid": [],
        "probe_step": [], "fields": [], "gradients": [], "geometry": [],
    }
    start = time.time()
    for step in range(1, args.steps + 1):
        inputs, targets = sample_batch(
            train_tokens,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            generator=generator,
        )
        probing = step % args.log_every == 0 or step == 1
        logits, activations = model(inputs, retain=probing)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradients = gradient_report(model, activations) if probing else None
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        history["step"].append(step)
        history["train"].append(float(loss.detach()))

        if probing:
            valid = evaluate(
                model,
                valid_tokens,
                batch_size=args.batch_size,
                seq_len=args.seq_len,
                batches=args.eval_batches,
                seed=args.seed,
            )
            fields = field_balance(model, probe_inputs)
            geometry = embedding_geometry(model)
            history["eval_step"].append(step)
            history["valid"].append(valid)
            history["probe_step"].append(step)
            history["fields"].append(fields)
            history["gradients"].append(gradients)
            history["geometry"].append(geometry)

            tokens_per_second = step * args.batch_size * args.seq_len / (time.time() - start)
            console.row(
                f"{step}",
                float(loss.detach()),
                valid,
                float(grad_norm.detach()),
                f"{tokens_per_second:,.0f}",
                console.elapsed(),
            )
            print_control_room(step, fields, gradients, geometry)

    print(f"\nfinished in {console.elapsed()}")

    couplings, drives = export_probe(model, probe_inputs)
    return {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "history": history,
        "baselines": baselines,
        # Trained couplings and drives for experiment 01 --real, in the float64
        # the sampling there runs in.
        "couplings": couplings.double(),
        "drives": drives.double(),
        "beta": args.beta,
        "dim_head": args.dim // args.heads,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4, help="dim/heads is the D that sets large-D accuracy")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--refresh", action="store_true", help="retrain even if cached")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw from a cached run (pass the same hyperparameter flags), "
        "training nothing -- for tuning figures",
    )
    args = parser.parse_args()

    common.use_style()
    common.PLOT_ONLY = args.plot_only
    payload = common.cached(
        f"02/{run_tag(args)}", lambda: train(args), refresh=args.refresh
    )

    print("\nfigures")
    depth = payload["config"]["depth"]
    figure_training(payload["history"], payload["baselines"])
    figure_signal(payload["history"], depth)
    figure_field_balance(payload["history"], depth)


if __name__ == "__main__":
    main()
