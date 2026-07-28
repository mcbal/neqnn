# Baseline experiments

Compute and plotting are separate. Both experiments default to small CPU runs;
increase the sampling or training budgets only after the smoke commands work.
Fidelity uses `N=32`, `D=3,16,64`, and a 5-by-5 sampled phase grid. The language
models use `K=1`, amortized initialization, and depths 3, 6, and 12.

```bash
# Mean-field fidelity
uv run python experiments/fidelity_compute.py
uv run python experiments/fidelity_plot.py

# Character-level language model, depths 3 / 6 / 12
uv run python experiments/language_model_compute.py
uv run python experiments/language_model_plot.py
```

Fast wiring checks:

```bash
uv run python experiments/fidelity_compute.py \
  --dims 3 --u 0.5,2 --beta 1,4 --fine 3 --chains 2 --steps 4 --burn-in 2 \
  --output /tmp/fidelity-smoke.pt

uv run python experiments/language_model_compute.py \
  --depths 3 --steps 2 --eval-batches 1 --sample-tokens 2 \
  --output /tmp/language-model-smoke.pt
```

Suggested sampling budgets:

```bash
# ~2% floors in representative cells
uv run python experiments/fidelity_compute.py \
  --repeats 5 --chains 16 --steps 200 --burn-in 80

# ~1% floors; use a GPU if available
uv run python experiments/fidelity_compute.py \
  --repeats 5 --chains 64 --steps 400 --burn-in 150
```

These are starting points, not promises: inspect the grey resolution mask per
observable, then rerun only unresolved `--dims`, `--u`, or `--beta` slices.
Strong-coupling mixing and multistability are diagnostics, not problems that a
larger sample count necessarily fixes.

The fidelity figures show noise-corrected relative errors. Grey regions are
below their own Monte Carlo resolution; crosses and plus signs mark multistable
or failed exact mean-field solves. Smooth contours interpolate only for visual
presentation, and the measured sampling locations remain visible.

The corpus loader removes Project Gutenberg boilerplate, normalizes line
endings, unwraps print-width line breaks, and preserves paragraph breaks as
learned newline characters.

The language-model control room reports raw per-layer gradient RMS values plus
pre/post global clipping norms. The default cap is 10: high enough not to
rescale the healthy depth-dependent initial norms measured here, while retaining
an emergency guard against genuine spikes. Smoothed unigram, bigram, and trigram
held-out cross entropies provide scale for the learned-model loss.
