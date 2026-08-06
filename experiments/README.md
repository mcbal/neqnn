# Experiments

## Mean-field fidelity

To reproduce the blog post experiments:

```bash
uv run python experiments/fidelity_compute.py \
  --sites 64 --dims 3,16,64,128 \
  --u 0.25,0.5,0.75,1,1.5,2,4 --beta 0.25,0.5,0.75,1,1.5,2,4 --fine 13 \
  --repeats 5 --chains 32 --steps 300 --burn-in 120 \
  --starts 5 --power-iterations 60 --device cpu \
  --output experiments/outputs/fidelity_projection.pt

uv run python experiments/fidelity_plot.py \
  --input experiments/outputs/fidelity_projection.pt \
  --output-dir experiments/figures --prefix fidelity

```

## Toy autoregressive language model training

To reproduce the blog post experiments:

```bash
uv run python experiments/language_model_compute.py \
  --depths 3,6,12,24 --steps 3000 --batch-size 32 --seq-len 128 \
  --dim 128 --heads 4 --input-mode magnetization --lr 1e-3 \
  --log-every 50 --eval-batches 32 --ablation-batches 4 \
  --intervention-depths 6 \
  --intervention-batches 32 \
  --intervention-steps 0,1,50,250,500,1000,1500,2000,2500,3000 \
  --sample-every 100 --sample-tokens 256 --temperature 0.8 --top-k 20 \
  --device cpu --output experiments/outputs/language_model.pt

uv run python experiments/language_model_plot.py \
  --input experiments/outputs/language_model.pt \
  --output-dir experiments/figures --prefix language_model
