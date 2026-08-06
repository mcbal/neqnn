# 🧲 Nonequilibrium dynamics in spin-model transformers

Companion experimental code and `neqnn` package for
[Nonequilibrium Dynamics in Spin-Model Transformers (Bal, 2026)](https://mcbal.github.io/post/nonequilibrium-dynamics-in-spin-model-transformers/).

A transformer block whose forward pass is the relaxation of a vector-spin
system after a quench. Spins live on a sphere of radius `R`, softmax attention supplies the drive-conditioned coupling rule `J(X_t)`, and the layer iterates a mean-field magnetization recurrence relation on three separated timescales: internal magnetization update (relax), external drive update (quench), and slow parameter update (learn).

```python
import torch
from neqnn import SpinModelTransformerModule

module = SpinModelTransformerModule(
    dim=256,
    num_heads=4,
    num_steps=1,          # int -> finite relaxation horizon;  None -> fixed point
    init="learned",       # "reset" | "learned" | "carried"
)

out = module(torch.randn(1, 32, 256))
out.magnetizations        # magnetizations, (1, 32, 256)
```
