# 🧲 Nonequilibrium dynamics in spin-model transformers

Companion experimental code and `neqnn` package for
[Nonequilibrium Dynamics in Spin-Model Transformers (Bal, 2026)](https://mcbal.github.io/post/nonequilibrium-dynamics-in-spin-model-transformers/).

A `SpinModelTransformerModule` is an example implementation of a parallel transformer block whose forward pass performs one or more mean-field update steps following a drive quench.

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
