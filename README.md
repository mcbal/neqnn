# Nonequilibrium dynamics in spin-model transformers

Companion code and package for blog post [Nonequilibrium Dynamics in Spin-Model Transformers (Bal, 2026)](https://mcbal.github.io/post/nonequilibrium-dynamics-in-spin-model-transformers/).

## Getting started

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

See the `experiments` scripts for examples on how to use and probe the module.