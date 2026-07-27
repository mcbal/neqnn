# neqnn

Nonequilibrium dynamics in spin-model transformers — companion code for
[this post](https://mcbal.github.io/post/entropy-production-in-nonequilibrium-neural-networks/).

A transformer block whose forward pass *is* the relaxation of a vector-spin
system. Spins live on a sphere of radius `R`, softmax attention supplies the
coupling rule `J(X_t)`, and the layer iterates the mean-field magnetization
recurrence

    m_{k+1} = phi(x + f_FFN(x) + J(X_t) m_k)

on three separated timescales: `k` for internal relaxation, `t` for the drive,
`n` for learning.

```python
import torch
from neqnn import SpinModelTransformerModule

module = SpinModelTransformerModule(
    dim=256,
    num_heads=4,
    num_steps=1,          # int -> finite horizon;  None -> fixed point
    init="amortized",     # "reset" | "amortized" | "carried"
    rope=True,
    measure_entropy_production=True,
)

out = module(torch.randn(1, 32, 256))
out.magnetizations       # physical mean-field state, (1, 32, 256)
out.output               # possibly post-mixed neural-network readout
out.state                # feed to the next drive step
out.entropy_production   # housekeeping cost of the steady state
out.fixed_point          # solver evidence when num_steps=None, otherwise None
```
