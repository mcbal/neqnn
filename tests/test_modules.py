"""The spin-model transformer module: quadrants, physics constraints, probes."""

from __future__ import annotations

import pytest
import torch

from neqnn import SpinModelTransformerModule, advance


@pytest.mark.parametrize("num_steps", [1, 4, None])
@pytest.mark.parametrize("init", ["reset", "amortized", "carried"])
def test_every_quadrant_runs_forward_and_backward(num_steps, init):
    module = SpinModelTransformerModule(
        dim=64,
        num_heads=4,
        num_steps=num_steps,
        init=init,
        beta=1.0,
        rope=True,
        measure_entropy_production=True,
    )
    x = torch.randn(2, 16, 64)
    readout = module(x)
    assert readout.magnetizations.shape == x.shape
    assert readout.state.magnetizations.shape == (2, 4, 16, 16)
    assert float(readout.entropy_production.detach().min()) >= 0
    readout.magnetizations.sum().backward()
    assert module.drive_norm.weight.grad is not None


def test_magnetizations_respect_the_head_radius():
    dim, num_heads = 128, 8
    module = SpinModelTransformerModule(
        dim=dim, num_heads=num_heads, num_steps=8, beta=2.0
    )
    state = module(torch.randn(2, 16, dim)).state.magnetizations
    assert float(state.detach().norm(dim=-1).max()) <= module.radius_head + 1e-9


def test_probe_returns_the_ingredients_of_the_same_pass():
    """The probe must expose exactly what the forward consumed, not a re-run."""
    dim, num_heads, sites = 64, 4, 16
    module = SpinModelTransformerModule(
        dim=dim, num_heads=num_heads, num_steps=1, init="amortized", beta=1.0
    )
    x = torch.randn(2, sites, dim)
    readout = module(x, probe=True)
    probe = readout.probe
    assert probe is not None
    assert module(x).probe is None  # opt-in, absent by default

    head_dim = dim // num_heads
    assert probe.x.shape == (2, num_heads, sites, head_dim)
    assert probe.drive.shape == (2, num_heads, sites, head_dim)
    assert probe.couplings.shape == (2, num_heads, sites, sites)
    assert probe.initial.shape == (2, num_heads, sites, head_dim)
    # Couplings are softmax rows.
    assert torch.allclose(probe.couplings.sum(-1), torch.ones(2, num_heads, sites))
    # Re-running the relaxation from the probed ingredients reproduces the state.
    settled, _ = module.settle(probe.initial, probe.drive, probe.couplings)
    assert torch.allclose(settled, readout.state.magnetizations)


def test_one_step_from_reset_cannot_see_the_couplings():
    """m_0 = 0 kills the coupling term outright, so nothing routing-related learns."""
    module = SpinModelTransformerModule(dim=64, num_steps=1, init="reset", beta=1.0)
    module(torch.randn(2, 16, 64)).magnetizations.sum().backward()
    assert module.to_qk.weight.grad.abs().max() == 0
    assert module.attn_temperature.grad == 0


def test_the_fixed_point_forgets_its_initialization():
    """At K -> inf the init is inert, so to_v receives no gradient at all."""
    module = SpinModelTransformerModule(
        dim=64, num_steps=None, init="amortized", beta=1.0
    )
    module(torch.randn(2, 16, 64)).magnetizations.sum().backward()
    # Not merely zero: the solve runs under no_grad and the implicit gradient is
    # reattached at the solution, so the init never enters the graph at all.
    assert module.to_v.weight.grad is None


def test_causal_masking_keeps_the_prefix_independent_of_the_suffix():
    module = SpinModelTransformerModule(
        dim=64, num_heads=2, num_steps=4, init="reset", beta=1.0, causal=True
    )
    x = torch.randn(1, 12, 64)
    changed = x.clone()
    changed[:, 6:] = torch.randn_like(changed[:, 6:])
    assert torch.allclose(
        module(x).magnetizations[:, :6], module(changed).magnetizations[:, :6], atol=1e-10
    )


def test_advance_realigns_the_window():
    state = SpinModelTransformerModule(dim=64, num_heads=2, num_steps=2).forward(
        torch.randn(1, 10, 64)
    ).state
    moved = advance(state)
    assert moved.magnetizations.shape == state.magnetizations.shape
    assert torch.allclose(moved.magnetizations[..., :-1, :], state.magnetizations[..., 1:, :])
    assert torch.all(moved.magnetizations[..., -1, :] == 0)


def test_relaxation_traces_a_converging_path():
    module = SpinModelTransformerModule(dim=64, num_heads=2, num_steps=4, beta=1.0)
    trace = module.relaxation(torch.randn(1, 16, 64), num_steps=48)
    assert trace.magnetizations.shape[0] == 49
    # Leading axis is k, the rest are batch and head.
    assert float(trace.residual[-1].max()) < float(trace.residual[0].max())
    assert float(trace.mismatch[-1].max()) < float(trace.mismatch[0].max())
    # Housekeeping cost is what survives at the steady state, so it must not
    # decay to zero the way the mismatch does.
    assert float(trace.entropy_production[-1].min()) > 0
