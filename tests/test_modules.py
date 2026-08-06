"""The spin-model transformer module: quadrants, physics constraints, probes."""

from __future__ import annotations

import pytest
import torch

from neqnn import SpinModelTransformerModule, advance, mean_field as mf, proxies, vmf


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
    assert readout.output.shape == x.shape
    assert readout.state.magnetizations.shape == (2, 4, 16, 16)
    assert float(readout.entropy_production.detach().min()) >= 0
    assert (readout.fixed_point is not None) == (num_steps is None)
    if readout.fixed_point is not None:
        assert readout.fixed_point.converged
    readout.magnetizations.sum().backward()
    assert module.drive_norm.weight.grad is not None


def test_magnetizations_respect_the_head_radius():
    dim, num_heads = 128, 8
    module = SpinModelTransformerModule(
        dim=dim, num_heads=num_heads, num_steps=8, beta=2.0
    )
    state = module(torch.randn(2, 16, dim)).state.magnetizations
    assert float(state.detach().norm(dim=-1).max()) <= module.radius_head + 1e-9


@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0])
def test_conjugate_carrier_is_an_exact_one_step_identity(beta):
    dim, num_heads = 32, 4
    module = SpinModelTransformerModule(
        dim=dim,
        num_heads=num_heads,
        num_steps=1,
        init="reset",
        input_mode="magnetization",
        beta=beta,
        causal=True,
        ffn=False,
    ).double()
    per_head = torch.nn.functional.normalize(
        torch.randn(2, num_heads, 5, dim // num_heads, dtype=torch.float64), dim=-1
    )
    per_head = 0.6 * module.radius_head * per_head
    x = module.merge_heads(per_head).detach().requires_grad_()

    readout = module(x, probe=True)
    assert torch.allclose(readout.magnetizations, x, rtol=1e-12, atol=1e-12)
    assert torch.allclose(
        readout.probe.drive,
        vmf.inverse_response_large_d(per_head, beta),
        rtol=1e-12,
        atol=1e-12,
    )

    cotangent = torch.randn_like(readout.magnetizations)
    input_gradient = torch.autograd.grad(
        readout.magnetizations, x, grad_outputs=cotangent
    )[0]
    assert torch.allclose(input_gradient, cotangent, rtol=1e-11, atol=1e-11)


def test_idle_conjugate_stack_preserves_signal_and_gradient_at_depth():
    dim, depth = 16, 64
    modules = torch.nn.ModuleList(
        SpinModelTransformerModule(
            dim=dim,
            num_steps=1,
            init="reset",
            input_mode="magnetization",
            beta=1.0,
            ffn=False,
        ).double()
        for _ in range(depth)
    )
    x = torch.nn.functional.normalize(
        torch.randn(1, 3, dim, dtype=torch.float64), dim=-1
    )
    x = (0.5 * modules[0].radius_head * x).detach().requires_grad_()
    carried = x
    for module in modules:
        carried = module(carried).magnetizations

    assert torch.allclose(carried, x, rtol=1e-11, atol=1e-11)
    cotangent = torch.randn_like(carried)
    input_gradient = torch.autograd.grad(carried, x, grad_outputs=cotangent)[0]
    assert torch.allclose(input_gradient, cotangent, rtol=1e-10, atol=1e-10)


def test_conjugate_carrier_rejects_boundary_magnetizations():
    module = SpinModelTransformerModule(
        dim=16,
        num_steps=1,
        init="reset",
        input_mode="magnetization",
        ffn=False,
    )
    x = torch.zeros(1, 2, 16)
    x[..., 0] = module.radius_head
    with pytest.raises(ValueError, match="strictly inside"):
        module(x)


def test_amortized_initializer_respects_the_head_radius():
    module = SpinModelTransformerModule(
        dim=128, num_heads=8, num_steps=1, init="amortized"
    )
    with torch.no_grad():
        module.to_v.weight.mul_(1e4)
    normalized = module.normalize(torch.randn(2, 16, 128))
    initial = module.initial(normalized, None)
    assert float(initial.detach().norm(dim=-1).max()) < module.radius_head


def test_amortized_initializer_is_response_to_value_field():
    module = SpinModelTransformerModule(
        dim=32, num_heads=2, num_steps=1, init="amortized", beta=1.7
    )
    normalized = module.normalize(torch.randn(2, 5, 32))
    initializer_field = module.split_heads(module.to_v(normalized))

    assert torch.allclose(
        module.initial(normalized, None),
        vmf.response_large_d(initializer_field, module.beta),
    )


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


def test_external_drive_offset_nudges_relaxation_without_changing_features():
    module = SpinModelTransformerModule(
        dim=16,
        num_steps=1,
        init="reset",
        beta=1.0,
        ffn=False,
    ).double()
    x = torch.zeros(1, 3, 16, dtype=torch.float64)
    offset = torch.zeros_like(x)
    offset[:, 0, 0] = 0.75

    baseline = module(x, probe=True)
    nudged = module(x, probe=True, drive_offset=offset)

    assert torch.allclose(baseline.magnetizations, torch.zeros_like(x))
    assert torch.allclose(nudged.probe.x, baseline.probe.x)
    assert torch.allclose(nudged.probe.couplings, baseline.probe.couplings)
    assert torch.allclose(
        nudged.probe.drive,
        baseline.probe.drive + module.split_heads(offset),
    )
    assert torch.allclose(
        module.split_heads(nudged.magnetizations),
        vmf.response_large_d(nudged.probe.drive, module.beta),
    )


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
        module(x).magnetizations[:, :6],
        module(changed).magnetizations[:, :6],
        atol=1e-10,
    )


def test_advance_realigns_the_window():
    state = (
        SpinModelTransformerModule(dim=64, num_heads=2, num_steps=2)
        .forward(torch.randn(1, 10, 64))
        .state
    )
    moved = advance(state)
    assert moved.magnetizations.shape == state.magnetizations.shape
    assert torch.allclose(
        moved.magnetizations[..., :-1, :], state.magnetizations[..., 1:, :]
    )
    assert torch.all(moved.magnetizations[..., -1, :] == 0)


def test_advance_validates_shape_and_fill():
    state = (
        SpinModelTransformerModule(dim=64, num_heads=2)
        .forward(torch.randn(1, 4, 64))
        .state
    )
    with pytest.raises(ValueError, match="cannot drop"):
        advance(state, drop=5)
    with pytest.raises(ValueError, match="fill must have shape"):
        advance(state, fill=torch.zeros(1, 2, 2, 32))
    with pytest.raises(ValueError, match="add=0"):
        advance(state, add=0, fill=torch.zeros(1, 2, 0, 32))


def test_post_mix_is_output_only():
    module = SpinModelTransformerModule(dim=64, num_heads=2, num_steps=2, post_mix=True)
    readout = module(torch.randn(2, 8, 64))
    physical = module.merge_heads(readout.state.magnetizations)
    assert torch.allclose(readout.magnetizations, physical)
    assert torch.allclose(readout.output, module.post_mix(physical))
    assert not torch.allclose(readout.output, readout.magnetizations)


def test_relaxation_indices_match_the_forward_pass():
    module = SpinModelTransformerModule(
        dim=64,
        num_heads=2,
        num_steps=1,
        init="amortized",
        causal=True,
        qk_bias=True,
        rope=True,
        pre_mix=True,
        measure_entropy_production=True,
    )
    x = torch.randn(2, 8, 64)
    readout = module(x, probe=True)
    trace = module.relaxation(x, num_steps=1)
    probe = readout.probe
    assert probe is not None

    # k=0 is the exact initializer used by forward; k=1 is its exact output.
    assert torch.allclose(trace.magnetizations[0], probe.initial)
    assert torch.allclose(trace.magnetizations[1], readout.state.magnetizations)
    field_0 = mf.effective_field(trace.magnetizations[0], probe.drive, probe.couplings)
    field_1 = mf.effective_field(trace.magnetizations[1], probe.drive, probe.couplings)
    assert torch.allclose(
        vmf.response_large_d(field_0, module.beta), trace.magnetizations[1]
    )

    # KL[k] is attached to h(m_k), while entropy[0] spans m_0 -> m_1.
    steady_field = mf.effective_field(
        trace.fixed_point.solution, probe.drive, probe.couplings
    )
    assert torch.allclose(
        trace.mismatch,
        proxies.mismatch(torch.stack([field_0, field_1]), steady_field, module.beta),
    )
    assert torch.allclose(trace.entropy_production[0], readout.entropy_production)
    assert trace.fixed_point.converged

    # A counterfactual start changes only the path and can reuse the exact target.
    counter_start = torch.zeros_like(trace.magnetizations[0])
    counter = module.relaxation(
        x,
        num_steps=1,
        start=counter_start,
        reference=trace.fixed_point.solution,
    )
    assert torch.allclose(counter.magnetizations[0], counter_start)
    assert torch.allclose(counter.fixed_point.solution, trace.fixed_point.solution)
    assert torch.allclose(
        counter.magnetizations[1],
        mf.step_large_d(counter_start, probe.drive, probe.couplings, module.beta),
    )
    assert counter.fixed_point.converged


def test_fixed_point_non_convergence_is_loud_and_attached():
    module = SpinModelTransformerModule(
        dim=64,
        num_heads=2,
        num_steps=None,
        max_iter=2,
        tol=1e-15,
    )
    with pytest.warns(RuntimeWarning, match="fixed-point forward residual"):
        readout = module(torch.randn(1, 8, 64))
    assert readout.fixed_point is not None
    assert not readout.fixed_point.converged


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dim": 0},
        {"dim": 64, "num_heads": 0},
        {"dim": 64, "init": "typo"},
        {"dim": 64, "input_mode": "typo"},
        {"dim": 64, "input_mode": "magnetization", "pre_mix": True},
        {"dim": 64, "num_steps": 0},
        {"dim": 64, "beta": 0},
        {"dim": 64, "max_iter": 1},
        {"dim": 64, "tol": 0},
        {"dim": 64, "rope_base": 0},
        {"dim": 15, "num_heads": 3, "rope": True},
        {"dim": 4, "num_heads": 2},
    ],
)
def test_constructor_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SpinModelTransformerModule(**kwargs)


def test_forward_validates_mask_and_state():
    module = SpinModelTransformerModule(dim=64, num_heads=2, causal=True)
    x = torch.randn(1, 4, 64)
    with pytest.raises(ValueError, match="no available key"):
        module(x, mask=torch.zeros(1, 4, dtype=torch.bool))
    with pytest.raises(TypeError, match="boolean"):
        module(x, mask=torch.ones(1, 4))
    with pytest.raises(ValueError, match="state magnetizations"):
        module(
            x,
            state=type(module(x).state)(magnetizations=torch.zeros(1, 2, 3, 32)),
        )
    with pytest.raises(ValueError, match="same shape"):
        module(x, drive_offset=torch.zeros(1, 3, 64))
    with pytest.raises(TypeError, match="floating point"):
        module(x, drive_offset=torch.zeros_like(x, dtype=torch.long))
    with pytest.raises(ValueError, match="start must have shape"):
        module.relaxation(x, start=torch.zeros(1, 2, 3, 32))
    with pytest.raises(ValueError, match="reference must share"):
        module.relaxation(
            x,
            reference=torch.zeros(1, 2, 4, 32, dtype=torch.float32),
        )


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
