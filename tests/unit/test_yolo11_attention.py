"""Tests for the YOLO11 C2PSA interface.

The topology used here is a synthetic experimental variant. It is not a claim
about the baseline nano detector; the cited sources leave those values open.
"""

from dataclasses import replace

import pytest
import torch

from libreyolo.models.yolo11.attention import (
    C2PSA,
    PSAAttention,
    PSABlock,
    PSATopology,
)

pytestmark = pytest.mark.unit

NANO_STAGE_CHANNELS = 256  # Algorithm A1, stage 10
TOPO = PSATopology(
    heads=3,
    query_width=8,
    key_width=8,
    value_width=16,
    packing="head_major",
    position_kernel=3,
    ffn_kernel=1,
    ffn_expansion=2.0,
    residual="both",
)


def test_shape_preservation_on_published_and_rectangular_maps():
    model = C2PSA(NANO_STAGE_CHANNELS, blocks=1, topology=TOPO, split_ratio=0.5).eval()
    square = torch.randn(1, NANO_STAGE_CHANNELS, 20, 20)
    rectangular = torch.randn(1, NANO_STAGE_CHANNELS, 20, 36)
    with torch.no_grad():
        assert model(square).shape == square.shape
        assert model(rectangular).shape == rectangular.shape


def test_finite_outputs_and_gradients():
    model = C2PSA(NANO_STAGE_CHANNELS, blocks=2, topology=TOPO, split_ratio=0.5).train()
    x = torch.randn(2, NANO_STAGE_CHANNELS, 20, 20, requires_grad=True)
    out = model(x)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_packing_selects_a_distinct_channel_layout():
    head_major = PSAAttention(32, TOPO).eval()
    channel_major = PSAAttention(32, replace(TOPO, packing="channel_major")).eval()
    channel_major.load_state_dict(head_major.state_dict())
    x = torch.randn(1, 32, 4, 4)
    with torch.no_grad():
        assert not torch.allclose(head_major(x), channel_major(x), atol=1e-6)


def test_residual_paths_return_input_when_sublayers_are_zeroed():
    block = PSABlock(32, TOPO).eval()
    with torch.no_grad():
        block.attn.proj.conv.weight.zero_()
        block.ffn[-1].conv.weight.zero_()
    x = torch.randn(1, 32, 5, 5)
    assert block.residual == "both"
    torch.testing.assert_close(block(x), x)


def test_position_term_can_be_omitted():
    topology = replace(TOPO, position_kernel=None)
    attention = PSAAttention(32, topology)
    assert attention.pe is None
    with torch.no_grad():
        assert attention(torch.randn(1, 32, 6, 6)).shape == (1, 32, 6, 6)


def test_rejects_open_topology_violations_and_bad_shapes():
    with pytest.raises(ValueError, match="packing"):
        PSATopology(
            heads=3,
            query_width=8,
            key_width=8,
            value_width=16,
            packing="interleaved",
            position_kernel=3,
            ffn_kernel=1,
            ffn_expansion=2.0,
            residual="both",
        )
    with pytest.raises(ValueError, match="residual"):
        replace(TOPO, residual="attn")
    with pytest.raises(ValueError, match="positive integer"):
        replace(TOPO, heads=True)
    with pytest.raises(ValueError, match="match"):
        replace(TOPO, key_width=4)
    with pytest.raises(ValueError, match="positive integer"):
        replace(TOPO, heads=3.0)
    with pytest.raises(ValueError, match="position_kernel"):
        replace(TOPO, position_kernel=2)
    with pytest.raises(ValueError, match="ffn_kernel"):
        replace(TOPO, ffn_kernel=2)
    with pytest.raises(ValueError, match="block count"):
        C2PSA(64, blocks=0, topology=TOPO, split_ratio=0.5)
    with pytest.raises(ValueError, match="split_ratio"):
        C2PSA(64, blocks=1, topology=TOPO, split_ratio=1.0)
    with pytest.raises(ValueError, match="channels"):
        C2PSA(64, blocks=1, topology=TOPO, split_ratio=0.5)(
            torch.zeros(1, 32, 4, 4)
        )


def test_residual_disabled_produces_zero_when_sublayers_are_zeroed():
    topology = replace(TOPO, residual="none")
    block = PSABlock(32, topology).eval()
    with torch.no_grad():
        block.attn.proj.conv.weight.zero_()
        block.ffn[-1].conv.weight.zero_()
    x = torch.randn(1, 32, 5, 5)
    assert block.residual == "none"
    torch.testing.assert_close(block(x), torch.zeros_like(x))


def test_attention_only_residual_uses_feed_forward_without_addition():
    topology = replace(TOPO, residual="attention")
    block = PSABlock(32, topology).eval()
    with torch.no_grad():
        block.attn.proj.conv.weight.zero_()
    x = torch.randn(1, 32, 5, 5)
    with torch.no_grad():
        expected = block.ffn(x)
        torch.testing.assert_close(block(x), expected)


def test_asymmetric_split_and_ffn_kernel_preserve_shape():
    model = C2PSA(
        256,
        blocks=1,
        topology=replace(TOPO, ffn_kernel=3),
        split_ratio=0.3,
    ).eval()
    x = torch.randn(2, 256, 7, 9)
    with torch.no_grad():
        assert model(x).shape == x.shape
    assert model.direct_channels + model.attended_channels == 256
    assert model.direct_channels != model.attended_channels


def _manual_reference(module: PSAAttention, x: torch.Tensor) -> torch.Tensor:
    """Attention computed with explicit channel slicing and einsum products."""
    topology = module.topology
    batch, _, height, width = x.shape
    sites = height * width
    packed = module.qkv(x)
    queries, keys, values = [], [], []
    for head in range(topology.heads):
        if topology.packing == "head_major":
            base = head * (
                topology.query_width + topology.key_width + topology.value_width
            )
            q_index = range(base, base + topology.query_width)
            k_index = range(
                base + topology.query_width,
                base + topology.query_width + topology.key_width,
            )
            v_index = range(
                base + topology.query_width + topology.key_width,
                base
                + topology.query_width
                + topology.key_width
                + topology.value_width,
            )
        else:
            q_index = range(
                head * topology.query_width, (head + 1) * topology.query_width
            )
            k_base = topology.heads * topology.query_width
            k_index = range(
                k_base + head * topology.key_width,
                k_base + (head + 1) * topology.key_width,
            )
            v_base = k_base + topology.heads * topology.key_width
            v_index = range(
                v_base + head * topology.value_width,
                v_base + (head + 1) * topology.value_width,
            )
        queries.append(packed[:, list(q_index)].reshape(batch, topology.query_width, sites))
        keys.append(packed[:, list(k_index)].reshape(batch, topology.key_width, sites))
        values.append(packed[:, list(v_index)].reshape(batch, topology.value_width, sites))
    query = torch.stack(queries, dim=1)
    key = torch.stack(keys, dim=1)
    value = torch.stack(values, dim=1)
    if module.pe is not None:
        value_channels = topology.heads * topology.value_width
        position = module.pe(value.reshape(batch, value_channels, height, width))
        value = value + position.view(
            batch, topology.heads, topology.value_width, sites
        )
    scores = torch.einsum("bhwn,bhwm->bhnm", query, key) * module.scale
    weights = scores.softmax(dim=-1)
    attended = torch.einsum("bhnm,bhvm->bhnv", weights, value)
    attended = attended.permute(0, 1, 3, 2)
    return module.proj(
        attended.reshape(
            batch, topology.heads * topology.value_width, height, width
        )
    )


def test_both_packing_layouts_match_an_explicit_reference():
    x = torch.randn(2, 32, 3, 5)
    for packing in ("head_major", "channel_major"):
        module = PSAAttention(32, replace(TOPO, packing=packing)).eval()
        with torch.no_grad():
            torch.testing.assert_close(
                module(x), _manual_reference(module, x), atol=1e-6, rtol=1e-5
            )