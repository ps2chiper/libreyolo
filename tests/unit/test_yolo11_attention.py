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
    ffn_expansion=2.0,
    residual=True,
)


def test_shape_preservation_on_published_and_rectangular_maps():
    model = C2PSA(NANO_STAGE_CHANNELS, blocks=1, topology=TOPO).eval()
    square = torch.randn(1, NANO_STAGE_CHANNELS, 20, 20)
    rectangular = torch.randn(1, NANO_STAGE_CHANNELS, 20, 36)
    with torch.no_grad():
        assert model(square).shape == square.shape
        assert model(rectangular).shape == rectangular.shape


def test_finite_outputs_and_gradients():
    model = C2PSA(NANO_STAGE_CHANNELS, blocks=2, topology=TOPO).train()
    x = torch.randn(2, NANO_STAGE_CHANNELS, 20, 20, requires_grad=True)
    out = model(x)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
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
    assert block.add is True
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
            heads=2,
            query_width=8,
            key_width=8,
            value_width=16,
            packing="interleaved",
            position_kernel=3,
            ffn_expansion=2.0,
            residual=True,
        )
    with pytest.raises(ValueError, match="position_kernel"):
        replace(TOPO, position_kernel=2)
    with pytest.raises(ValueError, match="PSABlock"):
        C2PSA(64, blocks=0, topology=TOPO)
    with pytest.raises(ValueError, match="channels"):
        C2PSA(64, blocks=1, topology=TOPO)(torch.zeros(1, 32, 4, 4))