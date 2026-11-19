"""Smoke tests for the named experimental nano preset."""

import pytest
import torch

from libreyolo.models.yolo11 import (
    EXPERIMENTAL_NANO,
    YOLO11nDecoder,
    YOLO11NPreset,
    build_experimental_nano,
)

pytestmark = pytest.mark.unit


def test_preset_is_named_versioned_and_consistent():
    assert isinstance(EXPERIMENTAL_NANO, YOLO11NPreset)
    assert EXPERIMENTAL_NANO.name == "yolo11n-experimental"
    assert EXPERIMENTAL_NANO.version == "0.1.0"
    c2psa = EXPERIMENTAL_NANO.build_c2psa()
    assert len(c2psa.m) == EXPERIMENTAL_NANO.blocks
    assert c2psa.direct_channels + c2psa.attended_channels == 256


def test_public_entry_point_builds_a_decodable_nano():
    c2psa, head = build_experimental_nano(7)
    c2psa.eval()
    head.eval()
    with torch.no_grad():
        stage = c2psa(torch.randn(2, 256, 20, 20))
        assert stage.shape == (2, 256, 20, 20)
        raw = head(
            [
                torch.randn(2, 64, 80, 80),
                torch.randn(2, 128, 40, 40),
                torch.randn(2, 256, 20, 20),
            ]
        )
        assert [tuple(level.shape) for level in raw] == [
            (2, 71, 80, 80),
            (2, 71, 40, 40),
            (2, 71, 20, 20),
        ]
        decoded = YOLO11nDecoder()(raw)
    assert decoded.shape == (2, 11, 8400)


def test_preset_builders_use_the_declared_topologies():
    c2psa = EXPERIMENTAL_NANO.build_c2psa()
    head = EXPERIMENTAL_NANO.build_head(5)
    assert c2psa.m[0].attn.num_heads == EXPERIMENTAL_NANO.attention.heads
    assert head.box_towers[0][0].conv.out_channels == (
        EXPERIMENTAL_NANO.towers.box_width
    )
    assert head.nc == 5
    for tower in head.class_towers:
        bias = tower[-1].bias
        torch.testing.assert_close(bias.sigmoid(), torch.full_like(bias, 0.01))
