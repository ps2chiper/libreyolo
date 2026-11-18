"""Tests for the YOLO11 detection-tower interface.

The topology used here is a synthetic experimental variant. It is not a claim
about the baseline nano detector; the cited sources leave those values open.
"""

import pytest
import torch

from libreyolo.models.yolo11.decoder import YOLO11nDecoder
from libreyolo.models.yolo11.detect import TowerTopology, YOLO11Head

pytestmark = pytest.mark.unit

NANO_LEVELS = (64, 128, 256)  # Algorithm A1, P3/P4/P5
TOPO = TowerTopology(
    box_width=32,
    class_width=24,
    box_kernel=3,
    class_kernel=3,
    projection_kernel=1,
    output_bias=True,
)


def features(batch: int, channels=(64, 128, 256), sizes=((80, 80), (40, 40), (20, 20))):
    return [
        torch.randn(batch, count, height, width)
        for count, (height, width) in zip(channels, sizes)
    ]


def test_raw_and_decoded_shapes_for_two_class_counts():
    for classes in (3, 80):
        head = YOLO11Head(classes, NANO_LEVELS, TOPO).eval()
        raw = head(features(2))
        assert [tuple(level.shape) for level in raw] == [
            (2, 64 + classes, 80, 80),
            (2, 64 + classes, 40, 40),
            (2, 64 + classes, 20, 20),
        ]
        for level in raw:
            assert level[:, :64].shape[1] == 64
        with torch.no_grad():
            decoded = YOLO11nDecoder()(raw)
        assert decoded.shape == (2, 4 + classes, 8400)


def test_rectangular_canvas_keeps_level_geometry():
    head = YOLO11Head(3, NANO_LEVELS, TOPO).eval()
    raw = head(
        features(2, sizes=((40, 80), (20, 40), (10, 20)))
    )
    assert [tuple(level.shape) for level in raw] == [
        (2, 67, 40, 80),
        (2, 67, 20, 40),
        (2, 67, 10, 20),
    ]
    with torch.no_grad():
        decoded = YOLO11nDecoder()(raw)
    assert decoded.shape == (2, 7, 4200)


def test_finite_outputs_and_gradients():
    head = YOLO11Head(5, (16, 32, 48), TOPO).train()
    raw = head(features(2, channels=(16, 32, 48), sizes=((8, 8), (4, 4), (2, 2))))
    total = torch.zeros(())
    for level in raw:
        total = total + level.square().mean()
    total.backward()
    for parameter in head.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_rejects_wrong_levels_and_channels():
    head = YOLO11Head(3, NANO_LEVELS, TOPO)
    with pytest.raises(ValueError, match="three"):
        YOLO11Head(3, (64, 128), TOPO)
    with pytest.raises(ValueError, match="levels"):
        head(features(1, sizes=((80, 80), (40, 40), (20, 20)))[:2])
    with pytest.raises(ValueError, match="channels"):
        head(features(1, channels=(64, 128, 512)))


def test_rejects_incomplete_topology():
    with pytest.raises(ValueError, match="class_kernel"):
        TowerTopology(
            box_width=32,
            class_width=24,
            box_kernel=3,
            class_kernel=2,
            projection_kernel=1,
            output_bias=True,
        )