"""Tests for the YOLO11 detection-tower interface.

The topology used here is a synthetic experimental variant. It is not a claim
about the baseline nano detector; the cited sources leave those values open.
"""

from dataclasses import replace

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
    box_groups=1,
    projection_kernel=1,
    class_stage_order="dw_pw",
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
            assert torch.isfinite(level).all()
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
        assert torch.isfinite(level).all()
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
            box_groups=1,
            projection_kernel=1,
            class_stage_order="dw_pw",
            output_bias=True,
        )
    with pytest.raises(ValueError, match="class_stage_order"):
        replace(TOPO, class_stage_order="depthwise")
    with pytest.raises(ValueError, match="box_groups"):
        YOLO11Head(3, NANO_LEVELS, replace(TOPO, box_groups=5))


def test_box_channels_precede_class_channels():
    head = YOLO11Head(3, NANO_LEVELS, TOPO).eval()
    with torch.no_grad():
        for parameter in head.class_towers.parameters():
            parameter.zero_()
    for level in head(features(1)):
        assert level[:, :64].abs().sum() > 0
        assert level[:, 64:].abs().sum() == 0


def test_decoded_level_order_and_strides():
    head = YOLO11Head(3, NANO_LEVELS, TOPO).eval()
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    with torch.no_grad():
        decoded = YOLO11nDecoder()(head(features(1)))
    # Uniform distributions give 7.5 grid cells per side; each level's first
    # anchor is stride // 2, so the boxes below pin level order and stride.
    torch.testing.assert_close(decoded[0, :4, 0], torch.tensor([-56.0, -56.0, 64.0, 64.0]))
    torch.testing.assert_close(
        decoded[0, :4, 6400], torch.tensor([-112.0, -112.0, 128.0, 128.0])
    )
    torch.testing.assert_close(
        decoded[0, :4, 8000], torch.tensor([-224.0, -224.0, 256.0, 256.0])
    )


def test_output_bias_can_be_disabled():
    head = YOLO11Head(3, NANO_LEVELS, replace(TOPO, output_bias=False))
    assert head.box_towers[0][-1].bias is None
    assert head.class_towers[0][-1].bias is None


def test_class_stage_order_is_honoured():
    topology = replace(TOPO, class_stage_order="pw_dw")
    head = YOLO11Head(3, NANO_LEVELS, topology).eval()
    first = head.class_towers[0][0]
    assert first[0].conv.kernel_size == (1, 1)
    assert first[1].conv.groups == topology.class_width
    with torch.no_grad():
        assert head(features(1))[0].shape == (1, 67, 80, 80)


def test_rejects_mixed_batch_and_dtype():
    head = YOLO11Head(3, NANO_LEVELS, TOPO)
    mixed_batch = features(2)
    mixed_batch[1] = torch.randn(1, 128, 40, 40)
    with pytest.raises(ValueError, match="batch"):
        head(mixed_batch)

    mixed_dtype = features(2)
    mixed_dtype[1] = mixed_dtype[1].double()
    with pytest.raises(ValueError, match="dtype"):
        head(mixed_dtype)