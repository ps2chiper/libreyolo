"""Pixel-first acceptance tests for the experimental YOLO11n detector."""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.yolo11 import build_experimental_detector

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def detector():
    return build_experimental_detector(3).eval()


def test_image_tensor_reaches_raw_maps_and_decoder(detector):
    images = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        features = detector.forward_features(images)
        raw = detector.head(features)
        decoded = detector.decoder(raw)

    assert [tuple(level.shape) for level in features] == [
        (2, 64, 80, 80),
        (2, 128, 40, 40),
        (2, 256, 20, 20),
    ]
    assert [tuple(level.shape) for level in raw] == [
        (2, 67, 80, 80),
        (2, 67, 40, 40),
        (2, 67, 20, 20),
    ]
    assert tuple(decoded.shape) == (2, 7, 8400)
    assert torch.isfinite(decoded).all()


def test_rectangular_image_keeps_level_order_and_stride(detector):
    images = torch.randn(1, 3, 320, 640)
    with torch.no_grad():
        features = detector.forward_features(images)
        raw = detector.head(features)
        decoded = detector.decoder(raw)

    assert [tuple(level.shape) for level in features] == [
        (1, 64, 40, 80),
        (1, 128, 20, 40),
        (1, 256, 10, 20),
    ]
    assert [tuple(level.shape) for level in raw] == [
        (1, 67, 40, 80),
        (1, 67, 20, 40),
        (1, 67, 10, 20),
    ]
    assert tuple(decoded.shape) == (1, 7, 4200)


def test_forward_entry_point_matches_the_explicit_path(detector):
    images = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        raw = detector.forward_raw(images)
        decoded = detector(images)

    assert [tuple(level.shape) for level in raw] == [
        (1, 67, 8, 8),
        (1, 67, 4, 4),
        (1, 67, 2, 2),
    ]
    assert tuple(decoded.shape) == (1, 7, 84)
    torch.testing.assert_close(decoded, detector.decoder(raw))


def test_pixels_backpropagate_through_every_network_stage(detector):
    detector.zero_grad(set_to_none=True)
    images = torch.randn(1, 3, 64, 64, requires_grad=True)
    detector(images).sum().backward()

    stages = [
        detector.stem,
        detector.stage1,
        detector.down1,
        detector.stage2,
        detector.down2,
        detector.stage3,
        detector.down3,
        detector.stage4,
        detector.sppf,
        detector.attention,
        detector.neck_td1,
        detector.neck_p3,
        detector.neck_down1,
        detector.neck_p4,
        detector.neck_down2,
        detector.neck_p5,
    ]
    stages.extend(detector.head.box_towers)
    stages.extend(detector.head.class_towers)
    for stage in stages:
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in stage.parameters()
        ), f"No gradient reached {type(stage).__name__}"

    detector.zero_grad(set_to_none=True)


def _snapshot_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: buffer.detach().clone() for name, buffer in model.named_buffers()
    }


def _assert_buffers_unchanged(
    model: torch.nn.Module,
    snapshot: dict[str, torch.Tensor],
) -> None:
    for name, buffer in model.named_buffers():
        assert torch.equal(buffer, snapshot[name]), f"{name} changed"


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((1, 3, 641, 640), "multiples of 32"),
        ((1, 3, 640, 638), "multiples of 32"),
        ((0, 3, 64, 64), "at least one sample"),
        ((1, 3, 0, 64), "must be positive"),
        ((1, 1, 640, 640), "three channels"),
        ((3, 640, 640), "BCHW"),
    ],
)
def test_image_validation_rejects_malformed_inputs(detector, shape, message):
    with pytest.raises(ValueError, match=message):
        detector.forward_raw(torch.randn(*shape))


def test_training_singleton_is_rejected_before_buffers_change(detector):
    detector.train()
    try:
        snapshot = _snapshot_buffers(detector)
        with pytest.raises(ValueError, match="more than one value per channel"):
            detector.forward_features(torch.randn(1, 3, 32, 32))
        _assert_buffers_unchanged(detector, snapshot)
    finally:
        detector.eval()


def test_empty_batch_is_rejected_before_buffers_change(detector):
    detector.train()
    try:
        snapshot = _snapshot_buffers(detector)
        with pytest.raises(ValueError, match="at least one sample"):
            detector.forward_features(torch.empty(0, 3, 64, 64))
        _assert_buffers_unchanged(detector, snapshot)
    finally:
        detector.eval()


def test_small_rectangular_training_input_remains_valid(detector):
    detector.train()
    try:
        with torch.no_grad():
            features = detector.forward_features(torch.randn(1, 3, 32, 64))
        assert [tuple(level.shape) for level in features] == [
            (1, 64, 4, 8),
            (1, 128, 2, 4),
            (1, 256, 1, 2),
        ]
    finally:
        detector.eval()


def test_singleton_training_is_allowed_when_p5_normalization_is_frozen(detector):
    detector.train()
    try:
        p5_roots = (
            detector.down3,
            detector.stage4,
            detector.sppf,
            detector.attention,
            detector.neck_down2,
            detector.neck_p5,
            detector.head.box_towers[2],
            detector.head.class_towers[2],
        )
        for root in p5_roots:
            for module in root.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()

        with torch.no_grad():
            features = detector.forward_features(torch.randn(1, 3, 32, 32))
        assert [tuple(level.shape) for level in features] == [
            (1, 64, 4, 4),
            (1, 128, 2, 2),
            (1, 256, 1, 1),
        ]
    finally:
        detector.eval()


def test_singleton_input_remains_valid_in_evaluation(detector):
    images = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        features = detector.forward_features(images)
        decoded = detector(images)

    assert [tuple(level.shape) for level in features] == [
        (1, 64, 4, 4),
        (1, 128, 2, 2),
        (1, 256, 1, 1),
    ]
    assert tuple(decoded.shape) == (1, 7, 21)


def test_builder_rejects_a_non_integer_class_count():
    with pytest.raises(TypeError, match="num_classes"):
        build_experimental_detector(1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="num_classes"):
        build_experimental_detector(True)  # type: ignore[arg-type]