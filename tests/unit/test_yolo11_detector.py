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


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((1, 3, 641, 640), "multiples of 32"),
        ((1, 3, 640, 638), "multiples of 32"),
        ((1, 1, 640, 640), "three channels"),
        ((3, 640, 640), "BCHW"),
    ],
)
def test_image_validation_rejects_malformed_inputs(detector, shape, message):
    with pytest.raises(ValueError, match=message):
        detector.forward_raw(torch.randn(*shape))


def test_builder_rejects_a_non_integer_class_count():
    with pytest.raises(TypeError, match="num_classes"):
        build_experimental_detector(1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="num_classes"):
        build_experimental_detector(True)  # type: ignore[arg-type]