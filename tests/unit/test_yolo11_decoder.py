import pytest
import torch

from libreyolo.models.yolo11.decoder import YOLO11nDecoder
from libreyolo.models.yolo11.transforms import restore_detections

pytestmark = pytest.mark.unit


def test_uniform_distributions_and_level_boundaries():
    sizes = [(24, 32), (12, 16), (6, 8)]
    levels = [torch.zeros(2, 66, h, w) for h, w in sizes]
    levels[0][1, 65, 20, 10] = torch.log(torch.tensor(3.0))
    decoded = YOLO11nDecoder()(levels)
    assert decoded.shape == (2, 6, 1008)
    site = 20 * 32 + 10
    torch.testing.assert_close(decoded[1, :, site], torch.tensor([24, 104, 144, 224, .5, .75]))
    assert decoded[0, 5, site] == .5
    for start, stride in [(0, 8), (768, 16), (960, 32)]:
        expected = torch.tensor([-7., -7., 8., 8.]) * stride
        torch.testing.assert_close(decoded[0, :4, start], expected)


def test_asymmetric_sides_through_detection_adapter():
    levels = [torch.full((1, 66, h, w), -80.) for h, w in [(8, 12), (4, 6), (2, 3)]]
    level = levels[0]
    for side, distance in enumerate((1, 2, 3, 4)):
        level[0, 16 * side + distance, 2, 4] = 80.
    level[0, 65, 2, 4] = 10.
    decoded = YOLO11nDecoder()(levels)
    result, = restore_detections(decoded, [(96, 64)], (64, 96))
    assert result["classes"] == [1]
    assert result["boxes"] == [[28, 4, 60, 52]]


def test_rejects_missing_level_or_missing_class_channels():
    decoder = YOLO11nDecoder()
    with pytest.raises(ValueError, match="three raw levels"):
        decoder([])
    with pytest.raises(ValueError, match="class logits"):
        decoder([torch.zeros(1, 64, 1, 1)] * 3)
