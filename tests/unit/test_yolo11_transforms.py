import numpy as np
import pytest
import torch

from libreyolo.models.yolo11.transforms import prepare_batch, restore_detections

pytestmark = pytest.mark.unit


def test_centered_letterbox_round_trip_for_mixed_aspect_batch():
    images = [
        np.full((8, 16, 3), (255, 0, 0), np.uint8),
        np.full((16, 8, 3), (0, 255, 0), np.uint8),
    ]
    batch, sizes = prepare_batch(images, (32, 32))
    assert sizes == [(16, 8), (8, 16)]
    assert batch.shape == (2, 3, 32, 32)
    torch.testing.assert_close(batch[0, :, 8, 0], torch.tensor([1., 0., 0.]))
    torch.testing.assert_close(batch[1, :, 0, 8], torch.tensor([0., 1., 0.]))
    torch.testing.assert_close(batch[:, :, 0, 0], torch.full((2, 3), 114 / 255))

    # Each image has one foreground box spanning its unpadded portion.
    predictions = torch.tensor([[0, 8, 32, 24, .9], [8, 0, 24, 32, .8]]).unsqueeze(2)
    results = restore_detections(predictions, sizes, (32, 32))
    assert [item["boxes"] for item in results] == [[[0, 0, 16, 8]], [[0, 0, 8, 16]]]
    assert [item["classes"] for item in results] == [[0], [0]]


def test_per_image_class_aware_suppression_and_empty_result():
    samples = torch.tensor([[0, 0, 8, 8, .9, .1], [0, 0, 8, 8, .8, .7]]).T
    results = restore_detections(torch.stack((samples, samples * 0)), [(8, 8)] * 2, 8)
    assert results[0]["classes"] == [0, 1]
    assert results[1]["num_detections"] == 0


def test_rejects_mismatched_batch_metadata():
    with pytest.raises(ValueError, match="original image size"):
        restore_detections(torch.zeros(2, 5, 1), [(8, 8)], 8)
