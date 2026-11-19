import numpy as np
import pytest
import torch
from PIL import Image

from libreyolo import LibreYOLO, LibreYOLO11Experimental

pytestmark = pytest.mark.unit


def test_public_prediction_and_checkpoint_round_trip(tmp_path):
    model = LibreYOLO11Experimental(nb_classes=3, device="cpu")
    model.names = {0: "first", 1: "second", 2: "third"}
    image = Image.fromarray(np.arange(48 * 80 * 3, dtype=np.uint8).reshape(48, 80, 3))
    options = {"imgsz": 64, "conf": 0.0, "max_det": 5}
    result = model.predict(image, **options)
    assert result.orig_shape == (48, 80)
    assert 0 < len(result.boxes) <= 5
    path = tmp_path / "custom.pt"
    model.save(str(path))
    restored = LibreYOLO(str(path), device="cpu")
    assert type(restored) is LibreYOLO11Experimental
    assert (restored.size, restored.task, restored.nb_classes) == ("n", "detect", 3)
    assert restored.names == model.names
    torch.testing.assert_close(
        restored.predict(image, **options).boxes.data, result.boxes.data, atol=0, rtol=0
    )
    restored_state = restored.model.state_dict()
    for key, tensor in model.model.state_dict().items():
        torch.testing.assert_close(restored_state[key], tensor, atol=0, rtol=0)

    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint["preset_name"] == "yolo11n-experimental"
    assert checkpoint["preset_version"] == "0.1.0"
    assert checkpoint["letterbox_pad"] == "center"
    checkpoint["preset_version"] = "unknown"
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="preset_version"):
        LibreYOLO(str(path), device="cpu")


def test_checkpoint_letterbox_metadata_backward_compatibility(tmp_path):
    model = LibreYOLO11Experimental(nb_classes=3, device="cpu")
    path = tmp_path / "legacy.pt"
    model.save(str(path))

    checkpoint = torch.load(path, weights_only=True)
    del checkpoint["letterbox_pad"]
    torch.save(checkpoint, path)
    restored = LibreYOLO(str(path), device="cpu")
    assert type(restored) is LibreYOLO11Experimental
    assert restored.letterbox_pad == "center"

    checkpoint = torch.load(path, weights_only=True)
    checkpoint["letterbox_pad"] = "topleft"
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="letterbox_pad"):
        LibreYOLO(str(path), device="cpu")


def test_public_prediction_rounds_input_to_stride():
    model = LibreYOLO11Experimental(nb_classes=2, device="cpu")
    image = Image.fromarray(np.zeros((48, 80, 3), dtype=np.uint8))
    result = model.predict(image, imgsz=100, conf=0.0, max_det=1)

    assert result.orig_shape == (48, 80)
    assert len(result.boxes) == 1


def test_public_prediction_on_mixed_aspect_batch():
    model = LibreYOLO11Experimental(nb_classes=2, device="cpu")
    images = [Image.new("RGB", (80, 48)), Image.new("RGB", (32, 64))]
    results = list(model.predict(images, imgsz=64, batch=2, conf=0.0, max_det=3))
    assert [result.orig_shape for result in results] == [(48, 80), (64, 32)]
    for result in results:
        assert 0 < len(result.boxes) <= 3
        height, width = result.orig_shape
        xyxy = result.boxes.xyxy
        assert (xyxy >= 0).all()
        assert (xyxy[:, ::2] <= width).all()
        assert (xyxy[:, 1::2] <= height).all()
