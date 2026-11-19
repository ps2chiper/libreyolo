import math
from pathlib import Path

import pytest
import torch
from PIL import Image, ImageDraw

from libreyolo import LibreYOLO, LibreYOLO11Experimental

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("imgsz", [64, (64, 96)])
def test_train_validate_and_reload_synthetic_dataset(tmp_path, imgsz):
    for split, count in (("train", 4), ("val", 2)):
        images = tmp_path / "images" / split
        labels = tmp_path / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for i in range(count):
            image = Image.new("RGB", (80, 48), (20 + i, 30 if split == "train" else 60, 35))
            label = ""
            if i % 2 == 0:
                ImageDraw.Draw(image).rectangle((20, 12, 60, 36), fill="white")
                label = "0 0.5 0.5 0.5 0.5\n"
            image.save(images / f"{i}.png")
            (labels / f"{i}.txt").write_text(label)
    data = tmp_path / "data.yaml"
    data.write_text("train: images/train\nval: images/val\nnames: [square]\n")

    model = LibreYOLO11Experimental(device="cpu")

    class CaptureInitialWeight:
        def on_train_start(self, event):
            self.weight = model.model.stem[0].conv.weight.detach().clone()

        def on_train_epoch_end(self, event):
            assert not torch.equal(self.weight, model.model.stem[0].conv.weight)

    initial = CaptureInitialWeight()
    run = model.train(
        data=str(data), pretrained=False, epochs=1, batch=2, imgsz=imgsz,
        device="cpu", workers=0, amp=False, warmup_epochs=0,
        mosaic_prob=0.0, eval_interval=1, val_loss=True,
        project=str(tmp_path / "runs"), name="scratch", callbacks=[initial],
    )
    assert math.isfinite(run["final_loss"]) and run["final_loss"] > 0
    for component in ("box", "cls", "dfl"):
        assert run["epoch_loss_items"][0][component] > 0
        assert math.isfinite(run["val_metrics"][0][f"metrics/loss/{component}"])
    assert Path(run["last_checkpoint"]).is_file()
    restored = LibreYOLO(run["last_checkpoint"], device="cpu")
    assert restored.names == {0: "square"}
    assert restored.nb_classes == 1
    assert not torch.equal(initial.weight, restored.model.stem[0].conv.weight)
    metrics = restored.val(
        data=str(data), imgsz=imgsz, batch=2, workers=0, device="cpu",
        project=str(tmp_path / "runs"), name="validation", plots=False,
    )
    assert 0 <= metrics["metrics/mAP50"] <= 1
    assert 0 <= metrics["metrics/mAP50-95"] <= 1
