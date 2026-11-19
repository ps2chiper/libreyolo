"""Public training and inference wrapper for LibreYOLO's experimental nano detector."""

from functools import partial
from pathlib import Path
from typing import ClassVar

from ...postprocess.yolo9 import postprocess
from ...preprocess.letterbox import LETTERBOX_CENTER
from ...preprocess.yolo9 import preprocess_image, preprocess_numpy
from ...training.ddp_spawn import ddp_aware
from ...utils.image_size import imgsz_to_hw
from ...validation.preprocessors import YOLO9ValPreprocessor
from ..base import BaseModel
from .config import ExperimentalNanoConfig
from .detector import build_experimental_detector


class LibreYOLO11Experimental(BaseModel):
    """Run the versioned nano preset on RGB images using centered letterboxing."""

    FAMILY = "yolo11_experimental"
    FILENAME_PREFIX = "LibreYOLO11Experimental"
    INPUT_SIZES: ClassVar[dict[str, int]] = {"n": 640}
    IMGSZ_STRIDE = 32
    TRAIN_CONFIG = ExperimentalNanoConfig
    val_preprocessor_class = YOLO9ValPreprocessor
    LETTERBOX_PAD = LETTERBOX_CENTER
    UNPUBLISHED_WEIGHTS: ClassVar[dict[str, str]] = {
        "LibreYOLO11Experimentaln": "Initialize the experimental preset from scratch."
    }

    def __init__(self, model_path=None, size="n", nb_classes=80, device="auto", task=None, **kwargs):
        self.letterbox_pad = self.LETTERBOX_PAD
        if isinstance(model_path, Path):
            model_path = str(model_path)
        super().__init__(model_path, size, nb_classes, device, task, **kwargs)
        if isinstance(model_path, str):
            self._load_weights(self.model_path)
        self.model.eval()

    @classmethod
    def can_load(cls, weights_dict):
        signature = (
            "neck_td1.cv1.conv.weight",
            "attention.m.0.attn.qkv.conv.weight",
            "head.box_towers.2.2.weight",
            "head.class_towers.2.2.weight",
        )
        return all(key in weights_dict for key in signature)

    @classmethod
    def detect_size(cls, weights_dict):
        return "n" if cls.can_load(weights_dict) else None

    @classmethod
    def detect_nb_classes(cls, weights_dict):
        weight = weights_dict.get("head.class_towers.2.2.weight")
        return weight.shape[0] if weight is not None else None

    def _init_model(self):
        return build_experimental_detector(self.nb_classes)

    def _get_available_layers(self):
        return dict(self.model.named_children())

    def _save_extra_metadata(self):
        return {
            "preset_name": self.model.preset.name,
            "preset_version": self.model.preset.version,
            "letterbox_pad": self.letterbox_pad,
        }

    def _validate_loaded_state_dict_for_task(self, state_dict, checkpoint=None):
        if checkpoint is not None and "model_family" in checkpoint:
            for key, expected in self._save_extra_metadata().items():
                if key not in checkpoint:
                    if key == "letterbox_pad":
                        # Checkpoints saved before letterbox metadata used the
                        # same centered geometry.
                        continue
                    actual = None
                else:
                    actual = checkpoint[key]
                if actual != expected:
                    raise ValueError(
                        f"Unsupported experimental checkpoint {key}: "
                        f"expected {expected!r}, got {actual!r}"
                    )

    @staticmethod
    def _get_preprocess_numpy():
        return partial(preprocess_numpy, letterbox_pad=LETTERBOX_CENTER)

    def _preprocess(self, image, color_format="auto", input_size=None):
        tensor, original, size = preprocess_image(
            image, self.input_size if input_size is None else input_size,
            color_format=color_format, letterbox_pad=self.letterbox_pad,
        )
        return tensor, original, size, 1.0

    def _forward(self, input_tensor):
        raw = self.model(input_tensor, return_raw=True)
        return {"predictions": self.model.decoder(raw), "raw_outputs": raw}

    def _get_val_preprocessor(self, img_size=None):
        if getattr(self, "input_profile", None) is not None:
            return super()._get_val_preprocessor(img_size)
        return self.val_preprocessor_class(
            img_size=imgsz_to_hw(self.input_size if img_size is None else img_size),
            letterbox_pad=self.letterbox_pad,
        )

    @ddp_aware()
    def train(self, data: str, *, device="", resume=False, seed=0, **kwargs):
        """Train on a LibreYOLO detection dataset. Use pretrained=False for a fresh run."""
        from ...data import load_data_config
        from .trainer import ExperimentalNanoTrainer

        dataset = load_data_config(
            data, autodownload=True,
            allow_scripts=kwargs.get("allow_download_scripts", False),
        )
        names = dataset.get("names")
        nc = int(dataset.get("nc", len(names) if names is not None else self.nb_classes))
        if nc != self.nb_classes:
            self._rebuild_for_new_classes(nc)
        if names is not None:
            if isinstance(names, list):
                names = dict(enumerate(names))
            self.names = self._sanitize_names(names, nc)
        resume_path = self._resume_checkpoint(resume) if resume else None
        trainer = ExperimentalNanoTrainer(
            model=self.model, wrapper_model=self, num_classes=nc, size=self.size,
            data=dataset.get("yaml_file", data), device=device or str(self.device),
            resume=bool(resume_path), seed=seed, **kwargs,
        )
        if resume_path:
            trainer.resume_source = resume_path
            trainer.setup()
            trainer.resume(resume_path)
        results = trainer.train()
        checkpoint = results.get("best_checkpoint") or results.get("last_checkpoint")
        if checkpoint:
            self._load_weights(str(checkpoint))
            self.model_path = str(checkpoint)
        self.model.eval()
        return results

    def _postprocess(self, output, conf_thres, iou_thres, original_size, max_det=300, ratio=1.0, **kwargs):
        return postprocess(
            output, conf_thres=conf_thres, iou_thres=iou_thres, original_size=original_size,
            max_det=max_det, input_size=kwargs.get("input_size", self.input_size),
            letterbox=kwargs.get("letterbox", True), letterbox_pad=self.letterbox_pad,
        )
