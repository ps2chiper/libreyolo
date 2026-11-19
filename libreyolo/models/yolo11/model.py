"""Public inference wrapper for LibreYOLO's experimental nano detector."""

from functools import partial
from pathlib import Path

from ...postprocess.yolo9 import postprocess
from ...preprocess.letterbox import LETTERBOX_CENTER
from ...preprocess.yolo9 import preprocess_image, preprocess_numpy
from ..base import BaseModel
from .detector import build_experimental_detector


class LibreYOLO11Experimental(BaseModel):
    """Run the versioned nano preset on RGB images using centered letterboxing."""

    FAMILY = "yolo11_experimental"
    FILENAME_PREFIX = "LibreYOLO11Experimental"
    INPUT_SIZES = {"n": 640}
    UNPUBLISHED_WEIGHTS = {
        "LibreYOLO11Experimentaln": "Initialize the experimental preset from scratch."
    }

    def __init__(self, model_path=None, size="n", nb_classes=80, device="auto", task=None, **kwargs):
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
        }

    def _validate_loaded_state_dict_for_task(self, state_dict, checkpoint=None):
        if checkpoint is not None and "model_family" in checkpoint:
            for key, expected in self._save_extra_metadata().items():
                if checkpoint.get(key) != expected:
                    raise ValueError(
                        f"Unsupported experimental checkpoint {key}: "
                        f"expected {expected!r}, got {checkpoint.get(key)!r}"
                    )

    @staticmethod
    def _get_preprocess_numpy():
        return partial(preprocess_numpy, letterbox_pad=LETTERBOX_CENTER)

    def _preprocess(self, image, color_format="auto", input_size=None):
        tensor, original, size = preprocess_image(
            image, self.input_size if input_size is None else input_size,
            color_format=color_format, letterbox_pad=LETTERBOX_CENTER,
        )
        return tensor, original, size, 1.0

    def _forward(self, input_tensor):
        return {"predictions": self.model(input_tensor)}

    def _postprocess(self, output, conf_thres, iou_thres, original_size, max_det=300, ratio=1.0, **kwargs):
        return postprocess(
            output, conf_thres=conf_thres, iou_thres=iou_thres, original_size=original_size,
            max_det=max_det, input_size=kwargs.get("input_size", self.input_size),
            letterbox=kwargs.get("letterbox", True), letterbox_pad=LETTERBOX_CENTER,
        )
