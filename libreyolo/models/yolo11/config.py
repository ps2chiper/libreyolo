from dataclasses import dataclass

from ...training.config import YOLO9Config


@dataclass(kw_only=True)
class ExperimentalNanoConfig(YOLO9Config):
    """Use the native dense detection recipe with the nano preset's geometry."""

    size: str = "n"
    name: str = "yolo11_experimental"
    letterbox_pad: str = "center"
    aux_weight: float = 0.0
