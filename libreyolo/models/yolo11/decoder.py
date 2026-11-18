"""Decode raw YOLO11n detection levels with LibreYOLO's native box decoder."""

from collections.abc import Sequence

from torch import Tensor

from ..yolo9.nn import DistributionBoxDecoder


class YOLO11nDecoder(DistributionBoxDecoder):
    """Accept P3, P4, P5 maps in BCHW order, each with 64 box logits then class logits."""

    def __init__(self):
        super().__init__(reg_max=16, strides=(8, 16, 32))

    def forward(self, levels: Sequence[Tensor]) -> Tensor:
        """Return canvas-pixel xyxy and all class probabilities as (B, 4 + C, N)."""
        if len(levels) != len(self.strides):
            raise ValueError("Expected three raw levels in P3, P4, P5 order")
        for level in levels:
            if level.ndim != 4 or level.shape[1] <= 64 or not level.is_floating_point():
                raise ValueError("Expected floating BCHW maps with 64 box logits and class logits")
            if level.shape[:2] != levels[0].shape[:2]:
                raise ValueError("All levels must share the batch size and class count")
        return self.decode(levels)
