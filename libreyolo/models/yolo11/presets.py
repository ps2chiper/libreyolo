"""Named experimental topology presets for the YOLO11 family.

A preset fixes the values a caller would otherwise choose independently, so two
callers that use the same preset build the same model. These are LibreYOLO
design decisions, not YOLO11n parity: the repeat count, attention dimensions
and packing, positional and residual placement, tower widths and operator
order, and class-width rule remain open evidence questions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from numbers import Integral

from torch import nn

from .attention import C2PSA, PSATopology
from .detect import TowerTopology, YOLO11Head


@dataclass(frozen=True)
class YOLO11NPreset:
    """A reproducible, experiment-only nano configuration."""

    name: str
    version: str
    level_channels: tuple[int, int, int]
    stage_channels: int
    expansion: float
    blocks: int
    split_ratio: float
    attention: PSATopology
    towers: TowerTopology

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("A preset needs a name and a version")
        if len(self.level_channels) != 3:
            raise ValueError("A preset needs three level channel counts")
        if any(channels <= 0 for channels in self.level_channels):
            raise ValueError("Level channel counts must be positive")

    def build_c2psa(self) -> C2PSA:
        """Build the stage-10 attention block this preset configures."""
        return C2PSA(
            self.stage_channels,
            blocks=self.blocks,
            topology=self.attention,
            split_ratio=self.split_ratio,
            expansion=self.expansion,
        )

    def build_head(self, num_classes: int) -> YOLO11Head:
        """Build a fresh detection head with a 1% per-class foreground prior."""
        head = YOLO11Head(num_classes, self.level_channels, self.towers)
        for tower in head.class_towers:
            bias = tower[-1].bias
            if bias is not None:
                nn.init.constant_(bias, log(0.01 / 0.99))
        return head


EXPERIMENTAL_NANO = YOLO11NPreset(
    name="yolo11n-experimental",
    version="0.1.0",
    level_channels=(64, 128, 256),
    stage_channels=256,
    expansion=0.5,
    blocks=1,
    split_ratio=0.5,
    attention=PSATopology(
        heads=3,
        query_width=16,
        key_width=16,
        value_width=32,
        packing="head_major",
        position_kernel=3,
        ffn_kernel=1,
        ffn_expansion=2.0,
        residual="both",
    ),
    towers=TowerTopology(
        box_width=48,
        class_width=40,
        box_kernel=3,
        class_kernel=3,
        box_groups=1,
        pointwise_kernel=1,
        projection_kernel=1,
        class_stage_order="dw_pw",
        output_bias=True,
    ),
)


def build_experimental_nano(num_classes: int) -> tuple[C2PSA, YOLO11Head]:
    """Public entry point for the named experimental nano preset."""
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral):
        raise TypeError("num_classes must be an int")
    return (
        EXPERIMENTAL_NANO.build_c2psa(),
        EXPERIMENTAL_NANO.build_head(int(num_classes)),
    )
