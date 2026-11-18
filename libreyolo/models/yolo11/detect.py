"""Detection-tower interface for the YOLO11 family.

The cited sources establish the three input levels, the two branches, the two
convolution stages in each, and the two depthwise convolutions in the
classification branch. Hidden widths, kernels, grouping, ordering, and output
bias are not established for the baseline nano detector. They are carried by
:class:`TowerTopology`, so this module makes no topology choice and no parity
claim.

The raw output contract is the native decoder's: ``64 + C`` channels per level,
with four consecutive 16-bin groups in left, top, right, bottom order followed
by one logit per class. Evidence is recorded in ``PORT.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..yolo9.nn import Conv, auto_pad


@dataclass(frozen=True)
class TowerTopology:
    """Caller-supplied choices for details the cited sources leave open.

    No field has a default. Supplying a topology defines an experimental
    variant; the cited papers do not establish these values for the baseline
    nano detector.
    """

    box_width: int
    class_width: int
    box_kernel: int
    class_kernel: int
    projection_kernel: int
    output_bias: bool

    def __post_init__(self) -> None:
        if self.box_width <= 0 or self.class_width <= 0:
            raise ValueError("Tower widths must be positive")
        for name in ("box_kernel", "class_kernel", "projection_kernel"):
            value = getattr(self, name)
            if value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")


class YOLO11Head(nn.Module):
    """Raw box and class towers over three pyramid levels.

    Each level produces ``4 * reg_max`` box-distribution logits followed by
    ``num_classes`` class logits, concatenated into one map. The box tower uses
    two convolutions and an output projection. The class tower uses two
    depthwise-convolution and pointwise-projection pairs, then an output
    projection.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: Sequence[int],
        topology: TowerTopology,
        *,
        reg_max: int = 16,
    ) -> None:
        super().__init__()
        in_channels = tuple(int(c) for c in in_channels)
        if len(in_channels) != 3:
            raise ValueError("YOLO11Head expects three input levels")
        if min(in_channels) <= 0:
            raise ValueError("Input channels must be positive")
        if int(num_classes) <= 0:
            raise ValueError("num_classes must be positive")
        if int(reg_max) <= 0:
            raise ValueError("reg_max must be positive")

        self.in_channels = in_channels
        self.nc = int(num_classes)
        self.nl = len(in_channels)
        self.reg_max = int(reg_max)
        self.no = 4 * self.reg_max + self.nc

        self.box_towers = nn.ModuleList(
            nn.Sequential(
                Conv(channels, topology.box_width, topology.box_kernel),
                Conv(topology.box_width, topology.box_width, topology.box_kernel),
                nn.Conv2d(
                    topology.box_width,
                    4 * self.reg_max,
                    topology.projection_kernel,
                    padding=auto_pad(topology.projection_kernel),
                    bias=topology.output_bias,
                ),
            )
            for channels in in_channels
        )
        self.class_towers = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(
                    Conv(channels, channels, topology.class_kernel, groups=channels),
                    Conv(channels, topology.class_width, topology.projection_kernel),
                ),
                nn.Sequential(
                    Conv(
                        topology.class_width,
                        topology.class_width,
                        topology.class_kernel,
                        groups=topology.class_width,
                    ),
                    Conv(
                        topology.class_width,
                        topology.class_width,
                        topology.projection_kernel,
                    ),
                ),
                nn.Conv2d(
                    topology.class_width,
                    self.nc,
                    topology.projection_kernel,
                    padding=auto_pad(topology.projection_kernel),
                    bias=topology.output_bias,
                ),
            )
            for channels in in_channels
        )

    def forward(self, features: Sequence[Tensor]) -> list[Tensor]:
        """Return one raw ``(B, 4 * reg_max + nc, H, W)`` map per level."""
        if len(features) != self.nl:
            raise ValueError(
                f"Expected {self.nl} feature levels, got {len(features)}"
            )
        outputs = []
        for level, channels, box, cls in zip(
            features, self.in_channels, self.box_towers, self.class_towers
        ):
            if level.ndim != 4 or level.shape[1] != channels:
                raise ValueError(
                    f"Expected a BCHW map with {channels} channels, "
                    f"got {tuple(level.shape)}"
                )
            outputs.append(torch.cat((box(level), cls(level)), dim=1))
        return outputs