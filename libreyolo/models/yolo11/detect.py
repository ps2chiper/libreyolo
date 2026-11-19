"""Detection-tower interface for the YOLO11 family.

The cited sources establish the three input levels, the two branches, the two
convolution stages in each, and the two depthwise convolutions in the
classification branch. Widths, kernels, box grouping, stage order, projection
kernel and output bias are not established for the baseline nano detector.
They are carried by :class:`TowerTopology`.

The two-stage arrangement is sourced. The stage wiring, widths, kernels and
grouping are LibreYOLO design decisions, as is the use of the native ``Conv``
block for normalization and activation. This module makes no parity claim.
Initialization is out of scope.

The raw output contract is the native decoder's: ``64 + C`` channels per level,
with four consecutive 16-bin groups in left, top, right, bottom order followed
by one logit per class. Evidence is recorded in ``PORT.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import torch
import torch.distributed as dist
from torch import Tensor, nn

from ..yolo9.nn import Conv, auto_pad

_STAGE_ORDERS = ("dw_pw", "pw_dw")
REG_MAX = 16


@dataclass(frozen=True)
class TowerTopology:
    """Caller-supplied choices for details the cited sources leave open.

    No field has a default. Supplying a topology defines an experimental
    variant; the cited papers do not establish these values for the baseline
    nano detector. Invalid ranges raise ``ValueError``; a non-bool
    ``output_bias`` raises ``TypeError``.
    """

    box_width: int
    class_width: int
    box_kernel: int
    class_kernel: int
    box_groups: int
    pointwise_kernel: int
    projection_kernel: int
    class_stage_order: str
    output_bias: bool

    def __post_init__(self) -> None:
        for name in ("box_width", "class_width", "box_groups"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        for name in ("box_kernel", "class_kernel", "pointwise_kernel", "projection_kernel"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value <= 0
                or value % 2 == 0
            ):
                raise ValueError(f"{name} must be a positive odd integer")
            object.__setattr__(self, name, int(value))
        if self.class_stage_order not in _STAGE_ORDERS:
            raise ValueError(f"class_stage_order must be one of {_STAGE_ORDERS}")
        if not isinstance(self.output_bias, bool):
            raise TypeError("output_bias must be a bool")


class YOLO11Head(nn.Module):
    """Raw box and class towers over three pyramid levels.

    Each level produces 64 box-distribution logits followed by
    ``num_classes`` class logits, concatenated into one map. The box tower uses
    two convolutions and an output projection. The class tower uses two
    depthwise-convolution and pointwise-projection stages, in the caller's
    order, then an output projection.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: Sequence[int],
        topology: TowerTopology,
    ) -> None:
        super().__init__()
        in_channels = tuple(in_channels)
        if len(in_channels) != 3:
            raise ValueError("YOLO11Head expects three input levels")
        if any(
            isinstance(c, bool) or not isinstance(c, Integral) or c <= 0
            for c in in_channels
        ):
            raise ValueError("Input channels must be positive integers")
        in_channels = tuple(int(c) for c in in_channels)
        if (
            isinstance(num_classes, bool)
            or not isinstance(num_classes, Integral)
            or num_classes <= 0
        ):
            raise ValueError("num_classes must be a positive integer")
        num_classes = int(num_classes)
        for channels in in_channels:
            if (
                channels % topology.box_groups
                or topology.box_width % topology.box_groups
            ):
                raise ValueError(
                    "box_groups must divide every input width and box_width"
                )

        self.in_channels = in_channels
        self.nc = num_classes
        self.nl = len(in_channels)
        self.reg_max = REG_MAX
        self.no = 4 * self.reg_max + self.nc

        self.box_towers = nn.ModuleList(
            nn.Sequential(
                Conv(
                    channels,
                    topology.box_width,
                    topology.box_kernel,
                    groups=topology.box_groups,
                ),
                Conv(
                    topology.box_width,
                    topology.box_width,
                    topology.box_kernel,
                    groups=topology.box_groups,
                ),
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
                self._class_stage(channels, topology.class_width, topology),
                self._class_stage(
                    topology.class_width, topology.class_width, topology
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

    @staticmethod
    def _class_stage(
        in_channels: int, out_channels: int, topology: TowerTopology
    ) -> nn.Sequential:
        """Depthwise and pointwise pair in the caller's order."""
        if topology.class_stage_order == "dw_pw":
            return nn.Sequential(
                Conv(
                    in_channels,
                    in_channels,
                    topology.class_kernel,
                    groups=in_channels,
                ),
                Conv(in_channels, out_channels, topology.pointwise_kernel),
            )
        return nn.Sequential(
            Conv(in_channels, out_channels, topology.pointwise_kernel),
            Conv(
                out_channels,
                out_channels,
                topology.class_kernel,
                groups=out_channels,
            ),
        )

    @staticmethod
    def _sync_world_size(module) -> int:
        """World size for this layer's process group, or 1 when sync is off."""
        if not (dist.is_available() and dist.is_initialized()):
            return 1
        try:
            return dist.get_world_size(getattr(module, "process_group", None))
        except (RuntimeError, ValueError):
            return 1

    def _batch_norm_needs_batch_statistics(self, module) -> bool:
        """Whether this layer normalizes with local batch statistics.

        A SyncBatchNorm layer counts as synchronized only when its process
        group spans more than one rank. That allowance assumes the input device
        supports synchronized BatchNorm; CPU-only PyTorch builds reject
        SyncBatchNorm inputs before any statistics are synchronized.
        """
        if module.training:
            is_synchronized = (
                isinstance(module, nn.SyncBatchNorm)
                and self._sync_world_size(module) > 1
            )
            return not is_synchronized
        return module.running_mean is None and module.running_var is None

    def _uses_batch_statistics(self, level: int) -> bool:
        """Whether any BatchNorm in this level's towers needs local batch statistics."""
        for tower in (self.box_towers[level], self.class_towers[level]):
            for module in tower.modules():
                if isinstance(
                    module, (nn.BatchNorm2d, nn.SyncBatchNorm)
                ) and self._batch_norm_needs_batch_statistics(module):
                    return True
        return False

    def forward(self, features: Sequence[Tensor]) -> list[Tensor]:
        """Return one raw ``(B, 64 + nc, H, W)`` map per level.

        All levels are validated before any tower runs. A singleton level
        (``N*H*W <= 1``) is accepted only when that level's BatchNorm layers
        use running statistics or synchronize statistics across ranks.
        """
        if len(features) != self.nl:
            raise ValueError(
                f"Expected {self.nl} feature levels, got {len(features)}"
            )
        batch = dtype = None
        for index, (level, channels) in enumerate(
            zip(features, self.in_channels)
        ):
            if level.ndim != 4 or level.shape[1] != channels:
                raise ValueError(
                    f"Expected a BCHW map with {channels} channels, "
                    f"got {tuple(level.shape)}"
                )
            if not level.is_floating_point():
                raise ValueError("Feature levels must be floating point")
            if level.shape[0] <= 0:
                raise ValueError("Feature levels need a positive batch size")
            if level.shape[2] <= 0 or level.shape[3] <= 0:
                raise ValueError(
                    "Feature levels need positive height and width"
                )
            if level.device.type == "cpu" and dist.is_available() and dist.is_initialized():
                for tower in (self.box_towers[index], self.class_towers[index]):
                    if any(
                        isinstance(module, nn.SyncBatchNorm) and module.training
                        for module in tower.modules()
                    ):
                        raise ValueError("Distributed training with SyncBatchNorm needs an accelerator")
            if (
                level.shape[0] * level.shape[2] * level.shape[3] <= 1
                and self._uses_batch_statistics(index)
            ):
                raise ValueError(
                    "BatchNorm with batch statistics needs more than one value "
                    "per channel; freeze this level's BatchNorm or use a larger map"
                )
            if index == 0:
                batch = level.shape[0]
                dtype = level.dtype
            elif level.shape[0] != batch:
                raise ValueError("All feature levels must share a batch size")
            elif level.dtype != dtype:
                raise ValueError("All feature levels must share a dtype")

        # No tower runs until every level has passed validation, so a rejected
        # batch cannot update BatchNorm statistics partway through.
        outputs = []
        for level, box, cls in zip(features, self.box_towers, self.class_towers):
            outputs.append(torch.cat((box(level), cls(level)), dim=1))
        return outputs
