"""Experimental YOLO11n detector assembly.

The stem, stage routing, widths and repeats follow Saeheaw, Algorithm A1, lines
2 to 41. The C3k2 and SPPF implementations are clean-room LibreYOLO code built
from the native convolution block; their inner split ratio, bottleneck form and
upsampling mode are experimental choices recorded in ``PORT.md``. The attention
block and detection head come from the named preset, and decoding uses the
native ``YOLO11nDecoder``.
"""

from __future__ import annotations

from numbers import Integral

import torch
from torch import Tensor, nn
from torch.nn.modules.batchnorm import _BatchNorm

from ..yolo9.nn import Conv
from .decoder import YOLO11nDecoder
from .detect import YOLO11Head
from .normalization import (
    batch_norm_needs_batch_statistics,
    sync_batch_norm_needs_accelerator,
)
from .presets import EXPERIMENTAL_NANO, YOLO11NPreset

# Algorithm A1, YOLOv11n: stem widths, stage widths and stage repeats.
STEM_CHANNELS = (16, 32)
STAGE_CHANNELS = (64, 128, 128, 256)
STAGE_SHORTCUTS = (False, False, True, True)
NECK_CHANNELS = (128, 64, 128, 256)
C3K2_EXPANSION = 0.5


class Bottleneck11(nn.Module):
    """Two 3x3 convolutions with an optional residual addition."""

    def __init__(self, channels: int, shortcut: bool) -> None:
        super().__init__()
        self.cv1 = Conv(channels, channels, 3)
        self.cv2 = Conv(channels, channels, 3)
        self.add = bool(shortcut)

    def forward(self, x: Tensor) -> Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k2(nn.Module):
    """Split, run ``n`` bottlenecks, concatenate every branch, then project.

    ``expansion`` fixes the split width at ``int(out_channels * expansion)``.
    The source graph fixes the output width and repeat count; the split ratio
    and bottleneck form are experimental choices.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        blocks: int,
        shortcut: bool,
        expansion: float = C3K2_EXPANSION,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("C3k2 channels must be positive")
        if blocks < 1:
            raise ValueError("C3k2 needs at least one block")
        inner = int(out_channels * expansion)
        if inner <= 0:
            raise ValueError("C3k2 expansion leaves no inner channels")
        self.inner = inner
        self.cv1 = Conv(in_channels, 2 * inner, 1)
        self.cv2 = Conv((2 + blocks) * inner, out_channels, 1)
        self.m = nn.ModuleList(
            Bottleneck11(inner, shortcut) for _ in range(blocks)
        )

    def forward(self, x: Tensor) -> Tensor:
        branches = list(self.cv1(x).split((self.inner, self.inner), dim=1))
        for block in self.m:
            branches.append(block(branches[-1]))
        return self.cv2(torch.cat(branches, dim=1))


class SPPF(nn.Module):
    """Three serial 5x5 max pools over a half-width projection."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int = 5) -> None:
        super().__init__()
        hidden = in_channels // 2
        self.cv1 = Conv(in_channels, hidden, 1)
        self.cv2 = Conv(4 * hidden, out_channels, 1)
        self.pool = nn.MaxPool2d(kernel_size=kernel, stride=1, padding=kernel // 2)

    def forward(self, x: Tensor) -> Tensor:
        branches = [self.cv1(x)]
        for _ in range(3):
            branches.append(self.pool(branches[-1]))
        return self.cv2(torch.cat(branches, dim=1))


class YOLO11Detector(nn.Module):
    """Image tensor to decoded detections for the experimental nano graph."""

    def __init__(
        self,
        num_classes: int,
        preset: YOLO11NPreset = EXPERIMENTAL_NANO,
    ) -> None:
        super().__init__()
        if preset.stage_channels != STAGE_CHANNELS[-1]:
            raise ValueError("The detector graph requires a 256-channel stage 10")
        if preset.level_channels != (64, 128, 256):
            raise ValueError("The detector graph requires 64/128/256 head levels")
        self.preset = preset

        self.stem = nn.Sequential(
            Conv(3, STEM_CHANNELS[0], 3, stride=2),
            Conv(STEM_CHANNELS[0], STEM_CHANNELS[1], 3, stride=2),
        )
        self.stage1 = C3k2(
            STEM_CHANNELS[1],
            STAGE_CHANNELS[0],
            blocks=1,
            shortcut=STAGE_SHORTCUTS[0],
        )
        self.down1 = Conv(STAGE_CHANNELS[0], STAGE_CHANNELS[0], 3, stride=2)
        self.stage2 = C3k2(
            STAGE_CHANNELS[0],
            STAGE_CHANNELS[1],
            blocks=1,
            shortcut=STAGE_SHORTCUTS[1],
        )
        self.down2 = Conv(STAGE_CHANNELS[1], STAGE_CHANNELS[1], 3, stride=2)
        self.stage3 = C3k2(
            STAGE_CHANNELS[1],
            STAGE_CHANNELS[2],
            blocks=1,
            shortcut=STAGE_SHORTCUTS[2],
        )
        self.down3 = Conv(STAGE_CHANNELS[2], STAGE_CHANNELS[3], 3, stride=2)
        self.stage4 = C3k2(
            STAGE_CHANNELS[3],
            STAGE_CHANNELS[3],
            blocks=1,
            shortcut=STAGE_SHORTCUTS[3],
        )
        self.sppf = SPPF(STAGE_CHANNELS[3], STAGE_CHANNELS[3])
        self.attention = preset.build_c2psa()

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.neck_td1 = C3k2(
            256 + 128,
            NECK_CHANNELS[0],
            blocks=2,
            shortcut=False,
        )
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.neck_p3 = C3k2(
            128 + 128,
            NECK_CHANNELS[1],
            blocks=2,
            shortcut=False,
        )
        self.neck_down1 = Conv(NECK_CHANNELS[1], NECK_CHANNELS[1], 3, stride=2)
        self.neck_p4 = C3k2(
            NECK_CHANNELS[1] + NECK_CHANNELS[0],
            NECK_CHANNELS[2],
            blocks=2,
            shortcut=False,
        )
        self.neck_down2 = Conv(NECK_CHANNELS[2], NECK_CHANNELS[2], 3, stride=2)
        self.neck_p5 = C3k2(
            NECK_CHANNELS[2] + STAGE_CHANNELS[3],
            NECK_CHANNELS[3],
            blocks=2,
            shortcut=True,
        )

        self.head: YOLO11Head = preset.build_head(num_classes)
        self.decoder = YOLO11nDecoder()

    def _validate_images(self, images: Tensor) -> tuple[int, int, int]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected a BCHW image tensor with three channels, "
                f"got {tuple(images.shape)}"
            )
        if not images.is_floating_point():
            raise ValueError("Image input must be floating point")
        batch = images.shape[0]
        if batch < 1:
            raise ValueError("Image batch must contain at least one sample")
        height, width = images.shape[2], images.shape[3]
        if height <= 0 or width <= 0:
            raise ValueError("Image height and width must be positive")
        if height % 32 or width % 32:
            raise ValueError("Image height and width must be multiples of 32")
        return batch, height, width

    def _check_normalization_preflight(
        self,
        batch: int,
        height: int,
        width: int,
        *,
        include_head: bool,
    ) -> None:
        """Validate the modules this entry point will execute before any conv."""
        feature_roots = (
            self.stem,
            self.stage1,
            self.down1,
            self.stage2,
            self.down2,
            self.stage3,
            self.down3,
            self.stage4,
            self.sppf,
            self.attention,
            self.neck_td1,
            self.neck_p3,
            self.neck_down1,
            self.neck_p4,
            self.neck_down2,
            self.neck_p5,
        )
        roots = feature_roots + ((self.head,) if include_head else ())
        for root in roots:
            for module in root.modules():
                if sync_batch_norm_needs_accelerator(module):
                    raise ValueError(
                        "Distributed training with SyncBatchNorm needs an "
                        "accelerator"
                    )

        minimum_sites = batch * (height // 32) * (width // 32)
        if minimum_sites > 1:
            return
        p5_roots = (
            self.down3,
            self.stage4,
            self.sppf,
            self.attention,
            self.neck_down2,
            self.neck_p5,
        )
        if include_head:
            p5_roots += (self.head.box_towers[2], self.head.class_towers[2])
        for root in p5_roots:
            for module in root.modules():
                if isinstance(module, _BatchNorm) and batch_norm_needs_batch_statistics(
                    module
                ):
                    raise ValueError(
                        "Training images must give every normalization layer more "
                        "than one value per channel; increase the batch or spatial "
                        "size, or put the affected layers in evaluation mode"
                    )

    def _run_features(self, images: Tensor) -> list[Tensor]:
        x = self.stem(images)
        x = self.stage1(x)
        x = self.down1(x)
        f4 = self.stage2(x)
        x = self.down2(f4)
        f6 = self.stage3(x)
        x = self.down3(f6)
        x = self.stage4(x)
        x = self.sppf(x)
        f10 = self.attention(x)

        x = self.up1(f10)
        td1 = self.neck_td1(torch.cat((x, f6), dim=1))
        p3 = self.neck_p3(torch.cat((self.up2(td1), f4), dim=1))
        x = self.neck_down1(p3)
        p4 = self.neck_p4(torch.cat((x, td1), dim=1))
        x = self.neck_down2(p4)
        p5 = self.neck_p5(torch.cat((x, f10), dim=1))
        return [p3, p4, p5]

    def forward_features(self, images: Tensor) -> list[Tensor]:
        """Return the neck's P3, P4 and P5 feature maps for an image tensor."""
        batch, height, width = self._validate_images(images)
        self._check_normalization_preflight(
            batch,
            height,
            width,
            include_head=False,
        )
        return self._run_features(images)

    def forward_raw(self, images: Tensor) -> list[Tensor]:
        """Run the image tensor through the network and return raw head maps."""
        batch, height, width = self._validate_images(images)
        self._check_normalization_preflight(
            batch,
            height,
            width,
            include_head=True,
        )
        return self.head(self._run_features(images))

    def forward(self, images: Tensor) -> Tensor:
        """Return canvas-pixel xyxy boxes and class probabilities."""
        return self.decoder(self.forward_raw(images))


def build_experimental_detector(
    num_classes: int,
    preset: YOLO11NPreset = EXPERIMENTAL_NANO,
) -> YOLO11Detector:
    """Public entry point for the experimental nano image-to-detections model."""
    if isinstance(num_classes, bool) or not isinstance(num_classes, Integral):
        raise TypeError("num_classes must be an int")
    return YOLO11Detector(int(num_classes), preset)