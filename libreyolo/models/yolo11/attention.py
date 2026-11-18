"""C2PSA interface for the YOLO11 family.

The cited sources establish the outer structure only: stage 10 receives the
SPPF output and returns the same shape, the outer expansion is 0.5, and the
block is an initial convolution, a sequence of PSA blocks, a concatenation and
a final convolution. PSA blocks carry residual connections.

Widths, packing, split ratio, position kernel, feed-forward kernel and
dimension, and residual use are caller decisions carried by
:class:`PSATopology` and the ``C2PSA`` arguments. The two-convolution
feed-forward form and the use of the native ``Conv`` block for normalization
and activation are LibreYOLO design decisions, not sourced details. This
module makes no parity claim.

Evidence for the verified contract is recorded in ``PORT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import torch
from torch import Tensor, nn

from ..yolo9.nn import Conv

_PACKINGS = ("head_major", "channel_major")
_RESIDUALS = ("both", "attention", "ffn", "none")


@dataclass(frozen=True)
class PSATopology:
    """Caller-supplied choices for details the cited sources leave open.

    No field has a default. A caller that supplies a topology is defining an
    experimental variant; the cited papers do not establish these values for
    the baseline nano detector.
    """

    heads: int
    query_width: int
    key_width: int
    value_width: int
    packing: str
    position_kernel: int | None
    ffn_kernel: int
    ffn_expansion: float
    residual: str

    def __post_init__(self) -> None:
        for name in ("heads", "query_width", "key_width", "value_width"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        if self.query_width != self.key_width:
            raise ValueError("query_width and key_width must match")
        if self.packing not in _PACKINGS:
            raise ValueError(f"packing must be one of {_PACKINGS}")
        if self.position_kernel is not None:
            if (
                isinstance(self.position_kernel, bool)
                or not isinstance(self.position_kernel, Integral)
                or self.position_kernel <= 0
                or self.position_kernel % 2 == 0
            ):
                raise ValueError("position_kernel must be a positive odd integer")
            object.__setattr__(self, "position_kernel", int(self.position_kernel))
        if (
            isinstance(self.ffn_kernel, bool)
            or not isinstance(self.ffn_kernel, Integral)
            or self.ffn_kernel <= 0
            or self.ffn_kernel % 2 == 0
        ):
            raise ValueError("ffn_kernel must be a positive odd integer")
        object.__setattr__(self, "ffn_kernel", int(self.ffn_kernel))
        if (
            isinstance(self.ffn_expansion, bool)
            or not isinstance(self.ffn_expansion, Real)
            or not isfinite(self.ffn_expansion)
            or self.ffn_expansion <= 0
        ):
            raise ValueError("ffn_expansion must be a positive finite number")
        object.__setattr__(self, "ffn_expansion", float(self.ffn_expansion))
        if self.residual not in _RESIDUALS:
            raise ValueError(f"residual must be one of {_RESIDUALS}")


class PSAAttention(nn.Module):
    """Multi-head self-attention over a feature map.

    ``packing`` fixes how the packed projection is read: ``head_major`` stores
    query, key and value for each head consecutively, while ``channel_major``
    stores all query channels, then all key channels, then all value channels.
    When ``position_kernel`` is set, a depthwise convolution of that kernel is
    added to the value map before the attention product.

    The explicit score matrix is quadratic in the number of sites. It is sized
    for the stage-10 20x20 map and is not intended for finer strides.
    """

    def __init__(self, channels: int, topology: PSATopology) -> None:
        super().__init__()
        if (
            isinstance(channels, bool)
            or not isinstance(channels, Integral)
            or channels <= 0
        ):
            raise ValueError("Attention channels must be a positive integer")
        self.channels = int(channels)
        self.topology = topology
        self.num_heads = topology.heads
        self.scale = topology.key_width**-0.5

        packed = topology.heads * (
            topology.query_width + topology.key_width + topology.value_width
        )
        self.qkv = Conv(channels, packed, 1, activation=False)
        value_channels = topology.heads * topology.value_width
        self.proj = Conv(value_channels, channels, 1, activation=False)
        self.pe = (
            None
            if topology.position_kernel is None
            else Conv(
                value_channels,
                value_channels,
                topology.position_kernel,
                groups=value_channels,
                activation=False,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        """Attend over one feature map and return a map of the same shape."""
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected a BCHW map with {self.channels} channels, "
                f"got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise ValueError("Attention input must be floating point")
        batch, _, height, width = x.shape
        sites = height * width
        topology = self.topology
        packed = self.qkv(x)

        if topology.packing == "head_major":
            packed = packed.view(
                batch,
                self.num_heads,
                topology.query_width + topology.key_width + topology.value_width,
                sites,
            )
            query, key, value = packed.split(
                (topology.query_width, topology.key_width, topology.value_width),
                dim=2,
            )
        else:
            query, key, value = packed.split(
                (
                    self.num_heads * topology.query_width,
                    self.num_heads * topology.key_width,
                    self.num_heads * topology.value_width,
                ),
                dim=1,
            )
            query = query.view(batch, self.num_heads, topology.query_width, sites)
            key = key.view(batch, self.num_heads, topology.key_width, sites)
            value = value.view(batch, self.num_heads, topology.value_width, sites)

        if self.pe is not None:
            value_channels = self.num_heads * topology.value_width
            position = self.pe(value.reshape(batch, value_channels, height, width))
            value = value + position.view(
                batch, self.num_heads, topology.value_width, sites
            )

        logits = (query.transpose(-2, -1) @ key) * self.scale
        attended = logits.softmax(dim=-1) @ value.transpose(-2, -1)
        attended = attended.transpose(-2, -1).reshape(
            batch, self.num_heads * topology.value_width, height, width
        )
        return self.proj(attended)


class PSABlock(nn.Module):
    """Attention sublayer followed by a two-convolution feed-forward sublayer.

    ``topology.residual`` selects where the residual additions apply: both
    sublayers, attention only, feed-forward only, or neither.
    """

    def __init__(self, channels: int, topology: PSATopology) -> None:
        super().__init__()
        hidden = int(channels * topology.ffn_expansion)
        if hidden <= 0:
            raise ValueError("ffn_expansion leaves no hidden channels")
        self.attn = PSAAttention(channels, topology)
        self.ffn = nn.Sequential(
            Conv(channels, hidden, topology.ffn_kernel),
            Conv(hidden, channels, topology.ffn_kernel, activation=False),
        )
        self.residual = topology.residual

    def forward(self, x: Tensor) -> Tensor:
        if self.residual in ("both", "attention"):
            x = x + self.attn(x)
        else:
            x = self.attn(x)
        if self.residual in ("both", "ffn"):
            x = x + self.ffn(x)
        else:
            x = self.ffn(x)
        return x


class C2PSA(nn.Module):
    """Split, attention blocks on one branch, concatenate, then project.

    ``cv1`` produces two branches. The second passes through ``blocks``
    PSABlocks while the first reaches the concatenation directly. ``blocks`` is
    required because the sources do not settle the nano repeat count after
    scaling, and ``split_ratio`` is required because the split widths are not
    specified.
    """

    def __init__(
        self,
        channels: int,
        *,
        blocks: int,
        topology: PSATopology,
        split_ratio: float,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        if (
            isinstance(channels, bool)
            or not isinstance(channels, Integral)
            or channels <= 0
        ):
            raise ValueError("C2PSA channels must be a positive integer")
        channels = int(channels)
        if (
            isinstance(expansion, bool)
            or not isinstance(expansion, Real)
            or not isfinite(expansion)
            or not 0.0 < expansion < 1.0
        ):
            raise ValueError("expansion must be a finite number between 0 and 1")
        if not isinstance(blocks, Integral) or isinstance(blocks, bool) or blocks < 1:
            raise ValueError("C2PSA needs a positive integer block count")
        if (
            isinstance(split_ratio, bool)
            or not isinstance(split_ratio, Real)
            or not isfinite(split_ratio)
            or not 0.0 < split_ratio < 1.0
        ):
            raise ValueError("split_ratio must be a finite number between 0 and 1")
        total = 2 * int(channels * float(expansion))
        direct = int(total * float(split_ratio))
        attended = total - direct
        if direct <= 0 or attended <= 0:
            raise ValueError("split_ratio leaves an empty branch")

        self.channels = channels
        self.direct_channels = direct
        self.attended_channels = attended
        self.cv1 = Conv(channels, total, 1)
        self.cv2 = Conv(total, channels, 1)
        self.m = nn.Sequential(
            *(PSABlock(attended, topology) for _ in range(int(blocks)))
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected a BCHW map with {self.channels} channels, "
                f"got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise ValueError("C2PSA input must be floating point")
        direct, attended = self.cv1(x).split(
            (self.direct_channels, self.attended_channels), dim=1
        )
        return self.cv2(torch.cat((direct, self.m(attended)), dim=1))
