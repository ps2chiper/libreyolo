"""C2PSA interface for the YOLO11 family.

The cited sources establish the outer contract only: stage 10 receives the
SPPF output and returns the same shape, the outer expansion is 0.5, and the
block is an initial convolution, a sequence of PSA blocks, a concatenation and
a final convolution. Every inner detail is a caller decision carried by
:class:`PSATopology`; this module does not choose a topology and makes no
parity claim.

Evidence for the verified contract is recorded in ``PORT.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from ..yolo9.nn import Conv

_PACKINGS = ("head_major", "channel_major")


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
    ffn_expansion: float
    residual: bool

    def __post_init__(self) -> None:
        if self.heads <= 0:
            raise ValueError("heads must be positive")
        if min(self.query_width, self.key_width, self.value_width) <= 0:
            raise ValueError("query, key and value widths must be positive")
        if self.packing not in _PACKINGS:
            raise ValueError(f"packing must be one of {_PACKINGS}")
        if self.position_kernel is not None and (
            self.position_kernel <= 0 or self.position_kernel % 2 == 0
        ):
            raise ValueError("position_kernel must be a positive odd integer")
        if self.ffn_expansion <= 0:
            raise ValueError("ffn_expansion must be positive")


class PSAAttention(nn.Module):
    """Multi-head self-attention over a feature map.

    ``packing`` fixes how the packed projection is read: ``head_major`` stores
    query, key and value for each head consecutively, while ``channel_major``
    stores all query channels, then all key channels, then all value channels.
    When ``position_kernel`` is set, a depthwise convolution of that kernel is
    added to the value map before the attention product.
    """

    def __init__(self, channels: int, topology: PSATopology) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("Attention channels must be positive")
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

    ``residual`` controls whether each sublayer output is added to its input.
    """

    def __init__(self, channels: int, topology: PSATopology) -> None:
        super().__init__()
        hidden = int(channels * topology.ffn_expansion)
        if hidden <= 0:
            raise ValueError("ffn_expansion leaves no hidden channels")
        self.attn = PSAAttention(channels, topology)
        self.ffn = nn.Sequential(
            Conv(channels, hidden, 1),
            Conv(hidden, channels, 1, activation=False),
        )
        self.add = topology.residual

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    """Split, attention blocks on one branch, concatenate, then project.

    ``cv1`` produces two equal halves. The second half passes through ``blocks``
    PSABlocks while the first reaches the concatenation directly. ``blocks`` is
    required because the sources do not settle the nano repeat count after
    scaling.
    """

    def __init__(
        self,
        channels: int,
        *,
        blocks: int,
        topology: PSATopology,
        expansion: float = 0.5,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("C2PSA channels must be positive")
        if not 0.0 < expansion < 1.0:
            raise ValueError("expansion must be between 0 and 1")
        if blocks < 1:
            raise ValueError("C2PSA needs at least one PSABlock")
        inner = int(channels * expansion)
        if inner <= 0:
            raise ValueError("C2PSA expansion leaves no inner channels")

        self.channels = int(channels)
        self.c = inner
        self.cv1 = Conv(channels, 2 * inner, 1)
        self.cv2 = Conv(2 * inner, channels, 1)
        self.m = nn.Sequential(
            *(PSABlock(inner, topology) for _ in range(int(blocks)))
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.shape[1] != self.channels:
            raise ValueError(
                f"Expected a BCHW map with {self.channels} channels, "
                f"got {tuple(x.shape)}"
            )
        direct, attended = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((direct, self.m(attended)), dim=1))