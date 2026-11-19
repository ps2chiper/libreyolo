"""BatchNorm mode predicates shared by the YOLO11 head and detector."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm


def sync_world_size(module: nn.Module) -> int:
    """World size for this layer's process group, or one when sync is off."""
    if not (dist.is_available() and dist.is_initialized()):
        return 1
    try:
        return dist.get_world_size(getattr(module, "process_group", None))
    except (RuntimeError, ValueError):
        return 1


def batch_norm_needs_batch_statistics(
    module: _BatchNorm,
    world_size: Callable[[nn.Module], int] | None = None,
) -> bool:
    """Whether this layer normalizes with batch statistics.

    In evaluation mode, PyTorch falls back to batch statistics when both
    running buffers are absent. A SyncBatchNorm layer in training mode uses
    local statistics unless its process group spans more than one rank.
    """
    if module.training:
        resolve = world_size or sync_world_size
        is_synchronized = (
            isinstance(module, nn.SyncBatchNorm) and resolve(module) > 1
        )
        return not is_synchronized
    return module.running_mean is None and module.running_var is None


def sync_batch_norm_needs_accelerator(module: _BatchNorm) -> bool:
    """Whether a training SyncBatchNorm layer would reject a CPU input."""
    if not isinstance(module, nn.SyncBatchNorm) or not module.training:
        return False
    if not (dist.is_available() and dist.is_initialized()):
        return False
    if module.weight is not None:
        device = module.weight.device
    elif module.running_mean is not None:
        device = module.running_mean.device
    else:
        device = torch.device("cpu")
    return device.type == "cpu"