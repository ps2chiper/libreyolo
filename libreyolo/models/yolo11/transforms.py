"""Image and detection adapters for the prospective YOLO11 detection family."""

import numpy as np
import torch

from ...postprocess.yolo9 import ImageSize, postprocess
from ...preprocess.letterbox import LETTERBOX_CENTER
from ...preprocess.yolo9 import preprocess_numpy


def prepare_batch(
    images: list[np.ndarray], input_size: ImageSize
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """Letterbox RGB uint8 images and retain the original (width, height) for each."""
    if not images:
        raise ValueError("Image batch must contain at least one image")
    canvases = []
    original_sizes = []
    for image in images:
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 image with three channels")
        canvas, _ = preprocess_numpy(image, input_size, letterbox_pad=LETTERBOX_CENTER)
        canvases.append(canvas)
        original_sizes.append((image.shape[1], image.shape[0]))
    return torch.from_numpy(np.ascontiguousarray(np.stack(canvases))), original_sizes


def restore_detections(
    predictions: torch.Tensor,
    original_sizes: list[tuple[int, int]],
    input_size: ImageSize,
    *,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> list[dict]:
    """Filter decoded pixel-space xyxy boxes and class probabilities for each image."""
    if predictions.ndim != 3 or predictions.shape[1] < 5:
        raise ValueError("Expected predictions shaped (batch, 4 + classes, anchors)")
    if predictions.shape[0] != len(original_sizes):
        raise ValueError("Each prediction batch item needs an original image size")
    return [
        postprocess(
            {"predictions": sample},
            original_size=size,
            input_size=input_size,
            letterbox_pad=LETTERBOX_CENTER,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            max_det=max_det,
        )
        for sample, size in zip(predictions.unbind(), original_sizes)
    ]
