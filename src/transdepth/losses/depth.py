"""Per-image, region-balanced metric log-depth loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class DepthLossResult:
    loss: Tensor
    per_image: Tensor
    transparent_pixels: tuple[int, ...]
    background_pixels: tuple[int, ...]


def balanced_log_huber_loss(
    prediction_m: Tensor,
    target_m: Tensor,
    valid_depth: Tensor,
    mask: Tensor,
    *,
    delta: float = 0.1,
) -> DepthLossResult:
    if prediction_m.shape != target_m.shape or prediction_m.shape != valid_depth.shape:
        raise ValueError("prediction, target, and validity tensors must have identical shapes")
    if mask.shape != prediction_m.shape:
        raise ValueError("mask must have the same shape as depth tensors")
    if prediction_m.ndim != 4 or prediction_m.shape[1] != 1:
        raise ValueError("depth tensors must be [B,1,H,W]")
    if delta <= 0:
        raise ValueError("Huber delta must be positive")

    image_losses: list[Tensor] = []
    transparent_counts: list[int] = []
    background_counts: list[int] = []
    for index in range(prediction_m.shape[0]):
        valid = valid_depth[index].bool() & ((mask[index] == 0) | (mask[index] == 1))
        class_losses: list[Tensor] = []
        counts = []
        for label in (1, 0):
            selected = valid & (mask[index] == label)
            count = int(selected.sum())
            counts.append(count)
            if count == 0:
                continue
            prediction = prediction_m[index][selected].float()
            target = target_m[index][selected].float()
            if not bool(torch.isfinite(prediction).all()) or not bool((prediction > 0).all()):
                raise FloatingPointError("predicted metric depth must be finite and positive")
            if not bool(torch.isfinite(target).all()) or not bool((target > 0).all()):
                raise FloatingPointError("valid target metric depth must be finite and positive")
            residual = prediction.log() - target.log()
            class_losses.append(F.huber_loss(residual, torch.zeros_like(residual), delta=delta))
        transparent_counts.append(counts[0])
        background_counts.append(counts[1])
        if not class_losses:
            raise ValueError(f"image {index} contains no valid known-class depth")
        image_losses.append(torch.stack(class_losses).mean())
    per_image = torch.stack(image_losses)
    return DepthLossResult(
        loss=per_image.mean(),
        per_image=per_image,
        transparent_pixels=tuple(transparent_counts),
        background_pixels=tuple(background_counts),
    )
