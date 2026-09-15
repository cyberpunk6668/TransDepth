"""Reliable transparent/background patch targets from depth and semantic evidence."""

from __future__ import annotations

import torch
from torch import Tensor

from transdepth.data.schema import PatchTargets


def build_patch_targets(
    depth_m: Tensor,
    labels: Tensor,
    valid_depth: Tensor,
    content_mask: Tensor,
    *,
    patch_size: int = 16,
    min_valid_fraction: float = 0.9,
    min_class_fraction: float = 0.9,
    max_log_depth_iqr: float = 0.1,
) -> PatchTargets:
    tensors = [
        item.squeeze(0) if item.ndim == 3 and item.shape[0] == 1 else item
        for item in (depth_m, labels, valid_depth, content_mask)
    ]
    depth, label, valid, content = tensors
    if any(item.ndim != 2 for item in tensors) or len({tuple(item.shape) for item in tensors}) != 1:
        raise ValueError("patch target inputs must share a two-dimensional shape")
    height, width = depth.shape
    if height % patch_size or width % patch_size:
        raise ValueError("canvas dimensions must be divisible by patch_size")
    gh, gw = height // patch_size, width // patch_size
    state = torch.full((gh, gw), -1, dtype=torch.int8, device=depth.device)
    median = torch.full((gh, gw), torch.nan, dtype=torch.float32, device=depth.device)
    valid_fraction = torch.zeros((gh, gw), dtype=torch.float32, device=depth.device)
    transparent_fraction = torch.full_like(valid_fraction, torch.nan)
    log_iqr = torch.full_like(valid_fraction, torch.nan)
    relation_valid = valid.bool() & content.bool() & ((label == 0) | (label == 1))

    for row in range(gh):
        ys = slice(row * patch_size, (row + 1) * patch_size)
        for col in range(gw):
            xs = slice(col * patch_size, (col + 1) * patch_size)
            if not bool(content[ys, xs].all()):
                continue
            selected = relation_valid[ys, xs]
            count = int(selected.sum())
            if count == 0:
                continue
            values = depth[ys, xs][selected].float()
            if not bool(torch.isfinite(values).all()) or not bool((values > 0).all()):
                raise ValueError("valid patch depths must be finite and positive")
            fraction = count / float(patch_size * patch_size)
            class_fraction = (label[ys, xs][selected] == 1).float().mean()
            q25, _, q75 = torch.quantile(
                values.log(), torch.tensor([0.25, 0.5, 0.75], device=values.device)
            )
            valid_fraction[row, col] = fraction
            transparent_fraction[row, col] = class_fraction
            log_iqr[row, col] = q75 - q25
            median[row, col] = torch.quantile(values, 0.5)
            if fraction >= min_valid_fraction and (q75 - q25) <= max_log_depth_iqr:
                if class_fraction >= min_class_fraction:
                    state[row, col] = 1
                elif (1.0 - class_fraction) >= min_class_fraction:
                    state[row, col] = 0
    return PatchTargets(state, median, valid_fraction, transparent_fraction, log_iqr)
