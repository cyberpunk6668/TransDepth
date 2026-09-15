"""Shared half-pixel letterbox geometry for all training modalities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from transdepth.data.schema import CanonicalBundle, GeometryRecord

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _target_geometry(
    original_hw: tuple[int, int], canvas_hw: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int, int, int], GeometryRecord]:
    h0, w0 = original_hw
    height, width = canvas_hw
    if min(h0, w0, height, width) <= 0:
        raise ValueError("image and canvas dimensions must be positive")
    scale = min(height / h0, width / w0)
    h1 = min(height, max(1, round(scale * h0)))
    w1 = min(width, max(1, round(scale * w0)))
    top = (height - h1) // 2
    bottom = height - h1 - top
    left = (width - w1) // 2
    right = width - w1 - left
    sy, sx = h1 / h0, w1 / w0
    affine = (
        (sx, 0.0, left + (sx - 1.0) / 2.0),
        (0.0, sy, top + (sy - 1.0) / 2.0),
        (0.0, 0.0, 1.0),
    )
    record = GeometryRecord(
        original_hw=original_hw,
        resized_hw=(h1, w1),
        canvas_hw=canvas_hw,
        scale_yx=(sy, sx),
        padding_tblr=(top, bottom, left, right),
        pixel_center_convention="pytorch_align_corners_false_half_pixel",
        affine_3x3=affine,
    )
    return (h1, w1), (top, bottom, left, right), record


def letterbox_rgb(
    rgb_u8: np.ndarray, canvas_hw: tuple[int, int] = (384, 512)
) -> tuple[torch.Tensor, torch.Tensor, GeometryRecord]:
    if rgb_u8.dtype != np.uint8 or rgb_u8.ndim != 3 or rgb_u8.shape[2] != 3:
        raise ValueError("rgb_u8 must be uint8 [H,W,3]")
    (h1, w1), (top, _, left, _), geometry = _target_geometry(rgb_u8.shape[:2], canvas_hw)
    source = torch.from_numpy(rgb_u8.copy()).permute(2, 0, 1).float().div_(255.0)
    resized = F.interpolate(
        source.unsqueeze(0), size=(h1, w1), mode="bilinear", align_corners=False, antialias=True
    )[0]
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[:, None, None]
    resized = (resized - mean) / std
    rgb = torch.zeros((3, *canvas_hw), dtype=torch.float32)
    content = torch.zeros(canvas_hw, dtype=torch.bool)
    rgb[:, top : top + h1, left : left + w1] = resized
    content[top : top + h1, left : left + w1] = True
    return rgb, content, geometry


def letterbox_bundle(
    rgb_u8: np.ndarray,
    depth_m: np.ndarray,
    labels: np.ndarray,
    valid_depth: np.ndarray,
    canvas_hw: tuple[int, int] = (384, 512),
) -> CanonicalBundle:
    original_hw = tuple(rgb_u8.shape[:2])
    if any(tuple(item.shape) != original_hw for item in (depth_m, labels, valid_depth)):
        raise ValueError("RGB, depth, labels, and valid_depth must share HxW")
    rgb, content, geometry = letterbox_rgb(rgb_u8, canvas_hw)
    h1, w1 = geometry.resized_hw
    top, _, left, _ = geometry.padding_tblr

    def resize(array: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(np.ascontiguousarray(array)).float()[None, None]
        return F.interpolate(tensor, size=(h1, w1), mode="nearest-exact")[0, 0]

    depth_resized = resize(depth_m)
    labels_resized = resize(labels).to(torch.uint8)
    valid_resized = resize(valid_depth).bool()
    depth = torch.zeros(canvas_hw, dtype=torch.float32)
    mask = torch.full(canvas_hw, 255, dtype=torch.uint8)
    valid = torch.zeros(canvas_hw, dtype=torch.bool)
    region = (slice(top, top + h1), slice(left, left + w1))
    depth[region] = depth_resized
    mask[region] = labels_resized
    valid[region] = valid_resized
    valid &= content & torch.isfinite(depth) & (depth > 0)
    known = content & ((mask == 0) | (mask == 1))
    return CanonicalBundle(
        rgb=rgb,
        depth_m=depth.unsqueeze(0),
        mask=mask.unsqueeze(0),
        valid_depth=valid.unsqueeze(0),
        mask_known=known.unsqueeze(0),
        content_mask=content.unsqueeze(0),
        geometry=geometry,
    )
