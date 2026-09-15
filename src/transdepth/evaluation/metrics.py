"""Strict per-image metric depth evaluation on fixed GT-defined regions."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage


def edge_region(mask: np.ndarray, known: np.ndarray, radius: int = 3) -> np.ndarray:
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    near_transparent = ndimage.binary_dilation(mask == 1, structure=structure)
    near_background = ndimage.binary_dilation(mask == 0, structure=structure)
    safe_known = ndimage.binary_erosion(known, structure=structure, border_value=0)
    return near_transparent & near_background & safe_known


def _metrics(prediction: np.ndarray, target: np.ndarray, selected: np.ndarray) -> dict[str, Any]:
    count = int(selected.sum())
    if count == 0:
        return {"pixels": 0, "mae_mm": None, "rmse_mm": None, "abs_rel": None,
                "delta_1.05": None, "delta_1.10": None, "delta_1.25": None}
    pred = prediction[selected].astype(np.float64)
    gt = target[selected].astype(np.float64)
    residual = pred - gt
    ratio = np.maximum(pred / gt, gt / pred)
    return {
        "pixels": count,
        "mae_mm": float(np.mean(np.abs(residual)) * 1000.0),
        "rmse_mm": float(np.sqrt(np.mean(residual**2)) * 1000.0),
        "abs_rel": float(np.mean(np.abs(residual) / gt)),
        "delta_1.05": float(np.mean(ratio < 1.05)),
        "delta_1.10": float(np.mean(ratio < 1.10)),
        "delta_1.25": float(np.mean(ratio < 1.25)),
    }


def evaluate_image(
    prediction_m: np.ndarray,
    target_m: np.ndarray,
    valid_depth: np.ndarray,
    mask: np.ndarray,
    content: np.ndarray,
) -> dict[str, Any]:
    shapes = {item.shape for item in (prediction_m, target_m, valid_depth, mask, content)}
    if shapes != {(384, 512)}:
        raise ValueError(f"evaluation arrays must all be [384,512], got {shapes}")
    known = content & ((mask == 0) | (mask == 1))
    valid = content & valid_depth & known & np.isfinite(target_m) & (target_m > 0)
    if not np.isfinite(prediction_m[valid]).all() or not (prediction_m[valid] > 0).all():
        raise FloatingPointError("prediction is non-finite or non-positive on the fixed GT domain")
    regions = {
        "all": valid,
        "transparent": valid & (mask == 1),
        "background": valid & (mask == 0),
        "edge": valid & edge_region(mask, known),
    }
    return {name: _metrics(prediction_m, target_m, selected) for name, selected in regions.items()}
