"""Sampled RFTrans encoding, geometry, and reliable-patch audit."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from transdepth.data.depth_io import (
    read_depth_png_raw,
    read_exr_channel,
    read_rftrans_depth,
    read_rftrans_mask,
    read_rgb_u8,
)
from transdepth.data.geometry import IMAGENET_MEAN, IMAGENET_STD, letterbox_bundle
from transdepth.data.patches import build_patch_targets
from transdepth.data.schema import SampleRecord


def select_audit_records(
    records: list[SampleRecord], roles: set[str], per_role: int, seed: int
) -> list[SampleRecord]:
    if per_role <= 0:
        raise ValueError("per_role must be positive")
    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in records:
        if record.assigned_role in roles:
            grouped[record.assigned_role].append(record)
    missing = roles - set(grouped)
    if missing:
        raise ValueError(f"manifest has no records for roles: {sorted(missing)}")
    selected = []
    for role in sorted(roles):
        ranked = sorted(
            grouped[role],
            key=lambda item: hashlib.sha256(f"{seed}:{item.sample_id}".encode()).digest(),
        )
        selected.extend(ranked[:per_role])
    return selected


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError(f"asset escapes RFTrans root: {relative}")
    return path


def _quantiles(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return []
    return [float(item) for item in np.quantile(values, (0.01, 0.5, 0.99))]


def audit_record(record: SampleRecord, root: Path) -> tuple[dict[str, Any], Any, np.ndarray]:
    rgb = read_rgb_u8(_resolve(root, record.rgb_path))
    raw = read_depth_png_raw(_resolve(root, record.depth_gt_path))
    depth_m, valid = read_rftrans_depth(_resolve(root, record.depth_gt_path))
    labels = read_rftrans_mask(_resolve(root, record.mask_path))
    reference, exr_meta = read_exr_channel(_resolve(root, record.depth_reference_path), "Y")
    if rgb.shape[:2] != raw.shape or raw.shape != labels.shape or raw.shape != reference.shape:
        raise ValueError(f"unaligned modalities for {record.sample_id}")
    difference = np.abs(reference - depth_m)
    bundle = letterbox_bundle(rgb, depth_m, labels, valid)
    patches = build_patch_targets(
        bundle.depth_m, bundle.mask, bundle.valid_depth, bundle.content_mask
    )
    valid_values = depth_m[valid]
    row = {
        "sample_id": record.sample_id,
        "role": record.assigned_role,
        "frame_id": record.frame_id,
        "cad_ids": list(record.cad_ids),
        "rgb_shape": list(rgb.shape),
        "depth_raw_dtype": str(raw.dtype),
        "depth_raw_range": [int(raw.min()), int(raw.max())],
        "depth_zero_fraction": float((raw == 0).mean()),
        "depth_q01_q50_q99_m": _quantiles(valid_values),
        "mask_transparent_fraction": float((labels == 1).mean()),
        "mask_unknown_fraction": float((labels == 255).mean()),
        "exr": exr_meta,
        "png_exr_mae_m": float(difference.mean()),
        "png_exr_max_abs_m": float(difference.max()),
        "geometry": asdict(bundle.geometry),
        "patches": {
            "transparent": int((patches.state == 1).sum()),
            "background": int((patches.state == 0).sum()),
            "unknown": int((patches.state < 0).sum()),
        },
    }
    return row, bundle, rgb


def save_qa_figure(
    record: SampleRecord,
    row: dict[str, Any],
    bundle: Any,
    rgb: np.ndarray,
    path: Path,
) -> None:
    mean = torch.tensor(IMAGENET_MEAN)[:, None, None]
    std = torch.tensor(IMAGENET_STD)[:, None, None]
    canvas = ((bundle.rgb * std + mean).clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    depth = bundle.depth_m[0].numpy()
    valid = bundle.valid_depth[0].numpy()
    mask = bundle.mask[0].numpy()
    overlay = canvas.astype(np.float32)
    overlay[mask == 1] = 0.5 * overlay[mask == 1] + 0.5 * np.array([0, 255, 0])
    overlay[mask == 255] = 0.5 * overlay[mask == 255] + 0.5 * np.array([255, 0, 255])
    display_depth = np.where(valid, depth, np.nan)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    axes[0].imshow(rgb)
    axes[0].set_title("raw RGB")
    image = axes[1].imshow(display_depth, cmap="viridis")
    axes[1].set_title("metric z-depth (m)")
    figure.colorbar(image, ax=axes[1], fraction=0.046)
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title("green=T, magenta=ignore")
    axes[3].imshow(bundle.content_mask[0].numpy(), cmap="gray", vmin=0, vmax=1)
    axes[3].set_title(
        f"content; T/B={row['patches']['transparent']}/{row['patches']['background']}"
    )
    for axis in axes:
        axis.axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def summarize_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_role[row["role"]].append(row)
    roles: dict[str, Any] = {}
    passed = True
    for role, items in sorted(by_role.items()):
        max_difference = max(item["png_exr_max_abs_m"] for item in items)
        unknown_fraction = float(np.mean([item["mask_unknown_fraction"] for item in items]))
        role_passed = max_difference <= 0.001 and unknown_fraction == 0.0
        passed &= role_passed
        roles[role] = {
            "samples": len(items),
            "png_exr_max_abs_m": max_difference,
            "png_exr_mean_mae_m": float(np.mean([item["png_exr_mae_m"] for item in items])),
            "depth_zero_fraction_mean": float(
                np.mean([item["depth_zero_fraction"] for item in items])
            ),
            "mask_unknown_fraction_mean": unknown_fraction,
            "transparent_patch_mean": float(
                np.mean([item["patches"]["transparent"] for item in items])
            ),
            "background_patch_mean": float(
                np.mean([item["patches"]["background"] for item in items])
            ),
            "passed": role_passed,
        }
    return {
        "schema_version": "rftrans_audit_v1",
        "status": "passed" if passed else "failed",
        "criteria": {"png_exr_max_abs_m": 0.001, "mask_unknown_fraction": 0.0},
        "roles": roles,
        "important_limitations": [
            "PNG/EXR agreement validates storage consistency, not first-surface geometry alone.",
            "Recorder metadata supports per-generated-frame groups; "
            "no repeated-capture scene ID exists.",
            "No R_hold labels were read by this audit.",
        ],
    }
