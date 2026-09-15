"""Atomic float32-metric prediction files and label-free manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np

from transdepth.utils.io import atomic_write_json, atomic_write_jsonl


def save_depth_npy(depth_m: np.ndarray, path: str | Path) -> str:
    if depth_m.shape != (384, 512) or depth_m.dtype != np.float32:
        raise ValueError("prediction must be float32 [384,512]")
    if not np.isfinite(depth_m).all() or not (depth_m > 0).all():
        raise FloatingPointError("prediction must be finite and positive")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        np.save(stream, depth_m, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)
    return hashlib.sha256(destination.read_bytes()).hexdigest()


def write_prediction_manifest(
    rows: list[dict[str, Any]], output_dir: str | Path, model_checkpoint_sha256: str
) -> None:
    output = Path(output_dir)
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("prediction sample IDs must be unique")
    atomic_write_jsonl(rows, output / "predictions.jsonl")
    manifest_hash = hashlib.sha256((output / "predictions.jsonl").read_bytes()).hexdigest()
    atomic_write_json(
        {
            "schema_version": "predictions_complete_v1",
            "status": "complete",
            "samples": len(rows),
            "manifest_sha256": manifest_hash,
            "model_checkpoint_sha256": model_checkpoint_sha256,
        },
        output / "predictions.complete.json",
    )


def read_prediction_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            rows.append(json.loads(line))
    return rows
