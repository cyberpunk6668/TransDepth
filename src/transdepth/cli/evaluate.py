"""Evaluate complete persisted RFTrans predictions; this process alone reads GT labels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from transdepth.data.dataset import RFTransDataset
from transdepth.evaluation.metrics import evaluate_image
from transdepth.inference.serialization import read_prediction_manifest
from transdepth.utils.io import atomic_write_json, atomic_write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--rftrans-root", required=True)
    parser.add_argument("--role", default="R_hold")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    regions = ("all", "transparent", "background", "edge")
    metrics = ("mae_mm", "rmse_mm", "abs_rel", "delta_1.05", "delta_1.10", "delta_1.25")
    result: dict[str, Any] = {}
    for region in regions:
        result[region] = {}
        for metric in metrics:
            values = [row["metrics"][region][metric] for row in rows]
            finite = [value for value in values if value is not None]
            result[region][metric] = float(np.mean(finite)) if finite else None
        result[region]["images"] = sum(
            row["metrics"][region]["pixels"] > 0 for row in rows
        )
    return {
        "schema_version": "rftrans_evaluation_summary_v1",
        "status": "complete",
        "samples": len(rows),
        "aggregation": "image_macro; one generated frame is one group",
        "regions": result,
    }


def main() -> None:
    args = parse_args()
    prediction_dir = Path(args.predictions).resolve(strict=True)
    completion = prediction_dir / "predictions.complete.json"
    if not completion.is_file():
        raise ValueError("predictions.complete.json is required before labels may be read")
    prediction_rows = read_prediction_manifest(prediction_dir / "predictions.jsonl")
    predictions = {row["sample_id"]: row for row in prediction_rows}
    dataset = RFTransDataset(args.manifest, args.rftrans_root, {args.role})
    expected = {record.sample_id for record in dataset.records}
    if not args.allow_partial and set(predictions) != expected:
        raise ValueError("prediction IDs do not exactly cover the requested evaluation role")
    rows = []
    record_to_index = {record.sample_id: index for index, record in enumerate(dataset.records)}
    for sample_id in sorted(set(predictions) & expected):
        prediction_record = predictions[sample_id]
        depth_path = prediction_dir / prediction_record["depth_path"]
        prediction = np.load(depth_path, allow_pickle=False)
        sample = dataset[record_to_index[sample_id]]
        metrics = evaluate_image(
            prediction,
            sample.depth_m[0].numpy(),
            sample.valid_depth[0].numpy(),
            sample.mask[0].numpy(),
            sample.content_mask[0].numpy(),
        )
        rows.append({"sample_id": sample_id, "metrics": metrics})
    output = Path(args.output)
    atomic_write_jsonl(rows, output / "per_image.jsonl")
    atomic_write_json(_summary(rows), output / "summary.json")
    print(f"evaluated {len(rows)} persisted predictions into {output}")


if __name__ == "__main__":
    main()
