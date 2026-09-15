"""Generate immutable RGB-only metric-depth predictions for RFTrans or ClearGrasp."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from transdepth.config import load_config
from transdepth.data.depth_io import sha256_file
from transdepth.data.rgb_only import ManifestRgbOnlyDataset, RgbOnlyDataset
from transdepth.engine.checkpoint import load_checkpoint
from transdepth.inference.serialization import save_depth_npy, write_prediction_manifest
from transdepth.models.factory import build_predictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference/predict.yaml")
    parser.add_argument("--paths", default="configs/inference.paths.local.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source", choices=("cleargrasp", "rftrans"), required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--role", default="R_hold")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.paths)
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    checkpoint = load_checkpoint(checkpoint_path)
    kind = checkpoint["experiment_kind"]
    use_lora = kind in {"h1", "h2"}
    if use_lora:
        spec = checkpoint.get("model_spec", {})
        for key in ("selected_block", "selected_head", "beta", "rank", "eta"):
            if spec.get(key) is None:
                raise ValueError(f"checkpoint lacks model_spec.{key}")
            config["far"][key] = spec[key]
    if args.source == "cleargrasp":
        dataset = RgbOnlyDataset(config["roots"]["cleargrasp_inference"])
    else:
        if args.manifest is None or "rftrans" not in config.get("roots", {}):
            raise ValueError(
                "RFTrans prediction requires --manifest and a paths file with roots.rftrans"
            )
        dataset = ManifestRgbOnlyDataset(args.manifest, config["roots"]["rftrans"], args.role)
    if not torch.cuda.is_available():
        raise RuntimeError("prediction requires CUDA")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    model = build_predictor(
        config, use_lora=use_lora, trained_checkpoint=checkpoint_path
    ).eval().to(device)
    output = Path(args.output).resolve()
    depth_dir = output / "depth"
    limit = len(dataset) if args.max_images is None else min(len(dataset), args.max_images)
    checkpoint_sha = sha256_file(checkpoint_path)
    rows = []
    for index in range(limit):
        sample = dataset[index]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = model(sample.rgb[None].to(device), return_aux=False).depth_m
        depth = prediction[0, 0].float().cpu().numpy().astype(np.float32, copy=False)
        filename = f"{sample.sample_id}.npy"
        depth_sha = save_depth_npy(depth, depth_dir / filename)
        rgb_full_path = Path(dataset.root) / sample.rgb_path
        rows.append(
            {
                "schema_version": "prediction_v1",
                "sample_id": sample.sample_id,
                "source": args.source,
                "rgb_path": sample.rgb_path,
                "rgb_sha256": sha256_file(rgb_full_path),
                "depth_path": f"depth/{filename}",
                "depth_sha256": depth_sha,
                "dtype": "float32",
                "unit": "m",
                "coordinate": "camera_z_first_surface",
                "canvas_hw": [384, 512],
                "transform": asdict(sample.geometry),
                "model_checkpoint_sha256": checkpoint_sha,
            }
        )
    write_prediction_manifest(rows, output, checkpoint_sha)
    print(f"wrote {len(rows)} RGB-only predictions to {output}")
    manifest_sha = hashlib.sha256((output / "predictions.jsonl").read_bytes()).hexdigest()
    print(f"prediction manifest SHA256 {manifest_sha}")


if __name__ == "__main__":
    main()
