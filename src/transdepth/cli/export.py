"""Create a compact merged RGB-only deployment export from a trained H1/H2 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from transdepth.config import load_config
from transdepth.data.rgb_only import ManifestRgbOnlyDataset
from transdepth.engine.checkpoint import atomic_torch_save, load_checkpoint
from transdepth.inference.merge import merge_predictor_for_export
from transdepth.models.factory import build_predictor
from transdepth.utils.io import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference/predict.yaml")
    parser.add_argument("--paths", default="configs/paths.local.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--atol-m", type=float, default=0.0001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.paths)
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["experiment_kind"] not in {"h1", "h2"}:
        raise ValueError("merged export requires a trained H1 or H2 checkpoint")
    spec = checkpoint["model_spec"]
    for key in ("selected_block", "selected_head", "beta", "rank", "eta"):
        config["far"][key] = spec[key]
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    predictor = build_predictor(
        config, use_lora=True, trained_checkpoint=checkpoint_path
    ).eval().to(device)
    manifest = Path(config["storage"]["manifests"]) / "rftrans_only_v1.jsonl"
    sample = ManifestRgbOnlyDataset(manifest, config["roots"]["rftrans"], "R_dev")[0]
    rgb = sample.rgb[None].to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        before = predictor(rgb, return_aux=False).depth_m
    export = merge_predictor_for_export(predictor, config)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        after = predictor(rgb, return_aux=False).depth_m
    max_error = float((before - after).abs().max())
    if max_error > args.atol_m:
        raise RuntimeError(f"merged RGB prediction mismatch: {max_error} m")
    export["equivalence"] = {
        "sample_id": sample.sample_id,
        "max_abs_depth_m": max_error,
        "threshold_m": args.atol_m,
        "passed": True,
    }
    output = Path(args.output).resolve()
    atomic_torch_save(export, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    atomic_write_json(
        {
            "schema_version": "td_rgb_export_manifest_v1",
            "export": str(output),
            "sha256": digest,
            "source_checkpoint": str(checkpoint_path),
            "source_checkpoint_sha256": hashlib.sha256(
                checkpoint_path.read_bytes()
            ).hexdigest(),
            "equivalence": export["equivalence"],
        },
        output.with_suffix(output.suffix + ".json"),
    )
    print(json.dumps({"export": str(output), "sha256": digest, **export["equivalence"]}, indent=2))


if __name__ == "__main__":
    main()
