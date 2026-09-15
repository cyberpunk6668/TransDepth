"""Verify pinned DINOv3 H+/16 structure, weights, features, and zero-LoRA identity."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch

from transdepth.config import assert_runtime_assets, load_config
from transdepth.models.backbone.dinov3 import (
    DINOV3_COMMIT,
    DinoV3Features,
    load_dinov3_h16plus,
    verify_h16plus_structure,
)
from transdepth.utils.io import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/t0.yaml")
    parser.add_argument("--paths", default="configs/paths.local.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--random-structure-only", action="store_true")
    return parser.parse_args()


def _random_backbone(repo: Path):
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != DINOV3_COMMIT:
        raise RuntimeError(f"DINOv3 source commit mismatch: {commit}")
    sys.path.insert(0, str(repo))
    try:
        from dinov3.hub.backbones import dinov3_vith16plus

        model = dinov3_vith16plus(pretrained=False)
    finally:
        sys.path.pop(0)
    verify_h16plus_structure(model)
    return model.requires_grad_(False).eval()


def _tensor_digest(tensor: torch.Tensor) -> str:
    array = tensor.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.paths)
    assert_runtime_assets(config, require_weights=not args.random_structure_only)
    if not torch.cuda.is_available():
        raise RuntimeError("backbone integration verification requires CUDA")
    repo = Path(config["backbone"]["repo_dir"]).resolve(strict=True)
    if args.random_structure_only:
        backbone = _random_backbone(repo)
        status = "random_weights_structure_only"
        checkpoint_sha256 = None
    else:
        backbone = load_dinov3_h16plus(
            repo,
            config["backbone"]["checkpoint"],
            config["backbone"]["checkpoint_sha256"],
        )
        status = "official_weights_verified"
        checkpoint_sha256 = config["backbone"]["checkpoint_sha256"]
    device_index = 0
    torch.cuda.set_device(device_index)
    device = torch.device("cuda:0")
    torch.cuda.reset_peak_memory_stats(device_index)
    features = DinoV3Features(backbone).eval().to(device)
    generator = torch.Generator().manual_seed(17)
    rgb = torch.randn(1, 3, 384, 512, generator=generator).to(device)
    with torch.no_grad():
        baseline, _ = features(rgb)
    adapter = features.enable_lora(block_index=31, head_index=0, rank=4, seed=17)
    adapted, trace = features(rgb, return_trace=True)
    if trace is None:
        raise RuntimeError("adapter did not produce a trace")
    errors = [
        float((before - after).abs().max())
        for before, after in zip(baseline, adapted, strict=True)
    ]
    zero_identity_passed = max(errors) <= 1e-6
    loss = adapted[-1].float().square().mean()
    loss.backward()
    gradients = {
        "q_up": float(adapter.qkv.q_up.weight.grad.abs().sum()),
        "k_up": float(adapter.qkv.k_up.weight.grad.abs().sum()),
        "q_down": float(adapter.qkv.q_down.weight.grad.abs().sum()),
        "k_down": float(adapter.qkv.k_down.weight.grad.abs().sum()),
    }
    trainable_count = sum(
        parameter.numel() for parameter in adapter.parameters() if parameter.requires_grad
    )
    report = {
        "schema_version": "dinov3_backbone_check_v1",
        "status": status,
        "formal_training_ready": status == "official_weights_verified" and zero_identity_passed,
        "repo_commit": DINOV3_COMMIT,
        "checkpoint_sha256": checkpoint_sha256,
        "model": {
            "parameters": sum(parameter.numel() for parameter in backbone.parameters()),
            "blocks": backbone.n_blocks,
            "width": backbone.embed_dim,
            "heads": backbone.num_heads,
            "prefix_tokens": 1 + backbone.n_storage_tokens,
            "feature_shapes": [list(item.shape) for item in adapted],
            "feature_digests": [_tensor_digest(item) for item in adapted],
        },
        "zero_lora": {
            "max_abs_errors": errors,
            "threshold": 1e-6,
            "passed": zero_identity_passed,
        },
        "lora": {
            "trainable_parameters": trainable_count,
            "expected_trainable_parameters": 10752,
            "gradient_abs_sums": gradients,
            "initial_down_gradient_expected_zero": True,
        },
        "environment": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device_index),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device_index),
        },
        "limitations": (
            [
                "Random weights validate integration only; "
                "they cannot be used for training or research claims."
            ]
            if args.random_structure_only
            else []
        ),
    }
    output = Path(
        args.output
        or Path(config["storage"]["qa"])
        / "backbone"
        / ("random_structure.json" if args.random_structure_only else "official_weights.json")
    )
    atomic_write_json(report, output)
    print(json.dumps(report, indent=2))
    if trainable_count != 10752 or not zero_identity_passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
