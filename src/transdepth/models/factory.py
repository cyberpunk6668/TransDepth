"""Build predictors only from verified native DINOv3 assets and locked configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transdepth.config import ConfigError, assert_runtime_assets
from transdepth.engine.checkpoint import load_checkpoint, load_trainable_state
from transdepth.models.backbone.dinov3 import DinoV3Features, load_dinov3_h16plus
from transdepth.models.predictor import DepthPredictor


def apply_oracle_lock(config: dict[str, Any], lock_path: str | Path) -> dict[str, Any]:
    lock = json.loads(Path(lock_path).read_text(encoding="utf-8"))
    if lock.get("schema_version") != "oracle_lock_v1" or lock.get("status") != "passed":
        raise ConfigError("H1/H2 require a passed oracle_lock_v1")
    config["far"]["selected_block"] = int(lock["selected_block"])
    config["far"]["selected_head"] = int(lock["selected_head"])
    config["far"]["beta"] = float(lock["beta"])
    config["training"]["oracle_lock"] = str(Path(lock_path).resolve(strict=True))
    return lock


def build_predictor(
    config: dict[str, Any],
    *,
    use_lora: bool,
    fork_checkpoint: str | Path | None = None,
    trained_checkpoint: str | Path | None = None,
) -> DepthPredictor:
    if fork_checkpoint is not None and trained_checkpoint is not None:
        raise ConfigError("fork_checkpoint and trained_checkpoint are mutually exclusive")
    assert_runtime_assets(config, require_weights=True, require_training_data=False)
    backbone = load_dinov3_h16plus(
        config["backbone"]["repo_dir"],
        config["backbone"]["checkpoint"],
        config["backbone"]["checkpoint_sha256"],
    )
    features = DinoV3Features(backbone)
    features.activation_checkpoint_suffix = bool(
        config["runtime"].get("activation_checkpoint_suffix", True)
    )
    if use_lora:
        block = config["far"].get("selected_block")
        head = config["far"].get("selected_head")
        if block is None or head is None:
            raise ConfigError("LoRA needs a block/head from a passed oracle lock")
        features.enable_lora(
            block_index=int(block),
            head_index=int(head),
            rank=int(config["far"]["rank"]),
            eta=float(config["far"]["eta"]),
            seed=int(config["runtime"]["seed"]),
        )
    predictor = DepthPredictor(
        features,
        feature_width=int(config["backbone"]["feature_width"]),
        decoder_width=int(config["model"]["decoder_width"]),
    )
    if fork_checkpoint is not None:
        checkpoint = load_checkpoint(fork_checkpoint)
        if checkpoint.get("experiment_kind") != "t0":
            raise ConfigError("H1/H2 must fork from a T0 checkpoint")
        load_trainable_state(predictor, checkpoint["trainable_state"])
        config["training"]["fork_from"] = str(Path(fork_checkpoint).resolve(strict=True))
    if trained_checkpoint is not None:
        checkpoint = load_checkpoint(trained_checkpoint)
        load_trainable_state(predictor, checkpoint["trainable_state"])
    return predictor
