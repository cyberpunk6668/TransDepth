"""Merge selected Q/K LoRA into a fresh backbone and serialize a compact RGB-only export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from transdepth.config import ConfigError, assert_runtime_assets
from transdepth.engine.checkpoint import load_trainable_state
from transdepth.models.backbone.dinov3 import DinoV3Features, load_dinov3_h16plus
from transdepth.models.predictor import DepthPredictor, RGBDepthModule


@torch.no_grad()
def merge_predictor_for_export(
    predictor: DepthPredictor, config: dict[str, Any]
) -> dict[str, Any]:
    features = predictor.features
    adapter = getattr(features, "adapter", None)
    if adapter is None:
        raise ValueError("a LoRA-enabled predictor is required for merged export")
    q_delta = adapter.qkv.eta * (adapter.qkv.q_up.weight @ adapter.qkv.q_down.weight)
    k_delta = adapter.qkv.eta * (adapter.qkv.k_up.weight @ adapter.qkv.k_down.weight)
    block = adapter.selected_block
    head = adapter.selected_head
    adapter.qkv.merge_into_base_()
    features.disable_lora()
    decoder_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in predictor.named_parameters()
        if parameter.requires_grad
    }
    return {
        "schema_version": "td_rgb_export_v1",
        "backbone_sha256": config["backbone"]["checkpoint_sha256"],
        "selected_block": block,
        "selected_head": head,
        "rank": config["far"]["rank"],
        "eta": config["far"]["eta"],
        "q_delta": q_delta.detach().float().cpu(),
        "k_delta": k_delta.detach().float().cpu(),
        "decoder_state": decoder_state,
        "merged": True,
        "model_input": "normalized_rgb_only",
        "output_unit": "m",
    }


def build_exported_predictor(
    config: dict[str, Any], export_path: str | Path
) -> RGBDepthModule:
    assert_runtime_assets(config, require_weights=True, require_training_data=False)
    export = torch.load(Path(export_path), map_location="cpu", weights_only=False)
    if export.get("schema_version") != "td_rgb_export_v1" or export.get("merged") is not True:
        raise ConfigError("invalid or unmerged RGB deployment export")
    if export["backbone_sha256"] != config["backbone"]["checkpoint_sha256"]:
        raise ConfigError("deployment export backbone identity mismatch")
    backbone = load_dinov3_h16plus(
        config["backbone"]["repo_dir"],
        config["backbone"]["checkpoint"],
        config["backbone"]["checkpoint_sha256"],
    )
    block_index = int(export["selected_block"])
    head = int(export["selected_head"])
    width = int(config["backbone"]["feature_width"])
    head_dim = int(config["backbone"]["head_dim"])
    q_rows = slice(head * head_dim, (head + 1) * head_dim)
    k_rows = slice(width + head * head_dim, width + (head + 1) * head_dim)
    qkv = backbone.blocks[block_index].attn.qkv
    with torch.no_grad():
        qkv.weight[q_rows].add_(export["q_delta"].to(qkv.weight.dtype))
        qkv.weight[k_rows].add_(export["k_delta"].to(qkv.weight.dtype))
    features = DinoV3Features(
        backbone,
        tuple(int(item) for item in config["backbone"]["feature_blocks_0based"]),
    )
    predictor = DepthPredictor(
        features,
        feature_width=width,
        decoder_width=int(config["model"]["decoder_width"]),
    )
    load_trainable_state(predictor, export["decoder_state"])
    return RGBDepthModule(predictor).eval()
