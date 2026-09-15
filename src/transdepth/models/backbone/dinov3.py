"""Strict Meta-native DINOv3 H+/16 loading and four-layer dense features."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from transdepth.models.backbone.attention_adapter import FARSelfAttention
from transdepth.models.types import AttentionTrace

DINOV3_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_dinov3_h16plus(
    repo_dir: str | Path,
    checkpoint: str | Path,
    expected_sha256: str,
) -> nn.Module:
    repo = Path(repo_dir).expanduser().resolve(strict=True)
    weight = Path(checkpoint).expanduser().resolve(strict=True)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != DINOV3_COMMIT:
        raise RuntimeError(f"DINOv3 source commit mismatch: {commit}")
    status = subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip()
    if status:
        raise RuntimeError("DINOv3 vendor source is dirty")
    if len(expected_sha256) != 64 or _sha256(weight) != expected_sha256.lower():
        raise RuntimeError("DINOv3 checkpoint SHA256 mismatch")
    sys.path.insert(0, str(repo))
    try:
        from dinov3.hub.backbones import dinov3_vith16plus

        model = dinov3_vith16plus(pretrained=False)
    finally:
        if sys.path[0] == str(repo):
            sys.path.pop(0)
    state = torch.load(weight, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or "blocks.0.attn.qkv.weight" not in state:
        raise RuntimeError("checkpoint is not a native DINOv3 backbone state_dict")
    model.load_state_dict(state, strict=True)
    del state
    verify_h16plus_structure(model)
    return model.requires_grad_(False).eval()


def verify_h16plus_structure(model: nn.Module) -> None:
    expected = {
        "embed_dim": 1280,
        "n_blocks": 32,
        "num_heads": 20,
        "patch_size": 16,
        "n_storage_tokens": 4,
    }
    for name, value in expected.items():
        if getattr(model, name, None) != value:
            raise RuntimeError(f"DINOv3 identity mismatch: {name}")
    if len(model.blocks) != 32 or tuple(model.storage_tokens.shape) != (1, 4, 1280):
        raise RuntimeError("DINOv3 block or storage-token identity mismatch")
    if model.rope_embed.dtype != torch.float32 or model.rope_embed.normalize_coords != "separate":
        raise RuntimeError("DINOv3 RoPE identity mismatch")
    rope = model.rope_embed
    if (
        rope.base != 100
        or rope.rescale_coords != 2
        or rope.shift_coords is not None
        or rope.jitter_coords is not None
    ):
        raise RuntimeError("DINOv3 RoPE parameter mismatch")
    for block in model.blocks:
        attention = block.attn
        if attention.num_heads != 20 or tuple(attention.qkv.weight.shape) != (3840, 1280):
            raise RuntimeError("DINOv3 packed QKV shape mismatch")
        if attention.qkv.bias is None or not hasattr(attention.qkv, "bias_mask"):
            raise RuntimeError("DINOv3 masked key-bias projection is missing")
        bias_mask = attention.qkv.bias_mask
        if not (
            bool((bias_mask[:1280] == 1).all())
            and bool((bias_mask[1280:2560] == 0).all())
            and bool((bias_mask[2560:] == 1).all())
        ):
            raise RuntimeError("DINOv3 key-bias mask mismatch")
        if block.norm1.eps != 1e-5 or block.norm2.eps != 1e-5:
            raise RuntimeError("DINOv3 LayerNorm epsilon mismatch")
        mlp = block.mlp
        if not all(hasattr(mlp, name) for name in ("w1", "w2", "w3")):
            raise RuntimeError("DINOv3 H+/16 must use SwiGLU")
        if (
            mlp.w1.out_features != 5120
            or mlp.w2.out_features != 5120
            or mlp.w3.in_features != 5120
        ):
            raise RuntimeError("DINOv3 SwiGLU hidden width mismatch")


class DinoV3Features(nn.Module):
    feature_blocks = (7, 15, 23, 31)

    def __init__(
        self, backbone: nn.Module, feature_blocks: tuple[int, ...] = (7, 15, 23, 31)
    ) -> None:
        super().__init__()
        verify_h16plus_structure(backbone)
        if feature_blocks != self.feature_blocks:
            raise ValueError("this locked H+/16 protocol requires feature blocks 7,15,23,31")
        self.backbone = backbone.requires_grad_(False).eval()
        self.adapter: FARSelfAttention | None = None
        self.activation_checkpoint_suffix = True

    def enable_lora(
        self,
        *,
        block_index: int,
        head_index: int,
        rank: int = 4,
        eta: float = 1.0,
        seed: int = 17,
    ) -> FARSelfAttention:
        if self.adapter is not None:
            raise RuntimeError("only one attention adapter is allowed")
        if not 0 <= block_index < len(self.backbone.blocks):
            raise IndexError("selected block is outside the DINOv3 backbone")
        block = self.backbone.blocks[block_index]
        adapter = FARSelfAttention(
            block.attn,
            selected_block=block_index,
            selected_head=head_index,
            rank=rank,
            eta=eta,
            seed=seed,
            prefix_tokens=1 + self.backbone.n_storage_tokens,
        )
        block.attn = adapter
        self.adapter = adapter
        return adapter

    def disable_lora(self) -> None:
        if self.adapter is None:
            return
        block = self.backbone.blocks[self.adapter.selected_block]
        block.attn = self.adapter.restore_original()
        self.adapter = None

    def train(self, mode: bool = True) -> DinoV3Features:
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(
        self, rgb: Tensor, *, return_trace: bool = False
    ) -> tuple[tuple[Tensor, ...], AttentionTrace | None]:
        if rgb.ndim != 4 or rgb.shape[1:] != (3, 384, 512):
            raise ValueError("DINOv3 input must be [B,3,384,512]")
        self.backbone.eval()
        adapter_block = (
            self.adapter.selected_block if self.adapter is not None else len(self.backbone.blocks)
        )
        if return_trace and self.adapter is None:
            raise RuntimeError("attention trace requires an enabled LoRA adapter")
        if self.adapter is not None:
            self.adapter.set_capture_trace(return_trace)
        with torch.no_grad():
            tokens, (grid_h, grid_w) = self.backbone.prepare_tokens_with_masks(rgb, masks=None)
            rope = self.backbone.rope_embed(H=grid_h, W=grid_w)
        features: list[Tensor] = []
        for index, block in enumerate(self.backbone.blocks):
            if index < adapter_block:
                with torch.no_grad():
                    tokens = block(tokens, rope)
            elif (
                self.training
                and self.activation_checkpoint_suffix
                and index > adapter_block
            ):
                def run_block(inputs: Tensor, current_block: nn.Module = block) -> Tensor:
                    return current_block(inputs, rope)

                tokens = activation_checkpoint(run_block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens, rope)
            if index in self.feature_blocks:
                normalized = self.backbone.norm(tokens)
                patches = normalized[:, 1 + self.backbone.n_storage_tokens :]
                patches = patches.transpose(1, 2).reshape(
                    rgb.shape[0], 1280, grid_h, grid_w
                )
                features.append(patches)
        if len(features) != 4 or (grid_h, grid_w) != (24, 32):
            raise RuntimeError("unexpected DINOv3 dense feature geometry")
        trace = self.adapter.consume_trace() if return_trace and self.adapter is not None else None
        return tuple(features), trace
