"""DINOv3 SelfAttention-compatible selected-head Q/K adapter and trace capture."""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from transdepth.data.schema import PatchTargets
from transdepth.far.teacher import teacher_rows
from transdepth.models.backbone.selected_qk_lora import SelectedQKLoRA
from transdepth.models.types import AttentionTrace


def _rotate_half(tensor: Tensor) -> Tensor:
    first, second = tensor.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _rope_apply(tensor: Tensor, sine: Tensor, cosine: Tensor) -> Tensor:
    return tensor * cosine + _rotate_half(tensor) * sine


def apply_dinov3_rope(
    q: Tensor, k: Tensor, rope: tuple[Tensor, Tensor]
) -> tuple[Tensor, Tensor]:
    q_dtype, k_dtype = q.dtype, k.dtype
    sine, cosine = rope
    token_count = q.shape[-2]
    prefix = token_count - sine.shape[-2]
    if prefix < 0:
        raise ValueError("RoPE spatial sequence is longer than the token sequence")
    q_spatial = _rope_apply(q[:, :, prefix:].to(sine.dtype), sine, cosine)
    k_spatial = _rope_apply(k[:, :, prefix:].to(sine.dtype), sine, cosine)
    q = torch.cat((q[:, :, :prefix].to(sine.dtype), q_spatial), dim=-2).to(q_dtype)
    k = torch.cat((k[:, :, :prefix].to(sine.dtype), k_spatial), dim=-2).to(k_dtype)
    return q, k


@dataclass(frozen=True)
class OracleContext:
    targets: tuple[PatchTargets, ...]
    beta: float
    sigma_z: float
    cmax: float
    mass_eps: float
    prefix_tokens: int


class FARSelfAttention(nn.Module):
    """Drop-in adapter for the eval path of Meta's DINOv3 SelfAttention."""

    def __init__(
        self,
        original: nn.Module,
        *,
        selected_block: int,
        selected_head: int,
        rank: int = 4,
        eta: float = 1.0,
        seed: int = 17,
        prefix_tokens: int = 5,
    ) -> None:
        super().__init__()
        required = ("qkv", "proj", "proj_drop", "num_heads")
        if any(not hasattr(original, name) for name in required):
            raise TypeError("attention module does not match the DINOv3 SelfAttention contract")
        object.__setattr__(self, "_original_shell", original)
        self.num_heads = int(original.num_heads)
        self.scale = float(original.scale)
        self.qkv = SelectedQKLoRA(
            original.qkv,
            num_heads=self.num_heads,
            selected_head=selected_head,
            rank=rank,
            eta=eta,
            seed=seed,
        )
        self.proj = original.proj
        self.proj_drop = original.proj_drop
        self.selected_block = selected_block
        self.prefix_tokens = prefix_tokens
        self.capture_trace = False
        self._latest_trace: AttentionTrace | None = None
        self._oracle_context: OracleContext | None = None

    @property
    def selected_head(self) -> int:
        return self.qkv.selected_head

    def set_capture_trace(self, enabled: bool) -> None:
        self.capture_trace = enabled
        if not enabled:
            self._latest_trace = None

    def consume_trace(self) -> AttentionTrace:
        if self._latest_trace is None:
            raise RuntimeError("no fresh attention trace is available")
        trace = self._latest_trace
        self._latest_trace = None
        return trace

    @contextmanager
    def oracle(
        self,
        targets: list[PatchTargets],
        *,
        beta: float,
        sigma_z: float = 0.1,
        cmax: float = 4.0,
        mass_eps: float = 1e-8,
    ) -> Iterator[None]:
        if self._oracle_context is not None:
            raise RuntimeError("nested oracle contexts are forbidden")
        self._oracle_context = OracleContext(
            targets=tuple(targets),
            beta=beta,
            sigma_z=sigma_z,
            cmax=cmax,
            mass_eps=mass_eps,
            prefix_tokens=self.prefix_tokens,
        )
        try:
            yield
        finally:
            self._oracle_context = None

    def _reshape_qkv(self, packed: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, tokens, _ = packed.shape
        width = self.qkv.in_features
        shaped = packed.reshape(batch, tokens, 3, self.num_heads, width // self.num_heads)
        q, k, v = torch.unbind(shaped, dim=2)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def _apply_oracle(self, output: Tensor, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        context = self._oracle_context
        if context is None:
            return output
        if len(context.targets) != q.shape[0]:
            raise ValueError("oracle target batch does not match RGB batch")
        result = output.clone()
        head = self.selected_head
        scale = math.sqrt(q.shape[-1])
        for batch_index, target in enumerate(context.targets):
            states = target.state.flatten().to(q.device)
            depths = target.median_depth_m.flatten().to(q.device)
            if context.prefix_tokens + states.numel() != q.shape[-2]:
                raise ValueError("oracle patch grid does not match token sequence")
            query_patch = torch.nonzero(states == 1, as_tuple=False).flatten()
            key_patch = torch.nonzero(states >= 0, as_tuple=False).flatten()
            if query_patch.numel() == 0 or key_patch.numel() == 0:
                continue
            query_tokens = query_patch + context.prefix_tokens
            key_tokens = key_patch + context.prefix_tokens
            with torch.autocast(device_type=q.device.type, enabled=False):
                logits = (
                    q[batch_index, head, query_tokens].float()
                    @ k[batch_index, head].float().transpose(-2, -1)
                ) / scale
                teacher = teacher_rows(
                    logits,
                    key_tokens,
                    depths[query_patch],
                    depths[key_patch],
                    torch.ones_like(query_patch, dtype=torch.bool),
                    beta=context.beta,
                    sigma_z=context.sigma_z,
                    cmax=context.cmax,
                    mass_eps=context.mass_eps,
                )
                changed = teacher @ v[batch_index, head].float()
            result[batch_index, head, query_tokens] = changed.to(result.dtype)
        return result

    def compute_attention(
        self,
        student_packed: Tensor,
        base_packed: Tensor,
        *,
        attn_bias: Tensor | None = None,
        rope: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        if attn_bias is not None:
            raise ValueError("DINOv3 H+/16 FAR path does not support attention bias")
        q_student, k_student, v_student = self._reshape_qkv(student_packed)
        q_base, k_base, v_base = self._reshape_qkv(base_packed)
        if rope is not None:
            q_student, k_student = apply_dinov3_rope(q_student, k_student, rope)
            q_base, k_base = apply_dinov3_rope(q_base, k_base, rope)
        if not torch.equal(v_student, v_base):
            raise RuntimeError("selected Q/K LoRA unexpectedly changed V")
        output = F.scaled_dot_product_attention(q_student, k_student, v_student)
        output = self._apply_oracle(output, q_base, k_base, v_base)
        if self.capture_trace:
            selected = slice(self.selected_head, self.selected_head + 1)
            self._latest_trace = AttentionTrace(
                q_student=q_student[:, selected],
                k_student=k_student[:, selected],
                q_base=q_base[:, selected].detach(),
                k_base=k_base[:, selected].detach(),
                selected_block=self.selected_block,
                selected_heads=(self.selected_head,),
                prefix_tokens=self.prefix_tokens,
            )
        output = output.transpose(1, 2).reshape(
            student_packed.shape[0], student_packed.shape[1], -1
        )
        return output

    def forward(self, inputs: Tensor, attn_bias=None, rope=None) -> Tensor:
        self._latest_trace = None
        student, base = self.qkv.forward_with_base(inputs)
        output = self.compute_attention(student, base, attn_bias=attn_bias, rope=rope)
        return self.proj_drop(self.proj(output))

    def forward_list(self, inputs, attn_bias=None, rope_list=None):
        if len(inputs) != len(rope_list):
            raise ValueError("input and RoPE lists must have equal length")
        return [
            self.forward(item, attn_bias=attn_bias, rope=rope)
            for item, rope in zip(inputs, rope_list, strict=True)
        ]

    def restore_original(self) -> nn.Module:
        original = self._original_shell
        original.qkv = self.qkv.base
        original.proj = self.proj
        original.proj_drop = self.proj_drop
        return original
