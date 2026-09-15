"""Model-facing tensor contracts."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class AttentionTrace:
    q_student: Tensor
    k_student: Tensor
    q_base: Tensor
    k_base: Tensor
    selected_block: int
    selected_heads: tuple[int, ...]
    prefix_tokens: int = 5


@dataclass
class Prediction:
    depth_m: Tensor
    seg_logits: Tensor | None = None
    trace: AttentionTrace | None = None
