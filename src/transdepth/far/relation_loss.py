"""Full-key KL(P||A_student) with balanced transparent/background queries."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from transdepth.data.schema import PatchTargets
from transdepth.far.teacher import teacher_rows
from transdepth.models.types import AttentionTrace


@dataclass(frozen=True)
class RelationLossResult:
    loss: Tensor
    per_image: Tensor
    transparent_queries: tuple[int, ...]
    background_queries: tuple[int, ...]
    reliable_keys: tuple[int, ...]


def _sample(indices: Tensor, maximum: int, seed: int) -> Tensor:
    if indices.numel() <= maximum:
        return indices
    generator = torch.Generator(device=indices.device).manual_seed(seed)
    order = torch.randperm(indices.numel(), generator=generator, device=indices.device)[:maximum]
    return indices[order]


def relation_kl_loss(
    trace: AttentionTrace,
    targets: list[PatchTargets],
    *,
    beta: float,
    sigma_z: float = 0.1,
    cmax: float = 4.0,
    mass_eps: float = 1e-8,
    max_queries_per_class: int = 128,
    query_seed: int = 17,
) -> RelationLossResult:
    if (
        trace.q_student.shape != trace.k_student.shape
        or trace.q_base.shape != trace.q_student.shape
    ):
        raise ValueError("student/base Q/K trace shapes must match")
    if trace.k_base.shape != trace.q_student.shape or trace.q_student.ndim != 4:
        raise ValueError("attention traces must be [B,H,N,D]")
    batch, heads, token_count, head_dim = trace.q_student.shape
    if len(targets) != batch or heads != len(trace.selected_heads):
        raise ValueError("trace metadata does not match targets or selected heads")
    if max_queries_per_class <= 0:
        raise ValueError("max_queries_per_class must be positive")

    per_image: list[Tensor] = []
    transparent_counts: list[int] = []
    background_counts: list[int] = []
    key_counts: list[int] = []
    scale = math.sqrt(head_dim)
    for batch_index, patch_target in enumerate(targets):
        states = patch_target.state.flatten().to(trace.q_student.device)
        depths = patch_target.median_depth_m.flatten().to(trace.q_student.device)
        if trace.prefix_tokens + states.numel() != token_count:
            raise ValueError("patch grid plus prefix does not match the attention sequence")
        key_patch_ids = torch.nonzero(states >= 0, as_tuple=False).flatten()
        transparent_patch_ids = torch.nonzero(states == 1, as_tuple=False).flatten()
        background_patch_ids = torch.nonzero(states == 0, as_tuple=False).flatten()
        transparent_patch_ids = _sample(
            transparent_patch_ids, max_queries_per_class, query_seed + 2 * batch_index
        )
        background_patch_ids = _sample(
            background_patch_ids, max_queries_per_class, query_seed + 2 * batch_index + 1
        )
        query_patch_ids = torch.cat((transparent_patch_ids, background_patch_ids))
        transparent_counts.append(int(transparent_patch_ids.numel()))
        background_counts.append(int(background_patch_ids.numel()))
        key_counts.append(int(key_patch_ids.numel()))
        if query_patch_ids.numel() == 0 or key_patch_ids.numel() == 0:
            per_image.append(trace.q_student[batch_index].sum() * 0.0)
            continue
        query_token_ids = query_patch_ids + trace.prefix_tokens
        key_token_ids = key_patch_ids + trace.prefix_tokens
        query_depths = depths[query_patch_ids]
        key_depths = depths[key_patch_ids]
        query_is_transparent = states[query_patch_ids] == 1
        head_losses: list[Tensor] = []
        for head_index in range(heads):
            q_base = trace.q_base[batch_index, head_index, query_token_ids].float()
            k_base = trace.k_base[batch_index, head_index].float()
            base_logits = (q_base @ k_base.transpose(-2, -1)) / scale
            teacher = teacher_rows(
                base_logits,
                key_token_ids,
                query_depths,
                key_depths,
                query_is_transparent,
                beta=beta,
                sigma_z=sigma_z,
                cmax=cmax,
                mass_eps=mass_eps,
            )
            q_student = trace.q_student[batch_index, head_index, query_token_ids].float()
            k_student = trace.k_student[batch_index, head_index].float()
            student_logits = (q_student @ k_student.transpose(-2, -1)) / scale
            log_attention = student_logits.log_softmax(dim=-1)
            elementwise = torch.special.xlogy(teacher, teacher) - teacher * log_attention
            row_loss = elementwise.sum(dim=-1)
            class_losses = []
            for is_transparent in (True, False):
                selected = query_is_transparent == is_transparent
                if bool(selected.any()):
                    class_losses.append(row_loss[selected].mean())
            head_losses.append(torch.stack(class_losses).mean())
        per_image.append(torch.stack(head_losses).mean())
    stacked = torch.stack(per_image)
    return RelationLossResult(
        loss=stacked.mean(),
        per_image=stacked,
        transparent_queries=tuple(transparent_counts),
        background_queries=tuple(background_counts),
        reliable_keys=tuple(key_counts),
    )
