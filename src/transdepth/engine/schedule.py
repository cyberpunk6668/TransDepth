"""World-size-independent global sample plans for strict H1/H2 pairing."""

from __future__ import annotations

import hashlib

import torch
from torch import Tensor


def build_global_schedule(
    dataset_size: int,
    updates: int,
    global_batch_size: int,
    seed: int,
) -> Tensor:
    if min(dataset_size, updates, global_batch_size) <= 0:
        raise ValueError("dataset_size, updates, and global_batch_size must be positive")
    required = updates * global_batch_size
    values: list[Tensor] = []
    produced = 0
    cycle = 0
    while produced < required:
        generator = torch.Generator().manual_seed(seed + cycle * 1_000_003)
        permutation = torch.randperm(dataset_size, generator=generator)
        take = min(required - produced, dataset_size)
        values.append(permutation[:take])
        produced += take
        cycle += 1
    return torch.cat(values).reshape(updates, global_batch_size)


def rank_slot(
    schedule: Tensor,
    update_index: int,
    microstep: int,
    rank: int,
    world_size: int,
) -> int:
    if schedule.ndim != 2:
        raise ValueError("schedule must be [updates, global_batch]")
    slot = microstep * world_size + rank
    if not 0 <= slot < schedule.shape[1]:
        raise IndexError("rank/microstep maps outside the global batch")
    return int(schedule[update_index, slot])


def schedule_sha256(schedule: Tensor) -> str:
    contiguous = schedule.detach().cpu().to(torch.int64).contiguous().numpy()
    return hashlib.sha256(contiguous.tobytes()).hexdigest()
