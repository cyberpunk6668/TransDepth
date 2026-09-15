"""Atomic trainable-only checkpoints referencing the immutable DINOv3 backbone."""

from __future__ import annotations

import os
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
import torch
from torch import nn


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def load_trainable_state(
    model: nn.Module,
    state: dict[str, torch.Tensor],
    *,
    allowed_missing_substrings: tuple[str, ...] = (),
) -> None:
    parameters = dict(model.named_parameters())
    trainable = {name for name, parameter in parameters.items() if parameter.requires_grad}
    missing = [name for name in state if name not in parameters]
    if missing:
        raise KeyError(f"checkpoint parameters do not exist in model: {missing[:5]}")
    omitted = trainable - set(state)
    forbidden_omitted = [
        name
        for name in omitted
        if not any(token in name for token in allowed_missing_substrings)
    ]
    if forbidden_omitted:
        raise KeyError(f"checkpoint omits trainable parameters: {forbidden_omitted[:5]}")
    with torch.no_grad():
        for name, value in state.items():
            parameter = parameters[name]
            if parameter.shape != value.shape:
                raise ValueError(f"checkpoint shape mismatch for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"] is not None:
        torch.cuda.set_rng_state(state["torch_cuda"])


def atomic_torch_save(value: dict[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(destination)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or checkpoint.get("schema_version") != "td_checkpoint_v1":
        raise ValueError("unsupported or invalid TransDepth checkpoint")
    return checkpoint
