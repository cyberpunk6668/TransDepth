"""Execute normal and privileged oracle predictions on preloaded RFTrans samples."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import torch
from torch import nn

from transdepth.data.schema import Sample
from transdepth.evaluation.metrics import evaluate_image
from transdepth.models.backbone.attention_adapter import FARSelfAttention


def stable_sample_indices(sample_ids: list[str], maximum: int, seed: int) -> list[int]:
    ranked = sorted(
        range(len(sample_ids)),
        key=lambda index: hashlib.sha256(f"{seed}:{sample_ids[index]}".encode()).digest(),
    )
    return ranked[:maximum]


def _metrics(sample: Sample, prediction: np.ndarray) -> dict[str, Any]:
    return evaluate_image(
        prediction,
        sample.depth_m[0].numpy(),
        sample.valid_depth[0].numpy(),
        sample.mask[0].numpy(),
        sample.content_mask[0].numpy(),
    )


@torch.no_grad()
def baseline_predictions(
    model: nn.Module, samples: list[Sample], device: torch.device
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = []
    predictions = {}
    model.eval()
    for sample in samples:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(sample.rgb[None].to(device), return_aux=False).depth_m
        prediction = output[0, 0].float().cpu().numpy()
        sample_id = sample.metadata["sample_id"]
        predictions[sample_id] = prediction
        rows.append({"sample_id": sample_id, "metrics": _metrics(sample, prediction)})
    return rows, predictions


@torch.no_grad()
def oracle_predictions(
    model: nn.Module,
    adapter: FARSelfAttention,
    samples: list[Sample],
    device: torch.device,
    *,
    beta: float,
    sigma_z: float,
    cmax: float,
    mass_eps: float,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rows = []
    predictions = {}
    model.eval()
    for sample in samples:
        target = sample.patch_targets
        target = type(target)(
            state=target.state.to(device),
            median_depth_m=target.median_depth_m.to(device),
            valid_fraction=target.valid_fraction.to(device),
            transparent_fraction=target.transparent_fraction.to(device),
            log_depth_iqr=target.log_depth_iqr.to(device),
        )
        with adapter.oracle(
            [target], beta=beta, sigma_z=sigma_z, cmax=cmax, mass_eps=mass_eps
        ):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(sample.rgb[None].to(device), return_aux=False).depth_m
        prediction = output[0, 0].float().cpu().numpy()
        sample_id = sample.metadata["sample_id"]
        predictions[sample_id] = prediction
        rows.append({"sample_id": sample_id, "metrics": _metrics(sample, prediction)})
    return rows, predictions
