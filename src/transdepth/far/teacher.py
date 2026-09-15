"""Mass-preserving, depth-conditioned FAR attention teacher."""

from __future__ import annotations

import torch
from torch import Tensor


@torch.no_grad()
def teacher_rows(
    base_logits: Tensor,
    key_ids: Tensor,
    query_depth_m: Tensor,
    key_depth_m: Tensor,
    transparent_query: Tensor,
    *,
    beta: float,
    sigma_z: float = 0.1,
    cmax: float = 4.0,
    mass_eps: float = 1e-8,
) -> Tensor:
    if base_logits.ndim != 2:
        raise ValueError("base_logits must be [Q,N]")
    query_count, key_count = base_logits.shape
    if query_depth_m.shape != (query_count,) or transparent_query.shape != (query_count,):
        raise ValueError("query metadata does not match base_logits")
    if key_ids.ndim != 1 or key_depth_m.shape != key_ids.shape:
        raise ValueError("key IDs and depths must be one-dimensional and aligned")
    if not 0.0 <= beta <= 1.0 or sigma_z <= 0 or cmax < 0 or mass_eps < 0:
        raise ValueError("invalid FAR teacher hyperparameters")
    if key_ids.numel() and (int(key_ids.min()) < 0 or int(key_ids.max()) >= key_count):
        raise IndexError("reliable key index outside full attention sequence")
    if key_ids.unique().numel() != key_ids.numel():
        raise ValueError("reliable key IDs must be unique")

    logits = base_logits.float()
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("non-finite original attention logits")
    base_attention = logits.softmax(dim=-1)
    if key_ids.numel() == 0 or beta == 0:
        return base_attention
    if not bool(torch.isfinite(query_depth_m).all()) or not bool((query_depth_m > 0).all()):
        raise FloatingPointError("query depths must be finite and positive")
    if not bool(torch.isfinite(key_depth_m).all()) or not bool((key_depth_m > 0).all()):
        raise FloatingPointError("key depths must be finite and positive")

    cost = (
        (query_depth_m.float().log()[:, None] - key_depth_m.float().log()[None, :]).square()
        / (2.0 * sigma_z**2)
    ).clamp_max(cmax)
    conditioned = (logits[:, key_ids] - cost).softmax(dim=-1)
    mass = base_attention[:, key_ids].sum(dim=-1, keepdim=True)
    use = transparent_query.bool() & (mass[:, 0] > mass_eps)
    use &= torch.isfinite(conditioned).all(dim=-1)
    candidate = (1.0 - beta) * base_attention[:, key_ids] + beta * mass * conditioned
    result = base_attention.clone()
    result[:, key_ids] = torch.where(use[:, None], candidate, base_attention[:, key_ids])
    return result
