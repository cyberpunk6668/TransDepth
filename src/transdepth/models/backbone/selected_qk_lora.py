"""Rank-r LoRA restricted to one packed-QKV attention head's Q and K rows."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class SelectedQKLoRA(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        *,
        num_heads: int,
        selected_head: int,
        rank: int = 4,
        eta: float = 1.0,
        seed: int = 17,
    ) -> None:
        super().__init__()
        if base.out_features != 3 * base.in_features:
            raise ValueError("base projection must be packed QKV with out_features=3*in_features")
        if base.in_features % num_heads:
            raise ValueError("embedding width must be divisible by num_heads")
        if not 0 <= selected_head < num_heads or rank <= 0 or eta <= 0:
            raise ValueError("invalid selected head, rank, or eta")
        self.base = base.requires_grad_(False)
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.num_heads = num_heads
        self.selected_head = selected_head
        self.head_dim = self.in_features // num_heads
        self.rank = rank
        self.eta = float(eta)
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.q_down = nn.Linear(self.in_features, rank, bias=False, **factory_kwargs)
        self.q_up = nn.Linear(rank, self.head_dim, bias=False, **factory_kwargs)
        self.k_down = nn.Linear(self.in_features, rank, bias=False, **factory_kwargs)
        self.k_up = nn.Linear(rank, self.head_dim, bias=False, **factory_kwargs)
        generator = torch.Generator(device=base.weight.device).manual_seed(seed)
        std = 1.0 / math.sqrt(self.in_features)
        with torch.no_grad():
            self.q_down.weight.normal_(0.0, std, generator=generator)
            self.k_down.weight.normal_(0.0, std, generator=generator)
            self.q_up.weight.zero_()
            self.k_up.weight.zero_()
        self._merged = False

    @property
    def q_slice(self) -> slice:
        start = self.selected_head * self.head_dim
        return slice(start, start + self.head_dim)

    @property
    def k_slice(self) -> slice:
        start = self.in_features + self.selected_head * self.head_dim
        return slice(start, start + self.head_dim)

    def forward_with_base(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        if self._merged:
            raise RuntimeError("merged LoRA must not be applied a second time")
        base_output = self.base(inputs)
        student = base_output.clone()
        student[..., self.q_slice] = (
            student[..., self.q_slice] + self.eta * self.q_up(self.q_down(inputs))
        )
        student[..., self.k_slice] = (
            student[..., self.k_slice] + self.eta * self.k_up(self.k_down(inputs))
        )
        return student, base_output

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_with_base(inputs)[0]

    @torch.no_grad()
    def merge_into_base_(self) -> nn.Linear:
        if self._merged:
            raise RuntimeError("LoRA has already been merged")
        q_delta = self.eta * (self.q_up.weight @ self.q_down.weight)
        k_delta = self.eta * (self.k_up.weight @ self.k_down.weight)
        self.base.weight[self.q_slice].add_(q_delta.to(self.base.weight.dtype))
        self.base.weight[self.k_slice].add_(k_delta.to(self.base.weight.dtype))
        self._merged = True
        return self.base
