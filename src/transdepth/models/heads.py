"""Metric-positive full-resolution depth output."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


def positive_depth(raw: Tensor, epsilon: float = 1e-6) -> Tensor:
    return F.softplus(raw.float()) + epsilon


class DepthTail(nn.Module):
    def __init__(self, channels: int = 128, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, padding=1, bias=True)
        self.activation = nn.GELU(approximate="none")
        self.conv2 = nn.Conv2d(32, 1, 3, padding=1, bias=True)
        self.epsilon = epsilon

    def forward(self, inputs: Tensor) -> Tensor:
        output = F.interpolate(inputs, size=(192, 256), mode="bilinear", align_corners=False)
        output = self.conv2(self.activation(self.conv1(output)))
        output = F.interpolate(output, size=(384, 512), mode="bilinear", align_corners=False)
        return positive_depth(output, self.epsilon)
