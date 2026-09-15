"""Architecture A: coarse-to-fine DPT-style residual fusion."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from transdepth.models.blocks import ResidualBlock


class DPTFuse(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.coarse = ResidualBlock(channels)
        self.lateral = nn.ModuleList([ResidualBlock(channels) for _ in range(3)])
        self.fusion = nn.ModuleList([ResidualBlock(channels) for _ in range(3)])

    def forward(self, pyramid: tuple[Tensor, ...]) -> Tensor:
        if len(pyramid) != 4:
            raise ValueError("DPTFuse requires four pyramid levels")
        output = self.coarse(pyramid[3])
        for index in (2, 1, 0):
            lateral = self.lateral[index](pyramid[index])
            coarse = F.interpolate(
                output, size=lateral.shape[-2:], mode="bilinear", align_corners=False
            )
            output = self.fusion[index](lateral + coarse)
        return output
