"""Project four same-grid DINO layers into a trainable four-scale pyramid."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn


class ReassemblePyramid(nn.Module):
    def __init__(self, feature_width: int = 1280, decoder_width: int = 128) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Conv2d(feature_width, decoder_width, 1, bias=True) for _ in range(4)]
        )
        self.scale1 = nn.Conv2d(decoder_width, decoder_width, 3, padding=1, bias=True)
        self.scale2 = nn.Conv2d(decoder_width, decoder_width, 3, padding=1, bias=True)
        self.scale4 = nn.Conv2d(decoder_width, decoder_width, 3, stride=2, padding=1, bias=True)

    def forward(self, features: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        if len(features) != 4 or any(item.shape[-2:] != (24, 32) for item in features):
            raise ValueError("reassembly requires four [B,C,24,32] feature maps")
        projected = [
            layer(item) for layer, item in zip(self.projections, features, strict=True)
        ]
        p1 = self.scale1(
            F.interpolate(projected[0], size=(96, 128), mode="bilinear", align_corners=False)
        )
        p2 = self.scale2(
            F.interpolate(projected[1], size=(48, 64), mode="bilinear", align_corners=False)
        )
        p3 = projected[2]
        p4 = self.scale4(projected[3])
        return p1, p2, p3, p4
