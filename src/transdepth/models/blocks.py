"""Locked decoder_spec_v1 convolution blocks."""

from __future__ import annotations

from torch import Tensor, nn


def conv_gn_act(channels_in: int, channels_out: int, stride: int = 1) -> nn.Sequential:
    if channels_out % 8:
        raise ValueError("decoder_spec_v1 requires channels divisible by 8")
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 3, stride=stride, padding=1, bias=False),
        nn.GroupNorm(8, channels_out),
        nn.GELU(approximate="none"),
    )


class ResidualBlock(nn.Module):
    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            conv_gn_act(channels, channels),
            conv_gn_act(channels, channels),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs + self.body(inputs)
