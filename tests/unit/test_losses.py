import math

import torch

from transdepth.losses.depth import balanced_log_huber_loss


def test_log_huber_matches_documented_two_x_case() -> None:
    target = torch.ones(1, 1, 1, 1)
    prediction = 2.0 * target
    result = balanced_log_huber_loss(
        prediction,
        target,
        torch.ones_like(target, dtype=torch.bool),
        torch.ones_like(target, dtype=torch.uint8),
    )
    expected = 0.1 * (math.log(2.0) - 0.05)
    assert torch.isclose(result.loss, torch.tensor(expected), atol=1e-7)


def test_transparent_and_background_are_equally_weighted() -> None:
    target = torch.ones(1, 1, 1, 101)
    prediction = target.clone()
    prediction[..., 0] = 2.0
    prediction[..., 1:] = 1.1
    mask = torch.zeros_like(target, dtype=torch.uint8)
    mask[..., 0] = 1
    valid = torch.ones_like(target, dtype=torch.bool)
    result = balanced_log_huber_loss(prediction, target, valid, mask)
    transparent = 0.1 * (math.log(2.0) - 0.05)
    background = 0.5 * math.log(1.1) ** 2
    assert torch.isclose(result.loss, torch.tensor((transparent + background) / 2), atol=1e-7)
