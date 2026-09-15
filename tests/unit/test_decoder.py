import torch
from torch import nn

from transdepth.models.predictor import DepthPredictor, RGBDepthModule


class FakeFeatures(nn.Module):
    def forward(self, rgb, *, return_trace=False):
        batch = rgb.shape[0]
        features = tuple(torch.randn(batch, 32, 24, 32, device=rgb.device) for _ in range(4))
        return features, None


def test_dpt_predictor_is_positive_rgb_only_and_full_resolution() -> None:
    predictor = DepthPredictor(FakeFeatures(), feature_width=32, decoder_width=16)
    deployed = RGBDepthModule(predictor)
    rgb = torch.randn(1, 3, 384, 512)
    depth = deployed(rgb)
    assert depth.shape == (1, 1, 384, 512)
    assert torch.isfinite(depth).all()
    assert (depth > 0).all()
