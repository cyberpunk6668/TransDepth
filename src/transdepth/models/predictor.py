"""RGB-only DINOv3 H+/16 + DPT-A metric-depth predictor."""

from __future__ import annotations

from torch import Tensor, nn

from transdepth.models.decoders.dpt import DPTFuse
from transdepth.models.heads import DepthTail
from transdepth.models.reassembly import ReassemblePyramid
from transdepth.models.types import Prediction


class DepthPredictor(nn.Module):
    def __init__(
        self,
        features: nn.Module,
        *,
        feature_width: int = 1280,
        decoder_width: int = 128,
    ) -> None:
        super().__init__()
        self.features = features
        self.reassembly = ReassemblePyramid(feature_width, decoder_width)
        self.decoder = DPTFuse(decoder_width)
        self.tail = DepthTail(decoder_width)

    def forward(self, rgb: Tensor, *, return_aux: bool = False) -> Prediction:
        feature_maps, trace = self.features(rgb, return_trace=return_aux)
        depth = self.tail(self.decoder(self.reassembly(feature_maps)))
        return Prediction(depth_m=depth, trace=trace)


class RGBDepthModule(nn.Module):
    """Deployment signature deliberately exposes only forward(rgb)->depth_m."""

    def __init__(self, predictor: DepthPredictor) -> None:
        super().__init__()
        self.predictor = predictor

    def forward(self, rgb: Tensor) -> Tensor:
        return self.predictor(rgb, return_aux=False).depth_m
