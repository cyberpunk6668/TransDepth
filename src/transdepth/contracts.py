"""Public tensor contracts shared across data, model, and engine modules."""

from transdepth.data.schema import (
    CanonicalBundle,
    GeometryRecord,
    PatchTargets,
    RGBSample,
    Sample,
    SampleRecord,
    TrainBatch,
)
from transdepth.models.types import AttentionTrace, Prediction

__all__ = [
    "CanonicalBundle",
    "AttentionTrace",
    "GeometryRecord",
    "PatchTargets",
    "Prediction",
    "RGBSample",
    "Sample",
    "SampleRecord",
    "TrainBatch",
]
