"""Typed records for RFTrans training and label-isolated inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class SampleRecord:
    schema_version: str
    sample_id: str
    source: str
    release_id: str
    official_split: str
    assigned_role: str
    frame_id: int
    rgb_path: str
    depth_gt_path: str
    depth_reference_path: str
    mask_path: str
    metadata_path: str
    leakage_group_id: str
    group_confidence: str
    depth_encoding: str
    depth_coordinate: str
    invalid_rule_id: str
    mask_rule_id: str
    preprocessing_version: str
    cad_ids: tuple[str, ...] = ()
    rgb_sha256: str | None = None
    depth_sha256: str | None = None
    mask_sha256: str | None = None
    qa_status: str = "encoding_verified"

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SampleRecord:
        item = dict(value)
        item["cad_ids"] = tuple(item.get("cad_ids", ()))
        return cls(**item)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["cad_ids"] = list(self.cad_ids)
        return value


@dataclass(frozen=True)
class GeometryRecord:
    original_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    canvas_hw: tuple[int, int]
    scale_yx: tuple[float, float]
    padding_tblr: tuple[int, int, int, int]
    pixel_center_convention: str
    affine_3x3: tuple[tuple[float, float, float], ...]


@dataclass
class PatchTargets:
    state: Tensor  # int8 [Gh,Gw], -1=unknown, 0=background, 1=transparent
    median_depth_m: Tensor  # float32 [Gh,Gw], NaN where unavailable
    valid_fraction: Tensor
    transparent_fraction: Tensor
    log_depth_iqr: Tensor

    @property
    def transparent(self) -> Tensor:
        return self.state == 1

    @property
    def background(self) -> Tensor:
        return self.state == 0

    @property
    def reliable(self) -> Tensor:
        return self.state >= 0


@dataclass
class CanonicalBundle:
    rgb: Tensor
    depth_m: Tensor
    mask: Tensor
    valid_depth: Tensor
    mask_known: Tensor
    content_mask: Tensor
    geometry: GeometryRecord


@dataclass
class Sample:
    rgb: Tensor
    depth_m: Tensor
    mask: Tensor
    valid_depth: Tensor
    mask_known: Tensor
    content_mask: Tensor
    patch_targets: PatchTargets
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainBatch:
    rgb: Tensor
    depth_m: Tensor
    mask: Tensor
    valid_depth: Tensor
    mask_known: Tensor
    content_mask: Tensor
    patch_targets: list[PatchTargets]
    metadata: list[dict[str, Any]]

    def to(self, device: torch.device | str, *, non_blocking: bool = False) -> TrainBatch:
        tensor_names = ("rgb", "depth_m", "mask", "valid_depth", "mask_known", "content_mask")
        values = {
            name: getattr(self, name).to(device, non_blocking=non_blocking) for name in tensor_names
        }
        targets = [
            PatchTargets(
                **{
                    name: getattr(item, name).to(device, non_blocking=non_blocking)
                    for name in (
                        "state",
                        "median_depth_m",
                        "valid_fraction",
                        "transparent_fraction",
                        "log_depth_iqr",
                    )
                }
            )
            for item in self.patch_targets
        ]
        return TrainBatch(**values, patch_targets=targets, metadata=self.metadata)


@dataclass
class RGBSample:
    rgb: Tensor
    sample_id: str
    rgb_path: str
    geometry: GeometryRecord


def collate_samples(samples: list[Sample]) -> TrainBatch:
    if not samples:
        raise ValueError("cannot collate an empty sample list")
    tensor_names = ("rgb", "depth_m", "mask", "valid_depth", "mask_known", "content_mask")
    values = {
        name: torch.stack([getattr(item, name) for item in samples]) for name in tensor_names
    }
    return TrainBatch(
        **values,
        patch_targets=[item.patch_targets for item in samples],
        metadata=[item.metadata for item in samples],
    )
