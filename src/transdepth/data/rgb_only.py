"""Label-isolated RGB dataset for ClearGrasp or arbitrary inference images."""

from __future__ import annotations

import hashlib
from pathlib import Path

from torch.utils.data import Dataset

from transdepth.data.depth_io import read_rgb_u8
from transdepth.data.geometry import letterbox_rgb
from transdepth.data.manifest import read_manifest
from transdepth.data.schema import RGBSample, SampleRecord

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def discover_cleargrasp_rgb(root: str | Path) -> list[Path]:
    base = Path(root).expanduser().resolve(strict=True)
    candidates = []
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        name = path.name.lower()
        if name.endswith("-transparent-rgb-img.jpg") or name.endswith("-rgb.jpg"):
            candidates.append(path)
    return sorted(candidates)


class RgbOnlyDataset(Dataset[RGBSample]):
    def __init__(self, root: str | Path, paths: list[str | Path] | None = None) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        self.paths = (
            [Path(item).expanduser().resolve(strict=True) for item in paths]
            if paths is not None
            else discover_cleargrasp_rgb(self.root)
        )
        if not self.paths:
            raise ValueError(f"no ClearGrasp RGB inference images found under {self.root}")
        for path in self.paths:
            if not path.is_relative_to(self.root):
                raise ValueError(f"RGB path escapes inference root: {path}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> RGBSample:
        path = self.paths[index]
        rgb, _, geometry = letterbox_rgb(read_rgb_u8(path))
        relative = path.relative_to(self.root).as_posix()
        sample_id = "cg_" + hashlib.sha256(relative.encode()).hexdigest()[:20]
        return RGBSample(rgb=rgb, sample_id=sample_id, rgb_path=relative, geometry=geometry)


class ManifestRgbOnlyDataset(Dataset[RGBSample]):
    """Read only RGB fields from a supervised manifest; never resolve GT paths."""

    def __init__(self, manifest: str | Path, root: str | Path, allowed_role: str) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        self.records: list[SampleRecord] = read_manifest(manifest, {allowed_role})
        if not self.records:
            raise ValueError(f"manifest has no RGB records for {allowed_role}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> RGBSample:
        record = self.records[index]
        path = (self.root / record.rgb_path).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError(f"RGB path escapes source root: {record.rgb_path}")
        rgb, _, geometry = letterbox_rgb(read_rgb_u8(path))
        return RGBSample(
            rgb=rgb,
            sample_id=record.sample_id,
            rgb_path=record.rgb_path,
            geometry=geometry,
        )
