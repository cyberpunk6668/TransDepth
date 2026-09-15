"""RFTrans-only supervised dataset; model inputs remain RGB-only."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from transdepth.data.depth_io import read_rftrans_depth, read_rftrans_mask, read_rgb_u8
from transdepth.data.geometry import letterbox_bundle
from transdepth.data.manifest import read_manifest
from transdepth.data.patches import build_patch_targets
from transdepth.data.schema import Sample, SampleRecord


class RFTransDataset(Dataset[Sample]):
    def __init__(
        self,
        manifest: str | Path,
        root: str | Path,
        allowed_roles: set[str],
        *,
        canvas_hw: tuple[int, int] = (384, 512),
    ) -> None:
        disallowed = {role for role in allowed_roles if not role.startswith("R_")}
        if disallowed:
            raise ValueError(f"RFTrans training dataset rejects non-RFTrans roles: {disallowed}")
        self.records = read_manifest(manifest, allowed_roles)
        if not self.records:
            raise ValueError(f"manifest has no records for roles {sorted(allowed_roles)}")
        self.root = Path(root).expanduser().resolve(strict=True)
        self.canvas_hw = canvas_hw

    def __len__(self) -> int:
        return len(self.records)

    def _resolve(self, record: SampleRecord, relative: str) -> Path:
        path = (self.root / relative).resolve(strict=True)
        if not path.is_relative_to(self.root):
            raise ValueError(f"asset escapes RFTrans root: {relative}")
        return path

    def __getitem__(self, index: int) -> Sample:
        record = self.records[index]
        rgb = read_rgb_u8(self._resolve(record, record.rgb_path))
        depth, valid = read_rftrans_depth(self._resolve(record, record.depth_gt_path))
        mask = read_rftrans_mask(self._resolve(record, record.mask_path))
        bundle = letterbox_bundle(rgb, depth, mask, valid, self.canvas_hw)
        patches = build_patch_targets(
            bundle.depth_m, bundle.mask, bundle.valid_depth, bundle.content_mask
        )
        return Sample(
            rgb=bundle.rgb,
            depth_m=bundle.depth_m,
            mask=bundle.mask,
            valid_depth=bundle.valid_depth,
            mask_known=bundle.mask_known,
            content_mask=bundle.content_mask,
            patch_targets=patches,
            metadata={
                "sample_id": record.sample_id,
                "source": record.source,
                "role": record.assigned_role,
                "frame_id": record.frame_id,
                "cad_ids": list(record.cad_ids),
                "geometry": bundle.geometry,
            },
        )
