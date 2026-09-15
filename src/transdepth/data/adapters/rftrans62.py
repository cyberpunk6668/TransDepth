"""Adapter for the verified robotflow RFTrans 62CAD release layout."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path

from transdepth.data.depth_io import sha256_file
from transdepth.data.schema import SampleRecord

_PATTERNS = {
    "rgb": re.compile(r"rgb_(\d+)\.png$"),
    "depth": re.compile(r"depth_(\d+)\.png$"),
    "depth_reference": re.compile(r"depth_(\d+)\.exr$"),
    "mask": re.compile(r"mask_(\d+)\.png$"),
    "metadata": re.compile(r"record_(\d+)\.txt$"),
}


class PairingError(ValueError):
    """Raised if RFTrans modalities cannot be paired exactly by frame ID."""


def _ids(directory: Path, pattern: re.Pattern[str]) -> set[int]:
    result: set[int] = set()
    for path in directory.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            frame_id = int(match.group(1))
            if frame_id in result:
                raise PairingError(f"duplicate frame ID {frame_id} under {directory}")
            result.add(frame_id)
    return result


def verify_split_pairing(root: str | Path, split: str) -> list[int]:
    split_root = Path(root).resolve(strict=True) / split
    modality_ids = {
        "rgb": _ids(split_root / "RGB", _PATTERNS["rgb"]),
        "depth": _ids(split_root / "depth", _PATTERNS["depth"]),
        "depth_reference": _ids(split_root / "depth", _PATTERNS["depth_reference"]),
        "mask": _ids(split_root / "mask", _PATTERNS["mask"]),
        "metadata": _ids(split_root / "recorder", _PATTERNS["metadata"]),
    }
    union = set().union(*modality_ids.values())
    errors = {
        name: sorted(union - values)
        for name, values in modality_ids.items()
        if values != union
    }
    if errors:
        preview = {name: ids[:10] for name, ids in errors.items()}
        raise PairingError(f"unpaired {split} modalities (first IDs): {preview}")
    return sorted(union)


def parse_recorder_cad_ids(path: str | Path) -> tuple[str, ...]:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if len(lines) < 4:
        raise PairingError(f"recorder is truncated: {path}")
    try:
        count = int(lines[3])
    except ValueError as error:
        raise PairingError(f"invalid object count in {path}") from error
    expected = 4 + 6 * count
    if len(lines) < expected:
        raise PairingError(f"recorder {path} has {len(lines)} lines; expected at least {expected}")
    return tuple(lines[4 + 6 * index] for index in range(count))


def _stable_role_partition(records: list[SampleRecord], seed: int) -> list[SampleRecord]:
    expected = {"R_train": 4000, "R_dev": 500, "R_select": 250, "R_confirm": 250}
    if len(records) != sum(expected.values()):
        raise PairingError(f"expected 5000 official train frames, found {len(records)}")
    ranked = sorted(
        records,
        key=lambda item: hashlib.sha256(f"{seed}:{item.sample_id}".encode()).digest(),
    )
    assigned: dict[str, str] = {}
    cursor = 0
    for role, count in expected.items():
        for record in ranked[cursor : cursor + count]:
            assigned[record.sample_id] = role
        cursor += count
    return [replace(record, assigned_role=assigned[record.sample_id]) for record in records]


def scan_rftrans(
    root: str | Path,
    *,
    split_seed: int = 17,
    hash_files: bool = False,
) -> list[SampleRecord]:
    root_path = Path(root).expanduser().resolve(strict=True)
    records: list[SampleRecord] = []
    for split, expected_count in (("train", 5000), ("valid", 1000)):
        frame_ids = verify_split_pairing(root_path, split)
        if len(frame_ids) != expected_count:
            raise PairingError(f"expected {expected_count} {split} frames, found {len(frame_ids)}")
        for frame_id in frame_ids:
            prefix = Path(split)
            paths = {
                "rgb": prefix / "RGB" / f"rgb_{frame_id}.png",
                "depth": prefix / "depth" / f"depth_{frame_id}.png",
                "reference": prefix / "depth" / f"depth_{frame_id}.exr",
                "mask": prefix / "mask" / f"mask_{frame_id}.png",
                "metadata": prefix / "recorder" / f"record_{frame_id}.txt",
            }
            sample_id = f"rftrans62_{split}_{frame_id:06d}"
            hashes = (
                {
                    "rgb_sha256": sha256_file(root_path / paths["rgb"]),
                    "depth_sha256": sha256_file(root_path / paths["depth"]),
                    "mask_sha256": sha256_file(root_path / paths["mask"]),
                }
                if hash_files
                else {}
            )
            records.append(
                SampleRecord(
                    schema_version="far_data_v1",
                    sample_id=sample_id,
                    source="rftrans_62cad",
                    release_id="robotflow_rftrans_local_v1",
                    official_split=split,
                    assigned_role="unassigned" if split == "train" else "R_hold",
                    frame_id=frame_id,
                    rgb_path=paths["rgb"].as_posix(),
                    depth_gt_path=paths["depth"].as_posix(),
                    depth_reference_path=paths["reference"].as_posix(),
                    mask_path=paths["mask"].as_posix(),
                    metadata_path=paths["metadata"].as_posix(),
                    leakage_group_id=sample_id,
                    group_confidence="generated_frame_with_recorder",
                    depth_encoding="rftrans_ideal_png_u16_3_over_65536",
                    depth_coordinate="optical_z",
                    invalid_rule_id="raw_zero_invalid_v1",
                    mask_rule_id="rftrans_exact_black_red_green_v1",
                    preprocessing_version="letterbox_384x512_v1",
                    cad_ids=parse_recorder_cad_ids(root_path / paths["metadata"]),
                    **hashes,
                )
            )
    train_records = [item for item in records if item.official_split == "train"]
    train = _stable_role_partition(train_records, split_seed)
    hold = [item for item in records if item.official_split == "valid"]
    return [*train, *hold]
