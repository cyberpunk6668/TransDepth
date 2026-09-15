"""Lossless RFTrans RGB, depth, mask, and EXR reference readers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

RFTRANS_DEPTH_SCALE_M = 3.0 / 65536.0


class DataEncodingError(ValueError):
    """Raised when an asset does not match the locked RFTrans encoding."""


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rgb_u8(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise DataEncodingError(f"RGB image has invalid shape: {rgb.shape}")
    return rgb


def read_depth_png_raw(path: str | Path) -> np.ndarray:
    with Path(path).open("rb") as stream:
        header = stream.read(26)
    if (
        len(header) != 26
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
        or header[24] != 16
        or header[25] != 0
    ):
        raise DataEncodingError("RFTrans depth must be a 16-bit grayscale PNG")
    with Image.open(path) as image:
        raw = np.asarray(image).copy()
        mode = image.mode
    if raw.ndim != 2 or raw.dtype.kind not in "ui":
        raise DataEncodingError(
            f"depth PNG must be one integer channel, got {raw.shape}/{raw.dtype}"
        )
    minimum = int(raw.min())
    maximum = int(raw.max())
    if minimum < 0 or maximum > 65535:
        raise DataEncodingError(f"depth PNG values outside uint16 range: [{minimum}, {maximum}]")
    if mode not in {"I", "I;16", "I;16B", "I;16L"}:
        raise DataEncodingError(f"unexpected depth PNG mode: {mode}")
    return raw


def decode_rftrans_depth(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if raw.ndim != 2 or raw.dtype.kind not in "ui":
        raise DataEncodingError("RFTrans depth decoder requires a one-channel integer array")
    if int(raw.min()) < 0 or int(raw.max()) > 65535:
        raise DataEncodingError("RFTrans depth exceeds the verified 16-bit range")
    depth_m = np.ascontiguousarray(raw.astype(np.float32) * RFTRANS_DEPTH_SCALE_M)
    valid = np.ascontiguousarray(raw > 0)
    return depth_m, valid


def read_rftrans_depth(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    return decode_rftrans_depth(read_depth_png_raw(path))


def decode_rftrans_mask(mask_rgb: np.ndarray) -> np.ndarray:
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise DataEncodingError(f"mask must have three color channels, got {mask_rgb.shape}")
    labels = np.full(mask_rgb.shape[:2], 255, dtype=np.uint8)
    labels[np.all(mask_rgb == (0, 0, 0), axis=-1)] = 0
    labels[np.all(mask_rgb == (255, 0, 0), axis=-1)] = 0
    labels[np.all(mask_rgb == (0, 255, 0), axis=-1)] = 1
    return labels


def read_rftrans_mask(path: str | Path) -> np.ndarray:
    return decode_rftrans_mask(read_rgb_u8(path))


def read_exr_channel(path: str | Path, channel: str = "Y") -> tuple[np.ndarray, dict[str, Any]]:
    """Read a named EXR channel and preserve header evidence; never guess a channel."""
    import OpenEXR

    file = OpenEXR.File(str(path), separate_channels=True, header_only=False)
    try:
        channels = file.channels()
        if channel not in channels:
            raise DataEncodingError(f"EXR {path} has {sorted(channels)}, not channel {channel!r}")
        pixels = np.asarray(channels[channel].pixels, dtype=np.float32)
        header = file.header()
        metadata = {
            "channels": sorted(channels),
            "source_dtype": str(channels[channel].pixels.dtype),
            "shape": list(pixels.shape),
            "compression": str(header.get("compression", "unknown")),
        }
        return np.ascontiguousarray(pixels), metadata
    finally:
        close = getattr(file, "close", None)
        if close is not None:
            close()
