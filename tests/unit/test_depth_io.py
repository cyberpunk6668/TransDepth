import numpy as np
import pytest
from PIL import Image

from transdepth.data.depth_io import (
    RFTRANS_DEPTH_SCALE_M,
    DataEncodingError,
    decode_rftrans_depth,
    decode_rftrans_mask,
    read_depth_png_raw,
)


def test_rftrans_u16_depth_scale_and_zero_validity() -> None:
    raw = np.array([[0, 1, 32768, 65535]], dtype=np.uint16)
    depth, valid = decode_rftrans_depth(raw)
    np.testing.assert_allclose(depth, raw.astype(np.float32) * RFTRANS_DEPTH_SCALE_M)
    assert valid.tolist() == [[False, True, True, True]]
    assert np.isclose(depth[0, 2], 1.5)
    assert depth[0, 3] < 3.0


def test_rftrans_mask_uses_exact_color_semantics() -> None:
    colors = np.array([[[0, 0, 0], [0, 255, 0], [255, 0, 0], [0, 254, 0]]], dtype=np.uint8)
    labels = decode_rftrans_mask(colors)
    assert labels.tolist() == [[0, 1, 0, 255]]


def test_depth_reader_rejects_8bit_png(tmp_path) -> None:
    path = tmp_path / "depth.png"
    Image.fromarray(np.array([[0, 255]], dtype=np.uint8)).save(path)
    with pytest.raises(DataEncodingError, match="16-bit grayscale"):
        read_depth_png_raw(path)


def test_depth_reader_accepts_real_16bit_header(tmp_path) -> None:
    path = tmp_path / "depth.png"
    expected = np.array([[0, 32768, 65535]], dtype=np.uint16)
    Image.fromarray(expected).save(path)
    result = read_depth_png_raw(path)
    np.testing.assert_array_equal(result, expected)
