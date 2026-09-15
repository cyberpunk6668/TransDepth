import numpy as np

from transdepth.data.depth_io import (
    RFTRANS_DEPTH_SCALE_M,
    decode_rftrans_depth,
    decode_rftrans_mask,
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
