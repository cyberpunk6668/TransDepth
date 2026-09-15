import numpy as np
import torch

from transdepth.data.geometry import letterbox_bundle
from transdepth.data.patches import build_patch_targets


def test_square_rftrans_frame_is_not_stretched() -> None:
    rgb = np.full((512, 512, 3), 127, dtype=np.uint8)
    depth = np.ones((512, 512), dtype=np.float32)
    labels = np.zeros((512, 512), dtype=np.uint8)
    valid = np.ones((512, 512), dtype=bool)
    bundle = letterbox_bundle(rgb, depth, labels, valid)
    assert bundle.geometry.resized_hw == (384, 384)
    assert bundle.geometry.padding_tblr == (0, 0, 64, 64)
    assert not bundle.content_mask[:, :, :64].any()
    assert bundle.content_mask[:, :, 64:448].all()
    assert (bundle.mask[:, :, :64] == 255).all()
    assert not bundle.valid_depth[:, :, :64].any()
    assert torch.equal(bundle.rgb[:, :, :64], torch.zeros_like(bundle.rgb[:, :, :64]))


def _targets(valid_count: int, *, one_padding_pixel: bool = False):
    depth = torch.ones(16, 16)
    labels = torch.ones(16, 16, dtype=torch.uint8)
    valid = torch.zeros(16, 16, dtype=torch.bool)
    valid.flatten()[:valid_count] = True
    content = torch.ones(16, 16, dtype=torch.bool)
    if one_padding_pixel:
        content[0, 0] = False
    return build_patch_targets(depth, labels, valid, content)


def test_patch_validity_threshold_is_231_of_256() -> None:
    assert _targets(230).state.item() == -1
    assert _targets(231).state.item() == 1


def test_one_padding_pixel_invalidates_entire_patch() -> None:
    assert _targets(256, one_padding_pixel=True).state.item() == -1


def test_high_log_depth_iqr_is_unreliable() -> None:
    depth = torch.cat((torch.ones(128), torch.full((128,), 2.0))).reshape(16, 16)
    labels = torch.zeros(16, 16, dtype=torch.uint8)
    valid = torch.ones(16, 16, dtype=torch.bool)
    content = torch.ones_like(valid)
    result = build_patch_targets(depth, labels, valid, content)
    assert result.log_depth_iqr.item() > 0.1
    assert result.state.item() == -1
