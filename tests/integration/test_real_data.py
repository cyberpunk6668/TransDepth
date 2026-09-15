from collections import Counter
from pathlib import Path

import pytest

from transdepth.data.dataset import RFTransDataset
from transdepth.data.depth_io import read_exr_channel
from transdepth.data.manifest import read_manifest
from transdepth.data.rgb_only import RgbOnlyDataset, discover_cleargrasp_rgb

RFTRANS_ROOT = Path("/ssd/polyu/RFTrans_datas")
MANIFEST = Path("/ssd/polyu/TransDepth/data/derived/rftrans/manifests/rftrans_only_v1.jsonl")
CLEARGRASP_ROOT = Path("/ssd/polyu/TransDepth/data/raw/cleargrasp/test-val")

pytestmark = pytest.mark.data


@pytest.mark.skipif(not MANIFEST.is_file(), reason="local hashed RFTrans manifest unavailable")
def test_frozen_manifest_roles_and_rules() -> None:
    records = read_manifest(MANIFEST)
    assert Counter(item.assigned_role for item in records) == {
        "R_train": 4000,
        "R_dev": 500,
        "R_select": 250,
        "R_confirm": 250,
        "R_hold": 1000,
    }
    assert {item.mask_rule_id for item in records} == {"rftrans_exact_black_red_green_v1"}
    assert {item.depth_encoding for item in records} == {
        "rftrans_ideal_png_u16_3_over_65536"
    }
    assert len({item.leakage_group_id for item in records}) == 6000
    assert {item.group_confidence for item in records} == {
        "full_recorder_scene_fingerprint_unique_in_release"
    }


@pytest.mark.skipif(not MANIFEST.is_file(), reason="local RFTrans assets unavailable")
def test_real_rftrans_training_sample_contract() -> None:
    dataset = RFTransDataset(MANIFEST, RFTRANS_ROOT, {"R_train"})
    sample = dataset[0]
    assert sample.rgb.shape == (3, 384, 512)
    assert sample.depth_m.shape == (1, 384, 512)
    assert sample.mask.shape == (1, 384, 512)
    assert sample.content_mask.sum() == 384 * 384
    assert sample.patch_targets.state.shape == (24, 32)
    assert sample.patch_targets.reliable.any()


@pytest.mark.skipif(not RFTRANS_ROOT.is_dir(), reason="local RFTrans assets unavailable")
def test_real_rftrans_exr_is_named_y_half() -> None:
    depth, metadata = read_exr_channel(RFTRANS_ROOT / "train/depth/depth_0.exr", "Y")
    assert depth.shape == (512, 512)
    assert metadata["channels"] == ["Y"]
    assert metadata["source_dtype"] == "float16"


@pytest.mark.skipif(not CLEARGRASP_ROOT.is_dir(), reason="ClearGrasp inference archive unavailable")
def test_cleargrasp_is_rgb_only_and_complete() -> None:
    paths = discover_cleargrasp_rgb(CLEARGRASP_ROOT)
    assert len(paths) == 1226
    dataset = RgbOnlyDataset(CLEARGRASP_ROOT, paths[:1])
    sample = dataset[0]
    assert sample.rgb.shape == (3, 384, 512)
    assert not hasattr(sample, "depth_m")
    assert not hasattr(sample, "mask")
