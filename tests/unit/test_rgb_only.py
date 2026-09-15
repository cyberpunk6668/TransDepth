from pathlib import Path

import pytest

from transdepth.data.rgb_only import discover_cleargrasp_rgb


def test_cleargrasp_duplicate_rgb_variants_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "frame-rgb.jpg").touch()
    (tmp_path / "frame-transparent-rgb-img.jpg").touch()
    with pytest.raises(ValueError, match="multiple ClearGrasp RGB variants"):
        discover_cleargrasp_rgb(tmp_path)
