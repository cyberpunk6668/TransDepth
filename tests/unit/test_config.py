from copy import deepcopy
from pathlib import Path

import pytest

from transdepth.config import ConfigError, load_config, validate_config

ROOT = Path(__file__).parents[2]
PATHS = ROOT / "configs/paths.example.yaml"


def test_t0_config_composes_path_override() -> None:
    config = load_config(ROOT / "configs/experiments/t0.yaml", PATHS)
    assert config["roots"]["rftrans"] == "/path/to/RFTrans_datas"
    assert config["runtime"]["global_batch_size"] == 12
    assert config["protocol"]["training_sources"] == ["rftrans_62cad"]


def test_effective_batch_mismatch_is_rejected() -> None:
    config = load_config(ROOT / "configs/experiments/t0.yaml", PATHS)
    broken = deepcopy(config)
    broken["runtime"]["global_batch_size"] = 16
    with pytest.raises(ConfigError, match="global batch mismatch"):
        validate_config(broken)


def test_h1_cannot_enable_relation_loss() -> None:
    config = load_config(ROOT / "configs/experiments/h1.yaml", PATHS)
    config["loss"]["lambda_rel"] = 0.1
    with pytest.raises(ConfigError, match="H1"):
        validate_config(config)
