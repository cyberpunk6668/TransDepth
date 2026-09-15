"""Strict, reproducible YAML composition for TransDepth runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """Raised when a configuration cannot describe a valid run."""


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ConfigError(f"environment variable {name!r} is not set")
        return os.environ[name]

    return _ENV_PATTERN.sub(replace, value)


def _load_recursive(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ConfigError(f"cyclic config include: {chain}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"top level of {path} must be a mapping")
    includes = raw.pop("includes", [])
    if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
        raise ConfigError(f"includes in {path} must be a list of paths")
    merged: dict[str, Any] = {}
    for include in includes:
        merged = _merge(merged, _load_recursive(path.parent / include, (*stack, path)))
    return _merge(merged, raw)


def load_config(config_path: str | Path, paths_path: str | Path | None = None) -> dict[str, Any]:
    """Load includes, apply the local paths override last, expand env, and validate."""
    config = _load_recursive(Path(config_path), ())
    if paths_path is not None:
        config = _merge(config, _load_recursive(Path(paths_path), ()))
    config = _expand_env(config)
    validate_config(config)
    return config


def require(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            raise ConfigError(f"missing required configuration key: {dotted_key}")
        value = value[key]
    return value


def validate_config(config: dict[str, Any]) -> None:
    if require(config, "schema_version") != "transdepth_minimal_v1":
        raise ConfigError("unsupported schema_version")
    if list(require(config, "data.canvas_hw")) != [384, 512]:
        raise ConfigError("the locked canvas must be 384x512")
    runtime = require(config, "runtime")
    effective = (
        int(runtime["world_size"])
        * int(runtime["microbatch_per_gpu"])
        * int(runtime["gradient_accumulation"])
    )
    if effective != int(runtime["global_batch_size"]):
        raise ConfigError(f"global batch mismatch: computed {effective}")
    allowed = {0, 3, 5}
    if not set(runtime["allowed_physical_gpus"]).issubset(allowed):
        raise ConfigError("only physical GPUs 0, 3, and 5 are permitted")
    if require(config, "data.source") != "rftrans_62cad":
        raise ConfigError("this minimal protocol trains only on RFTrans-62CAD")
    backbone = require(config, "backbone")
    expected = {
        "hub_entry": "dinov3_vith16plus",
        "repo_commit": "6876159a11b4df116f30f667f8c9888617df0751",
        "checkpoint_format": "meta_pth_state_dict",
        "feature_blocks_0based": [7, 15, 23, 31],
        "feature_width": 1280,
        "depth": 32,
        "num_heads": 20,
        "head_dim": 64,
        "prefix_tokens": 5,
        "patch_size": 16,
        "rope_dtype": "fp32",
    }
    for key, value in expected.items():
        if backbone.get(key) != value:
            raise ConfigError(f"backbone.{key} must be {value!r}")
    kind = config.get("experiment", {}).get("kind")
    if kind not in {None, "t0", "h1", "h2", "oracle", "predict"}:
        raise ConfigError(f"unsupported experiment kind: {kind!r}")
    if kind == "h1" and float(require(config, "loss.lambda_rel")) != 0.0:
        raise ConfigError("H1 must use lambda_rel=0")
    if kind == "h2" and float(require(config, "loss.lambda_rel")) <= 0.0:
        raise ConfigError("H2 must use a positive lambda_rel")


def assert_runtime_assets(
    config: dict[str, Any],
    *,
    require_weights: bool = True,
    require_training_data: bool = True,
) -> None:
    keys = ["storage.runs", "backbone.repo_dir"]
    if require_training_data:
        keys.extend(("roots.rftrans", "storage.manifests"))
    for key in keys:
        path = Path(require(config, key)).expanduser()
        if key.startswith("storage."):
            path.mkdir(parents=True, exist_ok=True)
        elif not path.exists():
            raise ConfigError(f"configured path does not exist: {key}={path}")
    if require_weights:
        checkpoint = Path(require(config, "backbone.checkpoint"))
        digest = require(config, "backbone.checkpoint_sha256")
        if not checkpoint.is_file():
            raise ConfigError(f"approved DINOv3 checkpoint is missing: {checkpoint}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest.lower()):
            raise ConfigError("backbone.checkpoint_sha256 must be the verified 64-char digest")


def config_sha256(config: dict[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def dump_resolved(config: dict[str, Any], path: str | Path) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_sha256(config)
