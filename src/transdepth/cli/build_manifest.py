"""Build the RFTrans-only role manifest from verified frame-ID pairs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from transdepth.config import assert_runtime_assets, config_sha256, load_config
from transdepth.data.adapters.rftrans62 import scan_rftrans
from transdepth.data.manifest import write_manifest
from transdepth.utils.io import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/t0.yaml")
    parser.add_argument("--paths", default="configs/paths.local.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--hash-files", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.paths)
    assert_runtime_assets(config, require_weights=False)
    output = Path(args.output or Path(config["storage"]["manifests"]) / "rftrans_only_v1.jsonl")
    records = scan_rftrans(
        config["roots"]["rftrans"],
        split_seed=int(config["data"]["split_seed"]),
        hash_files=args.hash_files,
    )
    summary = write_manifest(records, output)
    summary.update(
        {
            "schema_version": "manifest_build_v1",
            "config_sha256": config_sha256(config),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "git_dirty": bool(
                subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
            ),
            "hash_files": args.hash_files,
            "excluded_training_root": str(Path(config["roots"]["rftrans"]) / "cleargrasp"),
        }
    )
    atomic_write_json(summary, output.with_suffix(".summary.json"))
    print(f"wrote {summary['records']} records to {output}")
    print(f"manifest sha256: {summary['sha256']}")


if __name__ == "__main__":
    main()
