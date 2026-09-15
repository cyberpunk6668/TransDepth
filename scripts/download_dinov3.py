#!/usr/bin/env python3
"""Download an approved Meta-native DINOv3 H+/16 URL without exposing it in shell history."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import subprocess
from pathlib import Path

import yaml

FILENAME = "dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="/ssd/polyu/TransDepth/models/dinov3/meta_native",
    )
    parser.add_argument(
        "--paths",
        default="/home/hengxianli/TransDepth/configs/paths.local.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / FILENAME
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    url = getpass.getpass("Approved complete DINOv3 H+/16 URL (hidden): ").strip()
    if not url.startswith("https://"):
        raise ValueError("only an approved HTTPS URL is accepted")
    process = subprocess.run(
        [
            "wget",
            "--continue",
            "--no-verbose",
            "--input-file=-",
            f"--output-document={partial}",
        ],
        input=url + "\n",
        text=True,
        check=False,
    )
    del url
    if process.returncode != 0 or not partial.is_file() or partial.stat().st_size < 1_000_000_000:
        raise RuntimeError(
            "download failed or result is too small to be the 840M-parameter backbone"
        )
    partial.replace(destination)
    digest = hashlib.sha256()
    with destination.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    checksum = digest.hexdigest()
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{checksum}  {destination.name}\n", encoding="utf-8"
    )
    paths_file = Path(args.paths).expanduser().resolve(strict=True)
    paths_config = yaml.safe_load(paths_file.read_text(encoding="utf-8"))
    paths_config["backbone"]["checkpoint"] = str(destination)
    paths_config["backbone"]["checkpoint_sha256"] = checksum
    paths_file.write_text(yaml.safe_dump(paths_config, sort_keys=False), encoding="utf-8")
    print(f"downloaded {destination} ({destination.stat().st_size} bytes)")
    print(f"SHA256 {checksum}")
    print(f"updated checkpoint identity in {paths_file}")


if __name__ == "__main__":
    main()
