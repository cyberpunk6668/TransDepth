"""Train paired RFTrans-only T0, H1, or H2 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from transdepth.config import load_config, validate_config
from transdepth.engine.trainer import train
from transdepth.models.factory import apply_oracle_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--paths", default="configs/paths.local.yaml")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--fork-from", default=None)
    parser.add_argument("--oracle-lock", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--updates", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fork_from and args.resume:
        raise ValueError("--fork-from and --resume are mutually exclusive")
    config = load_config(args.config, args.paths)
    kind = config["experiment"]["kind"]
    if args.updates is not None:
        config["training"]["updates"] = args.updates
        if config["training"]["warmup_updates"] >= args.updates:
            config["training"]["warmup_updates"] = max(0, min(20, args.updates // 10))
    if kind in {"h1", "h2"}:
        if args.oracle_lock is None:
            raise ValueError("paired H1/H2 runs require --oracle-lock")
        apply_oracle_lock(config, args.oracle_lock)
    validate_config(config)
    run_dir = Path(
        args.run_dir or Path(config["storage"]["runs"]) / config["experiment"]["name"]
    )
    train(config, run_dir, fork_from=args.fork_from, resume=args.resume)


if __name__ == "__main__":
    main()
