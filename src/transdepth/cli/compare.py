"""Compare paired H1/H2 offline evaluation records and issue a scoped evidence decision."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transdepth.evaluation.compare import paired_comparison
from transdepth.utils.io import atomic_write_json


def _read(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h1", required=True, help="H1 per_image.jsonl")
    parser.add_argument("--h2", required=True, help="H2 per_image.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = paired_comparison(
        _read(args.h1), _read(args.h2), resamples=args.resamples, seed=args.seed
    )
    atomic_write_json(result, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
