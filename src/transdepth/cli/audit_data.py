"""Audit RFTrans training-side encoding, geometry, and reliable patches."""

from __future__ import annotations

import argparse
from pathlib import Path

from transdepth.config import assert_runtime_assets, load_config
from transdepth.data.audit import (
    audit_record,
    save_qa_figure,
    select_audit_records,
    summarize_audit,
)
from transdepth.data.manifest import read_manifest
from transdepth.utils.io import atomic_write_json, atomic_write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/t0.yaml")
    parser.add_argument("--paths", default="configs/paths.local.yaml")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--per-role", type=int, default=30)
    parser.add_argument(
        "--roles", nargs="+", default=["R_train", "R_dev", "R_select", "R_confirm"]
    )
    parser.add_argument("--figures-per-role", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "R_hold" in args.roles:
        raise ValueError(
            "training-side QA refuses R_hold; use a separately authorized release audit"
        )
    config = load_config(args.config, args.paths)
    assert_runtime_assets(config, require_weights=False)
    manifest = Path(
        args.manifest or Path(config["storage"]["manifests"]) / "rftrans_only_v1.jsonl"
    )
    output = Path(args.output or Path(config["storage"]["qa"]) / "encoding_v1")
    records = read_manifest(manifest)
    selected = select_audit_records(
        records, set(args.roles), args.per_role, int(config["data"]["split_seed"])
    )
    root = Path(config["roots"]["rftrans"]).resolve(strict=True)
    rows = []
    figure_counts = {role: 0 for role in args.roles}
    for record in selected:
        row, bundle, rgb = audit_record(record, root)
        rows.append(row)
        if figure_counts[record.assigned_role] < args.figures_per_role:
            save_qa_figure(
                record,
                row,
                bundle,
                rgb,
                output / "figures" / record.assigned_role / f"{record.sample_id}.png",
            )
            figure_counts[record.assigned_role] += 1
    summary = summarize_audit(rows)
    summary["manifest"] = str(manifest)
    summary["samples_per_role_requested"] = args.per_role
    atomic_write_jsonl(rows, output / "per_sample.jsonl")
    atomic_write_json(summary, output / "summary.json")
    print(f"audit status: {summary['status']}")
    print(f"report: {output / 'summary.json'}")
    if summary["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
