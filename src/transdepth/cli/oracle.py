"""Run pre-registered RFTrans Select or one-shot Confirm FAR P@V oracle phases."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import torch
import yaml

from transdepth.config import load_config
from transdepth.data.dataset import RFTransDataset
from transdepth.engine.checkpoint import load_checkpoint
from transdepth.models.factory import build_predictor
from transdepth.oracle.runner import (
    baseline_predictions,
    oracle_predictions,
    stable_sample_indices,
)
from transdepth.oracle.statistics import confirm_candidate, summarize_candidate
from transdepth.utils.io import atomic_write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("search", "confirm"), required=True)
    parser.add_argument("--config", default="configs/experiments/t0.yaml")
    parser.add_argument("--paths", default="configs/paths.local.yaml")
    parser.add_argument("--oracle-config", default="configs/oracle/rftrans_h_a_minimal.yaml")
    parser.add_argument("--t0-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", type=int, default=0)
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_oracle_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if value.get("schema_version") != "rftrans_oracle_minimal_v1":
        raise ValueError("unsupported oracle configuration")
    return value


def _samples(
    config: dict[str, Any], role: str, maximum: int, seed: int
) -> list:
    manifest = Path(config["storage"]["manifests"]) / "rftrans_only_v1.jsonl"
    dataset = RFTransDataset(manifest, config["roots"]["rftrans"], {role})
    indices = stable_sample_indices(
        [record.sample_id for record in dataset.records], maximum, seed
    )
    return [dataset[index] for index in indices]


def _run_candidate(
    predictor,
    samples,
    baseline_rows,
    baseline_arrays,
    device,
    config,
    oracle_config,
    block: int,
    head: int,
    beta: float,
) -> tuple[dict[str, Any], float]:
    adapter = predictor.features.enable_lora(
        block_index=block,
        head_index=head,
        rank=int(config["far"]["rank"]),
        eta=float(config["far"]["eta"]),
        seed=int(config["runtime"]["seed"]),
    )
    try:
        identity_rows, identity_arrays = oracle_predictions(
            predictor,
            adapter,
            samples[:1],
            device,
            beta=0.0,
            sigma_z=float(config["far"]["sigma_z"]),
            cmax=float(config["far"]["cmax"]),
            mass_eps=float(config["far"]["mass_eps"]),
        )
        del identity_rows
        sample_id = samples[0].metadata["sample_id"]
        identity_error = float(
            abs(identity_arrays[sample_id] - baseline_arrays[sample_id]).max()
        )
        threshold = float(oracle_config["constraints"]["identity_depth_atol_m"])
        if identity_error > threshold:
            raise RuntimeError(
                f"P=A0 identity failed at block={block}, head={head}: {identity_error}"
            )
        candidate_rows, _ = oracle_predictions(
            predictor,
            adapter,
            samples,
            device,
            beta=beta,
            sigma_z=float(config["far"]["sigma_z"]),
            cmax=float(config["far"]["cmax"]),
            mass_eps=float(config["far"]["mass_eps"]),
        )
        summary = summarize_candidate(
            baseline_rows,
            candidate_rows,
            minimum_tolerance_mm=float(
                oracle_config["constraints"]["degradation_min_tolerance_mm"]
            ),
            relative_tolerance=float(
                oracle_config["constraints"]["degradation_relative_tolerance"]
            ),
        )
        summary.update({"block": block, "head": head, "beta": beta})
        return summary, identity_error
    finally:
        predictor.features.disable_lora()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.paths)
    oracle_config = _load_oracle_config(args.oracle_config)
    t0_path = Path(args.t0_checkpoint).resolve(strict=True)
    if load_checkpoint(t0_path)["experiment_kind"] != "t0":
        raise ValueError("oracle requires a T0 checkpoint")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    predictor = build_predictor(
        config, use_lora=False, trained_checkpoint=t0_path
    ).eval().to(device)
    seed = int(config["runtime"]["seed"])
    if args.phase == "search":
        spec = oracle_config["search"]
        samples = _samples(config, spec["role"], int(spec["samples"]), seed)
        baseline_rows, baseline_arrays = baseline_predictions(predictor, samples, device)
        generator = random.Random(int(oracle_config["candidates"]["head_seed"]))
        heads = {
            block: sorted(
                generator.sample(
                    range(int(config["backbone"]["num_heads"])),
                    int(oracle_config["candidates"]["heads_per_block"]),
                )
            )
            for block in oracle_config["candidates"]["blocks_0based"]
        }
        results = []
        identity_errors = []
        for block, block_heads in heads.items():
            for head in block_heads:
                for beta in oracle_config["candidates"]["betas"]:
                    result, identity_error = _run_candidate(
                        predictor,
                        samples,
                        baseline_rows,
                        baseline_arrays,
                        device,
                        config,
                        oracle_config,
                        int(block),
                        int(head),
                        float(beta),
                    )
                    results.append(result)
                    identity_errors.append(identity_error)
        eligible = [item for item in results if item["eligible"]]
        choice = (
            max(eligible, key=lambda item: item["transparent_improvement_mean_mm"])
            if eligible
            else None
        )
        report = {
            "schema_version": "oracle_choice_v1",
            "status": "selected" if choice is not None else "no_eligible_candidate",
            "t0_checkpoint": str(t0_path),
            "t0_sha256": _file_sha256(t0_path),
            "sample_ids": [sample.metadata["sample_id"] for sample in samples],
            "heads": heads,
            "results": results,
            "identity_max_abs_m": max(identity_errors),
            "choice": choice,
        }
        atomic_write_json(report, output / "oracle_choice.json")
        print(json.dumps({"status": report["status"], "choice": choice}, indent=2))
        if choice is None:
            raise SystemExit(3)
    else:
        choice_path = output / "oracle_choice.json"
        choice_report = json.loads(choice_path.read_text(encoding="utf-8"))
        if choice_report.get("status") != "selected":
            raise ValueError("confirm requires a successful search choice")
        if choice_report["t0_sha256"] != _file_sha256(t0_path):
            raise ValueError("T0 checkpoint differs from the selection phase")
        choice = choice_report["choice"]
        spec = oracle_config["confirm"]
        samples = _samples(config, spec["role"], int(spec["samples"]), seed)
        baseline_rows, baseline_arrays = baseline_predictions(predictor, samples, device)
        summary, identity_error = _run_candidate(
            predictor,
            samples,
            baseline_rows,
            baseline_arrays,
            device,
            config,
            oracle_config,
            int(choice["block"]),
            int(choice["head"]),
            float(choice["beta"]),
        )
        decision = confirm_candidate(
            summary,
            resamples=int(spec["bootstrap_resamples"]),
            seed=int(spec["bootstrap_seed"]),
            minimum_samples=int(spec["minimum_samples"]),
        )
        atomic_write_json(decision, output / "confirm_decision.json")
        lock = {
            "schema_version": "oracle_lock_v1",
            "status": "passed" if decision["confirm_passed"] else "failed",
            "selected_block": int(choice["block"]),
            "selected_head": int(choice["head"]),
            "beta": float(choice["beta"]),
            "t0_sha256": choice_report["t0_sha256"],
            "identity_max_abs_m": identity_error,
            "confirm": decision,
            "scope": "minimal RFTrans-only oracle; not the full paper candidate budget",
        }
        atomic_write_json(lock, output / "oracle_lock.json")
        print(json.dumps(lock, indent=2))
        if lock["status"] != "passed":
            raise SystemExit(4)


if __name__ == "__main__":
    main()
