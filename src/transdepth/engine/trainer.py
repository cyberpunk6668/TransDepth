"""Minimal deterministic T0/H1/H2 trainer for the RFTrans-only protocol."""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from transdepth.config import config_sha256, dump_resolved
from transdepth.data.dataset import RFTransDataset
from transdepth.data.schema import TrainBatch, collate_samples
from transdepth.engine.checkpoint import (
    atomic_torch_save,
    load_checkpoint,
    load_trainable_state,
    trainable_state_dict,
)
from transdepth.engine.schedule import build_global_schedule, rank_slot, schedule_sha256
from transdepth.far.relation_loss import relation_kl_loss
from transdepth.losses.depth import balanced_log_huber_loss
from transdepth.models.factory import build_predictor
from transdepth.utils.io import atomic_write_json


def _distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _set_seed(seed: int, rank: int) -> None:
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    decoder_decay, decoder_no_decay, lora = [], [], []
    parameter_rows = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_lora = any(token in name for token in ("q_down", "q_up", "k_down", "k_up"))
        if is_lora:
            lora.append(parameter)
            group = "lora"
        elif parameter.ndim == 1 or name.endswith(".bias"):
            decoder_no_decay.append(parameter)
            group = "decoder_no_decay"
        else:
            decoder_decay.append(parameter)
            group = "decoder_decay"
        parameter_rows.append({"name": name, "shape": list(parameter.shape), "group": group})
    if not decoder_decay and not decoder_no_decay:
        raise RuntimeError("no trainable decoder parameters found")
    groups: list[dict[str, Any]] = [
        {
            "params": decoder_decay,
            "lr": float(config["training"]["decoder_lr"]),
            "weight_decay": float(config["training"]["weight_decay"]),
        },
        {
            "params": decoder_no_decay,
            "lr": float(config["training"]["decoder_lr"]),
            "weight_decay": 0.0,
        },
    ]
    if lora:
        groups.append(
            {
                "params": lora,
                "lr": float(config["training"]["lora_lr"]),
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)
    optimizer.parameter_rows = parameter_rows  # type: ignore[attr-defined]
    return optimizer


def _scheduler(
    optimizer: torch.optim.Optimizer, updates: int, warmup: int
) -> torch.optim.lr_scheduler.LambdaLR:
    if not 0 <= warmup < updates:
        raise ValueError("warmup must be nonnegative and shorter than total updates")

    def multiplier(step: int) -> float:
        if warmup and step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, updates - warmup - 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
        stream.flush()


def _move_batch(batch: TrainBatch, device: torch.device) -> TrainBatch:
    return batch.to(device, non_blocking=False)


@torch.no_grad()
def _evaluate_dev(
    model: nn.Module,
    dataset: RFTransDataset,
    device: torch.device,
    *,
    rank: int,
    world_size: int,
    maximum: int,
) -> dict[str, float | int | None]:
    model.eval()
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    for index in range(rank, min(len(dataset), maximum), world_size):
        batch = _move_batch(collate_samples([dataset[index]]), device)
        prediction = model(batch.rgb, return_aux=False).depth_m
        valid = batch.valid_depth & batch.mask_known
        for offset, label in ((0, 1), (2, 0)):
            selected = valid & (batch.mask == label)
            if bool(selected.any()):
                mae = (prediction[selected] - batch.depth_m[selected]).abs().double().mean()
                totals[offset] += mae
                totals[offset + 1] += 1
    if world_size > 1:
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    result = {
        "transparent_mae_m": float(totals[0] / totals[1]) if totals[1] else None,
        "transparent_images": int(totals[1]),
        "background_mae_m": float(totals[2] / totals[3]) if totals[3] else None,
        "background_images": int(totals[3]),
    }
    model.train()
    return result


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict[str, Any],
    *,
    experiment_kind: str,
    next_update: int,
    plan_hash: str,
    best_transparent_mae_m: float | None,
) -> dict[str, Any]:
    return {
        "schema_version": "td_checkpoint_v1",
        "experiment_kind": experiment_kind,
        "next_update": next_update,
        "trainable_state": trainable_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config_sha256": config_sha256(config),
        "schedule_sha256": plan_hash,
        "backbone_sha256": config["backbone"]["checkpoint_sha256"],
        "model_spec": {
            "selected_block": config["far"].get("selected_block"),
            "selected_head": config["far"].get("selected_head"),
            "beta": config["far"].get("beta"),
            "rank": config["far"]["rank"],
            "eta": config["far"]["eta"],
        },
        "best_transparent_mae_m": best_transparent_mae_m,
    }


def train(
    config: dict[str, Any],
    run_dir: str | Path,
    *,
    fork_from: str | Path | None = None,
    resume: str | Path | None = None,
) -> None:
    rank, local_rank, world_size = _distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("training requires CUDA")
    if world_size != int(config["runtime"]["world_size"]):
        raise RuntimeError(
            f"launch world_size={world_size}, config requires {config['runtime']['world_size']}"
        )
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    seed = int(config["runtime"]["seed"])
    _set_seed(seed, rank)
    kind = config["experiment"]["kind"]
    use_lora = kind in {"h1", "h2"}
    if use_lora and fork_from is None and resume is None:
        raise ValueError("H1/H2 require --fork-from T0 or --resume")
    predictor = build_predictor(
        config,
        use_lora=use_lora,
        fork_checkpoint=fork_from if resume is None else None,
    ).to(device)
    if kind == "t0" and resume is None:
        with torch.no_grad():
            predictor.tail.conv2.bias.fill_(math.log(math.expm1(0.5)))
    optimizer = _optimizer(predictor, config)
    updates = int(config["training"]["updates"])
    scheduler = _scheduler(optimizer, updates, int(config["training"]["warmup_updates"]))
    start_update = 0
    best_mae: float | None = None
    resume_checkpoint = None
    if resume is not None:
        resume_checkpoint = load_checkpoint(resume)
        if resume_checkpoint["experiment_kind"] != kind:
            raise ValueError("resume checkpoint experiment kind mismatch")
        load_trainable_state(predictor, resume_checkpoint["trainable_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state"])
        start_update = int(resume_checkpoint["next_update"])
        best_mae = resume_checkpoint.get("best_transparent_mae_m")

    manifest = Path(config["storage"]["manifests"]) / "rftrans_only_v1.jsonl"
    train_data = RFTransDataset(manifest, config["roots"]["rftrans"], {"R_train"})
    dev_data = RFTransDataset(manifest, config["roots"]["rftrans"], {"R_dev"})
    schedule = build_global_schedule(
        len(train_data), updates, int(config["runtime"]["global_batch_size"]), seed
    )
    plan_hash = schedule_sha256(schedule)
    if resume_checkpoint is not None and resume_checkpoint["schedule_sha256"] != plan_hash:
        raise ValueError("resume sample schedule mismatch")

    run_path = Path(run_dir).expanduser().resolve()
    if rank == 0:
        run_path.mkdir(parents=True, exist_ok=True)
        if any(run_path.iterdir()) and resume is None:
            raise FileExistsError(f"run directory is not empty: {run_path}")
        dump_resolved(config, run_path / "resolved_config.yaml")
        plan = [
            [train_data.records[int(index)].sample_id for index in row]
            for row in schedule.tolist()
        ]
        atomic_write_json(
            {"schema_version": "global_plan_v1", "sha256": plan_hash, "updates": plan},
            run_path / "sample_plan.json",
        )
        atomic_write_json(
            {
                "schema_version": "run_manifest_v1",
                "experiment_kind": kind,
                "git_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "git_dirty": bool(
                    subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
                ),
                "world_size": world_size,
                "physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "global_batch_size": config["runtime"]["global_batch_size"],
                "schedule_sha256": plan_hash,
                "fork_from": str(fork_from) if fork_from else None,
                "resume": str(resume) if resume else None,
                "trainable_parameters": optimizer.parameter_rows,  # type: ignore[attr-defined]
            },
            run_path / "run_manifest.json",
        )
    _barrier(world_size)
    distributed: nn.Module = (
        DistributedDataParallel(
            predictor,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        if world_size > 1
        else predictor
    )
    accumulation = int(config["runtime"]["gradient_accumulation"])
    amp_dtype = torch.bfloat16 if config["runtime"]["amp_dtype"] == "bf16" else torch.float16
    training_started = time.perf_counter()
    for update in range(start_update, updates):
        update_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        depth_total = 0.0
        relation_total = 0.0
        for microstep in range(accumulation):
            dataset_index = rank_slot(schedule, update, microstep, rank, world_size)
            batch = _move_batch(collate_samples([train_data[dataset_index]]), device)
            sync = (
                distributed.no_sync()  # type: ignore[union-attr]
                if world_size > 1 and microstep < accumulation - 1
                else nullcontext()
            )
            with sync:
                with torch.autocast("cuda", dtype=amp_dtype):
                    output = distributed(batch.rgb, return_aux=kind == "h2")
                with torch.autocast("cuda", enabled=False):
                    depth_result = balanced_log_huber_loss(
                        output.depth_m,
                        batch.depth_m,
                        batch.valid_depth & batch.mask_known,
                        batch.mask,
                        delta=float(config["loss"]["depth_delta"]),
                    )
                    relation = output.depth_m.sum() * 0.0
                    if kind == "h2":
                        if output.trace is None:
                            raise RuntimeError("H2 model did not return an attention trace")
                        relation = relation_kl_loss(
                            output.trace,
                            batch.patch_targets,
                            beta=float(config["far"]["beta"]),
                            sigma_z=float(config["far"]["sigma_z"]),
                            cmax=float(config["far"]["cmax"]),
                            mass_eps=float(config["far"]["mass_eps"]),
                            max_queries_per_class=int(config["far"]["max_queries_per_class"]),
                            query_seed=seed + update * 10_007 + microstep,
                        ).loss
                    loss = depth_result.loss + float(config["loss"]["lambda_rel"]) * relation
                    (loss / accumulation).backward()
                depth_total += float(depth_result.loss.detach())
                relation_total += float(relation.detach())
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in predictor.parameters() if parameter.requires_grad],
            float(config["training"]["grad_clip_norm"]),
        )
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite trainable gradient norm")
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize(device)
        if rank == 0:
            _append_jsonl(
                run_path / "logs" / "train.jsonl",
                {
                    "update": update + 1,
                    "depth_loss": depth_total / accumulation,
                    "relation_loss": relation_total / accumulation,
                    "gradient_norm": float(gradient_norm),
                    "learning_rates": [group["lr"] for group in optimizer.param_groups],
                    "seconds": time.perf_counter() - update_started,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                },
            )

        should_evaluate = (update + 1) % int(config["training"]["eval_every"]) == 0
        should_save = (update + 1) % int(config["training"]["save_every"]) == 0
        if should_evaluate:
            metrics = _evaluate_dev(
                distributed,
                dev_data,
                device,
                rank=rank,
                world_size=world_size,
                maximum=int(config["training"]["eval_max_samples"]),
            )
            if rank == 0:
                metrics["update"] = update + 1
                _append_jsonl(run_path / "metrics" / "dev.jsonl", metrics)
                current = metrics["transparent_mae_m"]
                if isinstance(current, float) and (best_mae is None or current < best_mae):
                    best_mae = current
                    payload = _checkpoint_payload(
                        predictor,
                        optimizer,
                        scheduler,
                        config,
                        experiment_kind=kind,
                        next_update=update + 1,
                        plan_hash=plan_hash,
                        best_transparent_mae_m=best_mae,
                    )
                    atomic_torch_save(payload, run_path / "checkpoints" / "best.pt")
        if should_save or update + 1 == updates:
            _barrier(world_size)
            if rank == 0:
                payload = _checkpoint_payload(
                    predictor,
                    optimizer,
                    scheduler,
                    config,
                    experiment_kind=kind,
                    next_update=update + 1,
                    plan_hash=plan_hash,
                    best_transparent_mae_m=best_mae,
                )
                atomic_torch_save(payload, run_path / "checkpoints" / "last.pt")
            _barrier(world_size)
    if rank == 0:
        atomic_write_json(
            {
                "status": "completed",
                "updates": updates,
                "wall_seconds": time.perf_counter() - training_started,
                "best_transparent_mae_m": best_mae,
            },
            run_path / "completed.json",
        )
    _barrier(world_size)
    if dist.is_initialized():
        dist.destroy_process_group()
