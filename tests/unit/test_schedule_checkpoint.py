from pathlib import Path

import pytest
import torch
from torch import nn

from transdepth.engine.checkpoint import (
    atomic_torch_save,
    load_checkpoint,
    load_trainable_state,
    trainable_state_dict,
)
from transdepth.engine.schedule import build_global_schedule, rank_slot, schedule_sha256


def test_global_schedule_maps_three_ranks_and_four_microsteps_to_twelve_slots() -> None:
    schedule = build_global_schedule(100, updates=2, global_batch_size=12, seed=17)
    mapped = [rank_slot(schedule, 0, micro, rank, 3) for micro in range(4) for rank in range(3)]
    assert mapped == schedule[0].tolist()
    assert len(set(mapped)) == 12
    assert schedule_sha256(schedule) == schedule_sha256(schedule.clone())


def test_trainable_only_checkpoint_round_trip(tmp_path: Path) -> None:
    model = nn.Sequential(nn.Linear(3, 4), nn.Linear(4, 2))
    model[0].requires_grad_(False)
    state = trainable_state_dict(model)
    assert set(state) == {"1.weight", "1.bias"}
    expected = {name: value.clone() for name, value in state.items()}
    with torch.no_grad():
        model[1].weight.zero_()
        model[1].bias.zero_()
    load_trainable_state(model, expected)
    assert torch.equal(model[1].weight, expected["1.weight"])
    path = tmp_path / "checkpoint.pt"
    atomic_torch_save({"schema_version": "td_checkpoint_v1", "value": 3}, path)
    assert load_checkpoint(path)["value"] == 3


def test_partial_trainable_state_is_rejected_unless_explicitly_allowed() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    state = trainable_state_dict(model)
    del state["1.bias"]
    with pytest.raises(KeyError, match="omits trainable"):
        load_trainable_state(model, state)
    load_trainable_state(model, state, allowed_missing_substrings=("1.bias",))

