"""Selection constraints and one-sided paired bootstrap for FAR oracle candidates."""

from __future__ import annotations

from typing import Any

import numpy as np


def summarize_candidate(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    minimum_tolerance_mm: float = 1.0,
    relative_tolerance: float = 0.02,
) -> dict[str, Any]:
    base = {row["sample_id"]: row for row in baseline}
    changed = {row["sample_id"]: row for row in candidate}
    if set(base) != set(changed) or not base:
        raise ValueError("oracle candidate and baseline need the same nonempty sample set")
    ids = sorted(base)

    for sample_id in ids:
        if base[sample_id].get("leakage_group_id") != changed[sample_id].get(
            "leakage_group_id"
        ):
            raise ValueError("oracle baseline/candidate group IDs differ")

    def group_values(table: dict[str, dict[str, Any]], region: str) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for sample_id in ids:
            value = table[sample_id]["metrics"][region]["mae_mm"]
            if value is not None:
                group = table[sample_id]["leakage_group_id"]
                grouped.setdefault(group, []).append(float(value))
        return {group: float(np.mean(values)) for group, values in grouped.items()}

    def paired(region: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        base_groups = group_values(base, region)
        changed_groups = group_values(changed, region)
        groups = sorted(set(base_groups) & set(changed_groups))
        if not groups:
            raise ValueError(f"no paired groups for oracle region {region}")
        return (
            np.asarray([base_groups[group] for group in groups], dtype=np.float64),
            np.asarray([changed_groups[group] for group in groups], dtype=np.float64),
            groups,
        )

    base_t, changed_t, transparent_groups = paired("transparent")
    base_b, changed_b, _ = paired("background")
    base_e, changed_e, _ = paired("edge")
    tolerance_b = max(minimum_tolerance_mm, relative_tolerance * float(base_b.mean()))
    tolerance_e = max(minimum_tolerance_mm, relative_tolerance * float(base_e.mean()))
    improvement = base_t - changed_t
    regression_b = float(changed_b.mean() - base_b.mean())
    regression_e = float(changed_e.mean() - base_e.mean())
    eligible = (
        improvement.mean() > 0
        and regression_b <= tolerance_b
        and regression_e <= tolerance_e
    )
    return {
        "samples": len(ids),
        "groups": len(transparent_groups),
        "transparent_improvement_mean_mm": float(improvement.mean()),
        "transparent_improvement_per_group_mm": improvement.tolist(),
        "background_regression_mm": regression_b,
        "background_tolerance_mm": tolerance_b,
        "edge_regression_mm": regression_e,
        "edge_tolerance_mm": tolerance_e,
        "eligible": bool(eligible),
    }


def confirm_candidate(
    summary: dict[str, Any], *, resamples: int, seed: int, minimum_samples: int
) -> dict[str, Any]:
    differences = np.asarray(summary["transparent_improvement_per_group_mm"], dtype=np.float64)
    if differences.size < minimum_samples:
        return {
            **summary,
            "status": "insufficient_confirm_samples",
            "confirm_passed": False,
            "one_sided_95_lower_mm": None,
        }
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, differences.size, size=(resamples, differences.size))
    means = differences[indices].mean(axis=1)
    lower = float(np.quantile(means, 0.05, method="linear"))
    passed = bool(summary["eligible"] and differences.mean() > 0 and lower > 0)
    return {
        **summary,
        "status": "passed" if passed else "failed",
        "confirm_passed": passed,
        "one_sided_95_lower_mm": lower,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
    }
