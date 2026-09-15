"""Paired H2-minus-H1 evidence with one-sided group bootstrap."""

from __future__ import annotations

from typing import Any

import numpy as np


def paired_comparison(
    h1_rows: list[dict[str, Any]],
    h2_rows: list[dict[str, Any]],
    *,
    resamples: int = 2000,
    seed: int = 17,
) -> dict[str, Any]:
    h1 = {row["sample_id"]: row for row in h1_rows}
    h2 = {row["sample_id"]: row for row in h2_rows}
    if set(h1) != set(h2) or not h1:
        raise ValueError("H1 and H2 evaluations must contain the same nonempty sample set")
    ids = sorted(h1)
    for sample_id in ids:
        if h1[sample_id].get("leakage_group_id") != h2[sample_id].get(
            "leakage_group_id"
        ):
            raise ValueError("H1/H2 leakage group IDs differ")

    def group_values(method: dict[str, dict[str, Any]], region: str) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for sample_id in ids:
            value = method[sample_id]["metrics"][region]["mae_mm"]
            if value is not None:
                group = method[sample_id]["leakage_group_id"]
                grouped.setdefault(group, []).append(float(value))
        return {group: float(np.mean(values)) for group, values in grouped.items()}

    h1_transparent = group_values(h1, "transparent")
    h2_transparent = group_values(h2, "transparent")
    groups = sorted(set(h1_transparent) & set(h2_transparent))
    if not groups:
        raise ValueError("no paired leakage groups have transparent metrics")
    differences = np.asarray(
        [h1_transparent[group] - h2_transparent[group] for group in groups],
        dtype=np.float64,
    )
    if not np.isfinite(differences).all():
        raise ValueError("all paired samples need finite transparent MAE")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(groups), size=(resamples, len(groups)))
    bootstrap = differences[indices].mean(axis=1)
    baseline = float(np.mean([h1_transparent[group] for group in groups]))

    def mean_region(method: dict[str, dict[str, Any]], region: str) -> float:
        values = list(group_values(method, region).values())
        if not values:
            raise ValueError(f"no leakage groups have {region} metrics")
        result = float(np.mean(values))
        if not np.isfinite(result):
            raise ValueError(f"non-finite aggregate for {region}")
        return result

    background_regression = mean_region(h2, "background") - mean_region(h1, "background")
    edge_regression = mean_region(h2, "edge") - mean_region(h1, "edge")
    tolerance_background = max(1.0, 0.02 * mean_region(h1, "background"))
    tolerance_edge = max(1.0, 0.02 * mean_region(h1, "edge"))
    mean_improvement = float(differences.mean())
    lower_95 = float(np.quantile(bootstrap, 0.05, method="linear"))
    relative = mean_improvement / baseline if baseline > 0 else float("nan")
    passed = (
        mean_improvement > 0
        and lower_95 > 0
        and relative >= 0.02
        and background_regression <= tolerance_background
        and edge_regression <= tolerance_edge
    )
    return {
        "schema_version": "paired_far_comparison_v1",
        "status": "supports_far_on_rftrans_hold" if passed else "far_not_established",
        "paired_samples": len(ids),
        "paired_groups": len(groups),
        "transparent_mae_improvement_mm": mean_improvement,
        "transparent_relative_improvement": relative,
        "one_sided_95_lower_mm": lower_95,
        "background_regression_mm": background_regression,
        "background_tolerance_mm": tolerance_background,
        "edge_regression_mm": edge_regression,
        "edge_tolerance_mm": tolerance_edge,
        "bootstrap_resamples": resamples,
        "seed": seed,
        "scope": "RFTrans synthetic holdout only; ClearGrasp inference is qualitative.",
    }
