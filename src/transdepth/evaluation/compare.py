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
    differences = np.array(
        [
            h1[item]["metrics"]["transparent"]["mae_mm"]
            - h2[item]["metrics"]["transparent"]["mae_mm"]
            for item in ids
        ],
        dtype=np.float64,
    )
    if not np.isfinite(differences).all():
        raise ValueError("all paired samples need finite transparent MAE")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(ids), size=(resamples, len(ids)))
    bootstrap = differences[indices].mean(axis=1)
    baseline = float(
        np.mean([h1[item]["metrics"]["transparent"]["mae_mm"] for item in ids])
    )

    def mean_region(method: dict[str, dict[str, Any]], region: str) -> float:
        values = [method[item]["metrics"][region]["mae_mm"] for item in ids]
        return float(np.mean([value for value in values if value is not None]))

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
