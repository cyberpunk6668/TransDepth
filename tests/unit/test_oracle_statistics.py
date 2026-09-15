from transdepth.oracle.statistics import confirm_candidate, summarize_candidate


def _row(sample_id: str, transparent: float, background: float, edge: float):
    return {
        "sample_id": sample_id,
        "leakage_group_id": f"group-{sample_id}",
        "metrics": {
            "transparent": {"mae_mm": transparent},
            "background": {"mae_mm": background},
            "edge": {"mae_mm": edge},
        },
    }


def test_oracle_confirm_requires_positive_lower_bound_and_constraints() -> None:
    baseline = [_row(str(i), 20.0 + i, 5.0, 6.0) for i in range(24)]
    candidate = [_row(str(i), 18.0 + i, 5.2, 6.2) for i in range(24)]
    summary = summarize_candidate(baseline, candidate)
    decision = confirm_candidate(summary, resamples=2000, seed=17, minimum_samples=20)
    assert summary["eligible"]
    assert decision["confirm_passed"]
    assert decision["one_sided_95_lower_mm"] > 0


def test_oracle_confirm_rejects_background_regression() -> None:
    baseline = [_row(str(i), 20.0, 5.0, 6.0) for i in range(24)]
    candidate = [_row(str(i), 18.0, 8.0, 6.0) for i in range(24)]
    summary = summarize_candidate(baseline, candidate)
    assert not summary["eligible"]
