import numpy as np

from transdepth.evaluation.compare import paired_comparison
from transdepth.evaluation.metrics import evaluate_image


def test_metrics_use_fixed_gt_domain_and_metric_units() -> None:
    target = np.ones((384, 512), dtype=np.float32)
    prediction = np.full_like(target, 1.01)
    valid = np.ones_like(target, dtype=bool)
    mask = np.zeros_like(target, dtype=np.uint8)
    mask[:, 128:384] = 1
    content = np.ones_like(valid)
    result = evaluate_image(prediction, target, valid, mask, content)
    assert np.isclose(result["all"]["mae_mm"], 10.0, atol=1e-3)
    assert result["all"]["delta_1.05"] == 1.0
    assert result["transparent"]["pixels"] == 384 * 256
    assert result["edge"]["pixels"] > 0


def _evaluation(sample: str, transparent: float, background: float, edge: float):
    return {
        "sample_id": sample,
        "leakage_group_id": f"group-{sample}",
        "metrics": {
            "transparent": {"mae_mm": transparent},
            "background": {"mae_mm": background},
            "edge": {"mae_mm": edge},
        },
    }


def test_paired_comparison_accepts_consistent_far_improvement() -> None:
    h1 = [_evaluation(str(i), 10.0 + i, 5.0, 6.0) for i in range(20)]
    h2 = [_evaluation(str(i), 9.0 + 0.9 * i, 5.1, 6.1) for i in range(20)]
    result = paired_comparison(h1, h2)
    assert result["status"] == "supports_far_on_rftrans_hold"
    assert result["one_sided_95_lower_mm"] > 0
