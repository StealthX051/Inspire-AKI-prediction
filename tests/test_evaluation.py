from __future__ import annotations

import pandas as pd

from inspire_aki.evaluation.calibration import calibrate_prediction_groups
from inspire_aki.evaluation.dca import decision_curve_table
from inspire_aki.evaluation.delong import delong_comparison_table
from inspire_aki.evaluation.metrics import compute_group_metrics, summarize_group_metrics


def _raw_predictions() -> pd.DataFrame:
    rows = []
    y_true = [0, 0, 1, 1, 0, 1]
    probs_strong = [0.05, 0.20, 0.75, 0.95, 0.30, 0.80]
    probs_weaker = [0.30, 0.35, 0.60, 0.65, 0.40, 0.55]
    for model_key, probs in [("log_reg", probs_strong), ("asa_rule", probs_weaker)]:
        for idx, (truth, prob) in enumerate(zip(y_true, probs), start=1):
            rows.append(
                {
                    "op_id": idx,
                    "dataset_regime": "preop",
                    "population_id": "preop",
                    "repeat_id": 0,
                    "fold_id": idx % 2,
                    "split_name": "test",
                    "model_key": model_key,
                    "target": "aki_boolean",
                    "y_true": truth,
                    "y_prob_raw": prob,
                    "y_prob_calibrated": None,
                    "y_pred": int(prob >= 0.5),
                    "threshold": 0.5,
                    "calibration_method": None,
                    "run_id": f"preop:{model_key}",
                }
            )
    return pd.DataFrame(rows)


def test_calibration_and_metric_tables(loaded_synthetic_config) -> None:
    calibrated = calibrate_prediction_groups(_raw_predictions(), loaded_synthetic_config)
    assert set(calibrated.thresholds["model_key"]) == {"log_reg", "asa_rule"}
    assert calibrated.predictions["y_prob_calibrated"].notna().all()
    assert calibrated.predictions["calibration_method"].isin(["isotonic_cv", "identity"]).all()

    fold_metrics = compute_group_metrics(calibrated.predictions)
    summary_metrics, bootstrap_metrics = summarize_group_metrics(calibrated.predictions, loaded_synthetic_config)
    assert {"auroc", "precision", "threshold"}.issubset(fold_metrics.columns)
    assert {"dataset_regime", "model_key", "auprc"}.issubset(summary_metrics.columns)
    assert not bootstrap_metrics.empty


def test_delong_and_dca_outputs(loaded_synthetic_config) -> None:
    calibrated = calibrate_prediction_groups(_raw_predictions(), loaded_synthetic_config)
    matrix_df, long_df = delong_comparison_table(calibrated.predictions)
    dca_df = decision_curve_table(calibrated.predictions, loaded_synthetic_config)

    assert "preop_log_reg" in matrix_df.index
    assert "preop_asa_rule" in matrix_df.columns
    assert not long_df.empty
    assert {"threshold_prob", "net_benefit_model", "model_key"}.issubset(dca_df.columns)
    assert set(dca_df["model_key"]) == {"log_reg", "asa_rule"}

