from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inspire_aki.evaluation.calibration import _calibrate_group, calibrate_prediction_groups
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
    assert calibrated.predictions["calibration_method"].isin(
        ["isotonic_stratified_group_cv", "isotonic_group_cv", "identity", "identity_prespecified_asa_ge_3"]
    ).all()

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


def test_calibration_grouped_cv_keeps_repeated_op_ids_together(monkeypatch, loaded_synthetic_config) -> None:
    rows = []
    for repeat_id in [0, 1]:
        for fold_id in [0, 1]:
            for op_id, truth, raw_prob in [
                (101, 0, 0.2 + 0.05 * repeat_id + 0.01 * fold_id),
                (202, 1, 0.8 - 0.05 * repeat_id - 0.01 * fold_id),
                (303, 0, 0.3 + 0.02 * repeat_id + 0.01 * fold_id),
                (404, 1, 0.7 - 0.02 * repeat_id - 0.01 * fold_id),
            ]:
                rows.append(
                    {
                        "op_id": op_id,
                        "dataset_regime": "combined",
                        "population_id": "combined",
                        "repeat_id": repeat_id,
                        "fold_id": fold_id,
                        "split_name": "test",
                        "model_key": "log_reg",
                        "target": "aki_boolean",
                        "y_true": truth,
                        "y_prob_raw": raw_prob,
                        "y_prob_calibrated": None,
                        "y_pred": int(raw_prob >= 0.5),
                        "threshold": 0.5,
                        "calibration_method": None,
                        "run_id": f"combined:log_reg:r{repeat_id}:f{fold_id}",
                    }
                )

    seen: dict[str, object] = {}

    class FakeStratifiedGroupKFold:
        def __init__(self, n_splits: int, shuffle: bool, random_state: int) -> None:  # noqa: ARG002
            seen["n_splits"] = n_splits

        def split(self, x, y, groups=None):  # noqa: ARG002
            groups = np.asarray(groups)
            seen["groups"] = groups.copy()
            unique_groups = list(dict.fromkeys(groups.tolist()))
            midpoint = len(unique_groups) // 2
            test_groups = [set(unique_groups[:midpoint]), set(unique_groups[midpoint:])]
            index = np.arange(len(groups))
            for held_out_groups in test_groups:
                test_mask = np.isin(groups, list(held_out_groups))
                train_idx = index[~test_mask]
                test_idx = index[test_mask]
                assert not set(groups[train_idx]) & set(groups[test_idx])
                yield train_idx, test_idx

    monkeypatch.setattr("inspire_aki.evaluation.calibration.StratifiedGroupKFold", FakeStratifiedGroupKFold)

    calibrated_df, summary = _calibrate_group(pd.DataFrame(rows), loaded_synthetic_config)
    grouped_methods = {summary["calibration_method"]}

    assert grouped_methods <= {"isotonic_stratified_group_cv", "isotonic_group_cv"}
    assert seen["n_splits"] == min(loaded_synthetic_config["calibration"]["cv_folds"], 4)
    assert set(np.unique(seen["groups"])) == {101, 202, 303, 404}
    assert calibrated_df["y_prob_calibrated"].notna().all()


def test_calibration_preserves_prespecified_rule_thresholds_for_asa_and_gs_aki(loaded_synthetic_config) -> None:
    rows = []
    for model_key, probs in [
        ("asa_rule", [0.20, 0.50, 0.50, 0.83]),
        ("gs_aki_rule", [1.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 6.0 / 9.0]),
    ]:
        for idx, (truth, prob) in enumerate(zip([0, 0, 1, 1], probs), start=1):
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

    calibrated = calibrate_prediction_groups(pd.DataFrame(rows), loaded_synthetic_config)

    asa_threshold = calibrated.thresholds.loc[calibrated.thresholds["model_key"] == "asa_rule", "threshold"].iloc[0]
    gs_aki_threshold = calibrated.thresholds.loc[calibrated.thresholds["model_key"] == "gs_aki_rule", "threshold"].iloc[0]
    asa_method = calibrated.thresholds.loc[calibrated.thresholds["model_key"] == "asa_rule", "calibration_method"].iloc[0]
    gs_aki_method = calibrated.thresholds.loc[calibrated.thresholds["model_key"] == "gs_aki_rule", "calibration_method"].iloc[0]

    assert asa_threshold == pytest.approx(0.5)
    assert gs_aki_threshold == pytest.approx(4.0 / 9.0)
    assert asa_method == "identity_prespecified_asa_ge_3"
    assert gs_aki_method == "identity_prespecified_class_iii_plus"

    for model_key in ["asa_rule", "gs_aki_rule"]:
        subset = calibrated.predictions.loc[calibrated.predictions["model_key"] == model_key]
        assert np.allclose(subset["y_prob_calibrated"].to_numpy(), subset["y_prob_raw"].to_numpy())
