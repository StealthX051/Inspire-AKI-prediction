from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from inspire_aki.evaluation.calibration import _calibrate_group, calibrate_prediction_groups
from inspire_aki.evaluation.dca import decision_curve_table
from inspire_aki.evaluation.delong import delong_comparison_table
from inspire_aki.evaluation.metrics import compute_group_metrics, summarize_group_metrics


def _grouped_raw_predictions() -> pd.DataFrame:
    rows = []
    calibration_truth = [0, 0, 1, 1]
    calibration_probs_strong = [0.05, 0.20, 0.75, 0.95]
    calibration_probs_rule = [0.30, 0.35, 0.60, 0.65]
    test_truth = [0, 1, 0, 1]
    test_probs_strong = [0.10, 0.85, 0.25, 0.90]
    test_probs_rule = [0.45, 0.70, 0.35, 0.80]
    for model_key, calibration_probs, test_probs in [
        ("log_reg", calibration_probs_strong, test_probs_strong),
        ("asa_rule", calibration_probs_rule, test_probs_rule),
    ]:
        for idx, (truth, prob) in enumerate(zip(calibration_truth, calibration_probs), start=1):
            rows.append(
                {
                    "op_id": idx,
                    "patient_id": idx,
                    "dataset_regime": "preop",
                    "population_id": "preop",
                    "repeat_id": 0,
                    "fold_id": 0,
                    "split_name": "calibration",
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
        for idx, (truth, prob) in enumerate(zip(test_truth, test_probs), start=101):
            rows.append(
                {
                    "op_id": idx,
                    "patient_id": idx,
                    "dataset_regime": "preop",
                    "population_id": "preop",
                    "repeat_id": 0,
                    "fold_id": 0,
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
    config = dict(loaded_synthetic_config)
    config["evaluation_mode"] = "grouped_holdout"
    calibrated = calibrate_prediction_groups(_grouped_raw_predictions(), config)
    assert set(calibrated.thresholds["model_key"]) == {"log_reg", "asa_rule"}
    assert {"repeat_id", "fold_id", "n_calibration_rows", "n_test_rows"}.issubset(calibrated.thresholds.columns)
    assert set(calibrated.predictions["split_name"]) == {"test"}
    assert calibrated.predictions["y_prob_calibrated"].notna().all()
    assert calibrated.predictions["calibration_method"].isin(
        ["isotonic_outer_train_oof", "identity_prespecified_asa_ge_3", "identity_insufficient_calibration_support"]
    ).all()

    fold_metrics = compute_group_metrics(calibrated.predictions)
    summary_metrics, bootstrap_metrics = summarize_group_metrics(calibrated.predictions, config)
    assert {"auroc", "precision", "threshold"}.issubset(fold_metrics.columns)
    assert {"dataset_regime", "model_key", "auprc"}.issubset(summary_metrics.columns)
    assert not bootstrap_metrics.empty


def test_delong_and_dca_outputs(loaded_synthetic_config) -> None:
    config = dict(loaded_synthetic_config)
    config["evaluation_mode"] = "grouped_holdout"
    calibrated = calibrate_prediction_groups(_grouped_raw_predictions(), config)
    matrix_df, long_df = delong_comparison_table(calibrated.predictions)
    dca_df = decision_curve_table(calibrated.predictions, config)

    assert "preop_log_reg" in matrix_df.index
    assert "preop_asa_rule" in matrix_df.columns
    assert not long_df.empty
    assert {"threshold_prob", "net_benefit_model", "model_key"}.issubset(dca_df.columns)
    assert set(dca_df["model_key"]) == {"log_reg", "asa_rule"}


def test_grouped_summary_metrics_use_stored_row_level_predictions(monkeypatch, loaded_synthetic_config) -> None:
    config = dict(loaded_synthetic_config)
    config["evaluation_mode"] = "grouped_nested_cv"
    seen: dict[str, object] = {}
    rows = [
        {
            "op_id": 1,
            "patient_id": 1,
            "dataset_regime": "combined",
            "population_id": "combined",
            "repeat_id": 0,
            "fold_id": 0,
            "split_name": "test",
            "model_key": "log_reg",
            "target": "aki_boolean",
            "y_true": 1,
            "y_prob_raw": 0.30,
            "y_prob_calibrated": 0.30,
            "y_pred": 1,
            "threshold": 0.20,
            "calibration_method": "isotonic_outer_train_oof",
            "run_id": "combined:log_reg:r0:f0",
        },
        {
            "op_id": 2,
            "patient_id": 2,
            "dataset_regime": "combined",
            "population_id": "combined",
            "repeat_id": 0,
            "fold_id": 1,
            "split_name": "test",
            "model_key": "log_reg",
            "target": "aki_boolean",
            "y_true": 1,
            "y_prob_raw": 0.30,
            "y_prob_calibrated": 0.30,
            "y_pred": 0,
            "threshold": 0.80,
            "calibration_method": "isotonic_outer_train_oof",
            "run_id": "combined:log_reg:r0:f1",
        },
    ]

    def _fake_bootstrap_metric_intervals(y_true, y_prob, threshold, *, y_pred=None, **_kwargs):
        seen["y_true"] = np.asarray(y_true, dtype=int)
        seen["y_prob"] = np.asarray(y_prob, dtype=float)
        seen["threshold"] = threshold
        seen["y_pred"] = None if y_pred is None else np.asarray(y_pred, dtype=int)
        return pd.DataFrame([{"metric": "accuracy", "mean": 0.5, "ci_lower_95": 0.1, "ci_upper_95": 0.9}])

    monkeypatch.setattr("inspire_aki.evaluation.metrics.bootstrap_metric_intervals", _fake_bootstrap_metric_intervals)

    summary_metrics, bootstrap_metrics = summarize_group_metrics(pd.DataFrame(rows), config)

    assert summary_metrics.loc[0, "threshold"] == pytest.approx(0.5)
    assert seen["threshold"] is None
    assert np.array_equal(seen["y_pred"], np.array([1, 0]))
    assert np.array_equal(seen["y_true"], np.array([1, 1]))
    assert np.allclose(seen["y_prob"], np.array([0.30, 0.30]))
    assert not bootstrap_metrics.empty


def test_grouped_dca_suppresses_optimal_threshold_annotation_when_folds_disagree(loaded_synthetic_config) -> None:
    config = dict(loaded_synthetic_config)
    config["evaluation_mode"] = "grouped_nested_cv"
    rows = [
        {
            "op_id": 1,
            "patient_id": 1,
            "dataset_regime": "combined",
            "population_id": "combined",
            "repeat_id": 0,
            "fold_id": 0,
            "split_name": "test",
            "model_key": "log_reg",
            "target": "aki_boolean",
            "y_true": 0,
            "y_prob_raw": 0.20,
            "y_prob_calibrated": 0.20,
            "y_pred": 1,
            "threshold": 0.10,
            "calibration_method": "isotonic_outer_train_oof",
            "run_id": "combined:log_reg:r0:f0",
        },
        {
            "op_id": 2,
            "patient_id": 2,
            "dataset_regime": "combined",
            "population_id": "combined",
            "repeat_id": 0,
            "fold_id": 1,
            "split_name": "test",
            "model_key": "log_reg",
            "target": "aki_boolean",
            "y_true": 1,
            "y_prob_raw": 0.80,
            "y_prob_calibrated": 0.80,
            "y_pred": 0,
            "threshold": 0.90,
            "calibration_method": "isotonic_outer_train_oof",
            "run_id": "combined:log_reg:r0:f1",
        },
    ]

    dca_df = decision_curve_table(pd.DataFrame(rows), config)

    assert dca_df["threshold_optimal"].isna().all()


def test_grouped_calibration_fits_only_on_calibration_rows_and_returns_test_rows(monkeypatch, loaded_synthetic_config) -> None:
    config = dict(loaded_synthetic_config)
    config["evaluation_mode"] = "grouped_holdout"
    seen: dict[str, list[np.ndarray]] = {"predict_calls": []}

    class FakeIsotonicRegression:
        def __init__(self, out_of_bounds: str) -> None:
            seen["out_of_bounds"] = [np.array([out_of_bounds], dtype=object)]

        def fit(self, x, y):
            seen["fit_x"] = [np.asarray(x, dtype=float)]
            seen["fit_y"] = [np.asarray(y, dtype=int)]
            return self

        def predict(self, x):
            values = np.asarray(x, dtype=float)
            seen["predict_calls"].append(values.copy())
            return values

    monkeypatch.setattr("inspire_aki.evaluation.calibration.IsotonicRegression", FakeIsotonicRegression)

    calibrated_df, summary = _calibrate_group(
        _grouped_raw_predictions().loc[lambda df: df["model_key"] == "log_reg"].copy(),
        config,
    )

    assert np.allclose(seen["fit_x"][0], np.array([0.05, 0.20, 0.75, 0.95]))
    assert np.array_equal(seen["fit_y"][0], np.array([0, 0, 1, 1]))
    assert len(seen["predict_calls"]) == 2
    assert np.allclose(seen["predict_calls"][0], np.array([0.05, 0.20, 0.75, 0.95]))
    assert np.allclose(seen["predict_calls"][1], np.array([0.10, 0.85, 0.25, 0.90]))
    assert set(calibrated_df["split_name"]) == {"test"}
    assert calibrated_df["op_id"].tolist() == [101, 102, 103, 104]
    assert summary["n_calibration_rows"] == 4
    assert summary["n_test_rows"] == 4
    assert summary["calibration_method"] == "isotonic_outer_train_oof"


def test_calibration_preserves_prespecified_rule_thresholds_for_asa_and_gs_aki(loaded_synthetic_config) -> None:
    config = dict(loaded_synthetic_config)
    config["evaluation_mode"] = "grouped_holdout"
    rows = []
    for model_key, probs in [
        ("asa_rule", [0.20, 0.50, 0.50, 0.83]),
        ("gs_aki_rule", [1.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 6.0 / 9.0]),
    ]:
        for idx, (truth, prob) in enumerate(zip([0, 0, 1, 1], probs), start=1):
            rows.append(
                {
                    "op_id": idx,
                    "patient_id": idx,
                    "dataset_regime": "preop",
                    "population_id": "preop",
                    "repeat_id": 0,
                    "fold_id": 0,
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

    calibrated = calibrate_prediction_groups(pd.DataFrame(rows), config)

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
