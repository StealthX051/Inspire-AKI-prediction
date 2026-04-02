from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import IsotonicRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from inspire_aki.clinical_baselines import clinical_rule_calibration_method, clinical_rule_probability_threshold
from inspire_aki.evaluation.thresholds import find_optimal_fbeta_threshold
from inspire_aki.runtime import build_stage_runtime_plan, thread_limited_context


@dataclass
class CalibrationResult:
    predictions: pd.DataFrame
    thresholds: pd.DataFrame


def _calibrate_group(group_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict[str, object]]:
    calibration_cfg = config["calibration"]
    group_df = group_df.sort_values(["repeat_id", "fold_id", "op_id"]).reset_index(drop=True).copy()
    model_key = str(group_df["model_key"].iat[0])
    y_true = group_df["y_true"].astype(int).to_numpy()
    y_prob_raw = group_df["y_prob_raw"].astype(float).to_numpy()

    rule_threshold = clinical_rule_probability_threshold(model_key, config)
    if rule_threshold is not None:
        y_prob_calibrated = y_prob_raw.copy()
        threshold = float(rule_threshold)
        method_used = str(clinical_rule_calibration_method(model_key) or "identity")
    else:
        method = calibration_cfg["method"]
        if method != "isotonic":
            raise ValueError(f"Unsupported calibration method: {method}")

        unique_classes = np.unique(y_true)
        groups = group_df["op_id"].to_numpy()
        n_unique_groups = int(pd.Index(groups).nunique())
        n_splits = min(calibration_cfg["cv_folds"], n_unique_groups)
        if len(unique_classes) < 2 or n_unique_groups < 2 or n_splits < 2:
            y_prob_calibrated = y_prob_raw.copy()
            threshold = 0.5
            method_used = "identity"
        else:
            y_prob_calibrated = np.zeros_like(y_prob_raw, dtype=float)
            try:
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=config["splits"]["random_state"],
                )
                splits = list(splitter.split(y_prob_raw.reshape(-1, 1), y_true, groups=groups))
                method_used = "isotonic_stratified_group_cv"
            except ValueError:
                splitter = GroupKFold(n_splits=n_splits)
                splits = list(splitter.split(y_prob_raw.reshape(-1, 1), y_true, groups=groups))
                method_used = "isotonic_group_cv"
            for train_idx, test_idx in splits:
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(y_prob_raw[train_idx], y_true[train_idx])
                y_prob_calibrated[test_idx] = calibrator.predict(y_prob_raw[test_idx])
            threshold = find_optimal_fbeta_threshold(
                y_true,
                y_prob_calibrated,
                beta=2.0,
                threshold_min=calibration_cfg["threshold_min"],
                threshold_max=calibration_cfg["threshold_max"],
                steps=calibration_cfg["threshold_steps"],
            )

    group_df["y_prob_calibrated"] = y_prob_calibrated
    group_df["threshold"] = threshold
    group_df["y_pred"] = (group_df["y_prob_calibrated"] >= threshold).astype(int)
    group_df["calibration_method"] = method_used

    summary = {
        "dataset_regime": group_df["dataset_regime"].iat[0],
        "population_id": group_df["population_id"].iat[0],
        "model_key": model_key,
        "calibration_method": method_used,
        "threshold": float(threshold),
        "n_rows": int(len(group_df)),
        "n_positive": int(y_true.sum()),
    }
    return group_df, summary


def _calibrate_group_worker(group_df: pd.DataFrame, config: dict, nested_blas_threads: int) -> tuple[pd.DataFrame, dict[str, object]]:
    with thread_limited_context(nested_blas_threads):
        return _calibrate_group(group_df, config)


def calibrate_prediction_groups(predictions_df: pd.DataFrame, config: dict) -> CalibrationResult:
    if predictions_df.empty:
        return CalibrationResult(predictions=predictions_df.copy(), thresholds=pd.DataFrame())

    group_cols = ["dataset_regime", "population_id", "model_key"]
    groups = [group_df.copy() for _, group_df in predictions_df.groupby(group_cols, sort=False)]
    runtime_plan = build_stage_runtime_plan(config, "evaluate_calibration", {"group_count": len(groups)})
    results = Parallel(n_jobs=max(1, runtime_plan.evaluation_workers), backend="loky")(
        delayed(_calibrate_group_worker)(group_df, config, runtime_plan.nested_blas_threads)
        for group_df in groups
    )
    calibrated_groups = [result[0] for result in results]
    threshold_rows = [result[1] for result in results]

    return CalibrationResult(
        predictions=pd.concat(calibrated_groups, ignore_index=True),
        thresholds=pd.DataFrame(threshold_rows),
    )
