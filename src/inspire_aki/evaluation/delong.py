from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm

from inspire_aki.runtime import build_stage_runtime_plan


@dataclass
class DeLongResult:
    matrix: pd.DataFrame
    long: pd.DataFrame
    corrected_matrix: pd.DataFrame
    corrected_long: pd.DataFrame


def _structural_components(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    y_true = y_true.astype(int).ravel()
    scores = scores.ravel()
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    n_pos = pos.size
    n_neg = neg.size
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Need at least one positive and one negative sample.")

    neg_sorted = np.sort(neg)
    pos_sorted = np.sort(pos)

    left = np.searchsorted(neg_sorted, pos, side="left")
    right = np.searchsorted(neg_sorted, pos, side="right")
    v10 = (left + 0.5 * (right - left)) / n_neg

    left = np.searchsorted(pos_sorted, neg, side="left")
    right = np.searchsorted(pos_sorted, neg, side="right")
    v01 = (n_pos - right + 0.5 * (right - left)) / n_pos
    return v10.mean(), v10, v01


def delong_test(y_true: np.ndarray, scores1: np.ndarray, scores2: np.ndarray) -> tuple[float, float, float]:
    auc1, v10_1, v01_1 = _structural_components(y_true, scores1)
    auc2, v10_2, v01_2 = _structural_components(y_true, scores2)
    n_pos, n_neg = v10_1.size, v01_1.size
    s10 = np.cov(np.vstack([v10_1, v10_2]), ddof=1)
    s01 = np.cov(np.vstack([v01_1, v01_2]), ddof=1)
    var_auc_diff = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / n_pos + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / n_neg
    if var_auc_diff <= 0 or not np.isfinite(var_auc_diff):
        return auc1, auc2, 1.0
    z_score = (auc1 - auc2) / np.sqrt(var_auc_diff)
    p_value = 2 * (1 - norm.cdf(abs(z_score)))
    return auc1, auc2, float(p_value)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    if p_values.size == 0:
        return p_values
    order = np.argsort(p_values)
    ordered = p_values[order]
    n = float(len(ordered))
    adjusted = np.empty_like(ordered, dtype=float)
    running_min = 1.0
    for idx in range(len(ordered) - 1, -1, -1):
        rank = idx + 1
        candidate = ordered[idx] * n / rank
        running_min = min(running_min, candidate)
        adjusted[idx] = running_min
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return np.clip(adjusted[inverse], 0.0, 1.0)


def _apply_fdr_correction(matrix: pd.DataFrame, long_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if long_df.empty:
        return matrix.copy(), long_df.copy()
    corrected_long = long_df.copy()
    corrected_values = _benjamini_hochberg(corrected_long["p_value"].to_numpy(dtype=float))
    corrected_long["p_value"] = corrected_values
    corrected_matrix = matrix.copy()
    for _, row in corrected_long.iterrows():
        corrected_matrix.loc[str(row["model_left"]), str(row["model_right"])] = float(row["p_value"])
    return corrected_matrix, corrected_long


def delong_comparison_outputs(predictions_df: pd.DataFrame, config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_cols = ["dataset_regime", "population_id", "model_key"]
    prepared: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for (dataset_regime, population_id, model_key), group_df in predictions_df.groupby(group_cols, sort=False):
        group_df = group_df.sort_values(["repeat_id", "fold_id", "op_id"])
        name = f"{dataset_regime}_{model_key}"
        prepared[name] = (
            group_df["y_true"].astype(int).to_numpy(),
            group_df["y_prob_calibrated"].fillna(group_df["y_prob_raw"]).astype(float).to_numpy(),
        )

    model_names = sorted(prepared)
    matrix = pd.DataFrame(index=model_names, columns=model_names, dtype=object)

    def _pair_worker(left_name: str, right_name: str) -> tuple[str, str, object, dict[str, object] | None]:
        if left_name == right_name:
            return left_name, right_name, np.nan, None
        y_left, p_left = prepared[left_name]
        y_right, p_right = prepared[right_name]
        if len(y_left) != len(y_right) or not np.array_equal(y_left, y_right):
            return left_name, right_name, "N/A", None
        auc_left, auc_right, p_value = delong_test(y_left, p_left, p_right)
        return left_name, right_name, p_value, {
            "model_left": left_name,
            "model_right": right_name,
            "auc_left": auc_left,
            "auc_right": auc_right,
            "p_value": p_value,
        }

    if isinstance(config, dict):
        n_jobs = max(1, build_stage_runtime_plan(config, "evaluate_delong").evaluation_workers)
    else:
        n_jobs = 1
    pair_results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_pair_worker)(left_name, right_name)
        for left_name, right_name in product(model_names, model_names)
    )
    long_rows: list[dict[str, object]] = []
    for left_name, right_name, value, row in pair_results:
        matrix.loc[left_name, right_name] = value
        if row is not None:
            long_rows.append(row)
    long_df = pd.DataFrame(long_rows)
    corrected_matrix, corrected_long = _apply_fdr_correction(matrix, long_df)
    return matrix, long_df, corrected_matrix, corrected_long


def delong_comparison_table(predictions_df: pd.DataFrame, config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix, long_df, _, _ = delong_comparison_outputs(predictions_df, config)
    return matrix, long_df
