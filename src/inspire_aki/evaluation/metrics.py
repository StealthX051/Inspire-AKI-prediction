from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from inspire_aki.evaluation.bootstrap import bootstrap_metric_intervals
from inspire_aki.runtime import build_stage_runtime_plan, thread_limited_context


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)


def _summary_threshold(group_df: pd.DataFrame) -> float:
    thresholds = pd.to_numeric(group_df["threshold"], errors="coerce").dropna()
    if thresholds.empty:
        return np.nan
    unique = np.unique(thresholds.to_numpy(dtype=float))
    if unique.size == 1:
        return float(unique[0])
    return float(thresholds.mean())


def _metric_summary(group_df: pd.DataFrame) -> dict[str, float | int | str]:
    y_true = group_df["y_true"].astype(int).to_numpy()
    y_prob = group_df["y_prob_calibrated"].fillna(group_df["y_prob_raw"]).astype(float).to_numpy()
    y_pred = group_df["y_pred"].astype(int).to_numpy()
    auroc, auprc = _safe_auc(y_true, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "n_rows": int(len(group_df)),
        "n_positive": int(y_true.sum()),
        "auroc": auroc,
        "auprc": auprc,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "threshold": _summary_threshold(group_df),
    }


def _compute_metric_group_worker(keys: tuple, group_df: pd.DataFrame, nested_blas_threads: int) -> dict[str, object]:
    group_cols = ["dataset_regime", "population_id", "model_key", "repeat_id", "fold_id"]
    with thread_limited_context(nested_blas_threads):
        row = dict(zip(group_cols, keys))
        row.update(_metric_summary(group_df))
    return row


def compute_group_metrics(predictions_df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    group_cols = ["dataset_regime", "population_id", "model_key", "repeat_id", "fold_id"]
    groups = [(keys, group_df.copy()) for keys, group_df in predictions_df.groupby(group_cols, sort=False)]
    if not groups:
        return pd.DataFrame()
    if not isinstance(config, dict):
        rows = [_compute_metric_group_worker(keys, group_df, 1) for keys, group_df in groups]
        return pd.DataFrame(rows)
    runtime_plan = build_stage_runtime_plan(config, "evaluate_metrics", {"group_count": len(groups)})
    rows = Parallel(n_jobs=max(1, runtime_plan.evaluation_workers), backend="loky")(
        delayed(_compute_metric_group_worker)(keys, group_df, runtime_plan.nested_blas_threads)
        for keys, group_df in groups
    )
    return pd.DataFrame(rows)


def _summary_group_worker(
    keys: tuple,
    group_df: pd.DataFrame,
    config: dict,
    bootstrap_jobs: int,
    nested_blas_threads: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    with thread_limited_context(nested_blas_threads):
        row = dict(zip(["dataset_regime", "population_id", "model_key"], keys))
        row.update(_metric_summary(group_df))
        y_true = group_df["y_true"].astype(int).to_numpy()
        y_prob = group_df["y_prob_calibrated"].fillna(group_df["y_prob_raw"]).astype(float).to_numpy()
        y_pred = group_df["y_pred"].astype(int).to_numpy()
        bootstrap_df = bootstrap_metric_intervals(
            y_true,
            y_prob,
            None,
            y_pred=y_pred,
            n_bootstrap=config["evaluation"]["bootstrap_reps"],
            random_state=config["splits"]["random_state"],
            n_jobs=bootstrap_jobs,
        )
        if not bootstrap_df.empty:
            bootstrap_df.insert(0, "model_key", row["model_key"])
            bootstrap_df.insert(0, "population_id", row["population_id"])
            bootstrap_df.insert(0, "dataset_regime", row["dataset_regime"])
    return row, bootstrap_df


def summarize_group_metrics(predictions_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["dataset_regime", "population_id", "model_key"]
    groups = [(keys, group_df.copy()) for keys, group_df in predictions_df.groupby(group_cols, sort=False)]
    if not groups:
        return pd.DataFrame(), pd.DataFrame()
    runtime_plan = build_stage_runtime_plan(config, "evaluate_metrics", {"group_count": len(groups)})
    use_parallel_bootstrap = len(groups) < 4
    if use_parallel_bootstrap:
        results = [
            _summary_group_worker(
                keys,
                group_df,
                config,
                runtime_plan.bootstrap_workers,
                runtime_plan.nested_blas_threads,
            )
            for keys, group_df in groups
        ]
    else:
        results = Parallel(n_jobs=max(1, runtime_plan.evaluation_workers), backend="loky")(
            delayed(_summary_group_worker)(
                keys,
                group_df,
                config,
                1,
                runtime_plan.nested_blas_threads,
            )
            for keys, group_df in groups
        )
    summary_rows = [result[0] for result in results]
    bootstrap_rows = [result[1] for result in results if not result[1].empty]
    return pd.DataFrame(summary_rows), pd.concat(bootstrap_rows, ignore_index=True) if bootstrap_rows else pd.DataFrame()
