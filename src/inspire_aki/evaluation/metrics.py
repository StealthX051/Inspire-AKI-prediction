from __future__ import annotations

import numpy as np
import pandas as pd
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


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    return roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)


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
        "threshold": float(group_df["threshold"].iloc[0]),
    }


def compute_group_metrics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset_regime", "population_id", "model_key", "repeat_id", "fold_id"]
    rows: list[dict[str, object]] = []
    for keys, group_df in predictions_df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row.update(_metric_summary(group_df))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_group_metrics(predictions_df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_cols = ["dataset_regime", "population_id", "model_key"]
    summary_rows: list[dict[str, object]] = []
    bootstrap_rows: list[pd.DataFrame] = []

    for keys, group_df in predictions_df.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, keys))
        row.update(_metric_summary(group_df))
        summary_rows.append(row)

        y_true = group_df["y_true"].astype(int).to_numpy()
        y_prob = group_df["y_prob_calibrated"].fillna(group_df["y_prob_raw"]).astype(float).to_numpy()
        threshold = float(group_df["threshold"].iloc[0])
        bootstrap_df = bootstrap_metric_intervals(
            y_true,
            y_prob,
            threshold,
            n_bootstrap=config["evaluation"]["bootstrap_reps"],
            random_state=config["splits"]["random_state"],
        )
        if not bootstrap_df.empty:
            bootstrap_df.insert(0, "model_key", row["model_key"])
            bootstrap_df.insert(0, "population_id", row["population_id"])
            bootstrap_df.insert(0, "dataset_regime", row["dataset_regime"])
            bootstrap_rows.append(bootstrap_df)

    return pd.DataFrame(summary_rows), pd.concat(bootstrap_rows, ignore_index=True) if bootstrap_rows else pd.DataFrame()

