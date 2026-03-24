from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _metric_row(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        auroc = np.nan
        auprc = np.nan
        balanced = np.nan
    else:
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        balanced = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "auroc": auroc,
        "auprc": auprc,
        "balanced_accuracy": balanced,
        "accuracy": accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "f2": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
    }
    try:
        calib_model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
        calib_model.fit(y_prob.reshape(-1, 1), y_true)
        metrics["calib_intercept"] = float(calib_model.intercept_[0])
        metrics["calib_slope"] = float(calib_model.coef_[0, 0])
    except Exception:
        metrics["calib_intercept"] = np.nan
        metrics["calib_slope"] = np.nan
    return metrics


def _bootstrap_chunk(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    *,
    n_bootstrap: int,
    random_state: int,
) -> list[dict[str, float]]:
    rng = np.random.default_rng(random_state)
    bootstrap_rows: list[dict[str, float]] = []
    n_samples = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_samples, n_samples)
        y_true_boot = y_true[idx]
        y_prob_boot = y_prob[idx]
        if len(np.unique(y_true_boot)) < 2:
            continue
        bootstrap_rows.append(_metric_row(y_true_boot, y_prob_boot, threshold))
    return bootstrap_rows


def bootstrap_metric_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    *,
    n_bootstrap: int,
    random_state: int,
    n_jobs: int = 1,
) -> pd.DataFrame:
    if len(y_true) == 0:
        return pd.DataFrame()

    if n_jobs <= 1 or n_bootstrap <= 1:
        bootstrap_rows = _bootstrap_chunk(
            y_true,
            y_prob,
            threshold,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
    else:
        batch_sizes = [n_bootstrap // n_jobs] * n_jobs
        for idx in range(n_bootstrap % n_jobs):
            batch_sizes[idx] += 1
        bootstrap_rows_nested = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_bootstrap_chunk)(
                y_true,
                y_prob,
                threshold,
                n_bootstrap=batch_size,
                random_state=random_state + batch_idx,
            )
            for batch_idx, batch_size in enumerate(batch_sizes)
            if batch_size > 0
        )
        bootstrap_rows = [row for rows in bootstrap_rows_nested for row in rows]

    if not bootstrap_rows:
        return pd.DataFrame()

    bootstrap_df = pd.DataFrame(bootstrap_rows)
    out_rows = []
    for metric in bootstrap_df.columns:
        values = bootstrap_df[metric].dropna().to_numpy()
        if len(values) == 0:
            continue
        out_rows.append(
            {
                "metric": metric,
                "mean": float(np.mean(values)),
                "ci_lower_95": float(np.percentile(values, 2.5)),
                "ci_upper_95": float(np.percentile(values, 97.5)),
            }
        )
    return pd.DataFrame(out_rows)
