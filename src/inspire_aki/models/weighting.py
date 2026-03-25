from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score


def positive_balance_weight(y: Any) -> float:
    labels = np.asarray(y, dtype=int)
    positives = int(np.sum(labels == 1))
    negatives = int(np.sum(labels == 0))
    if positives == 0 or negatives == 0:
        return 1.0
    return negatives / positives


def balance_sample_weights(y: Any, *, positive_weight: float | None = None) -> np.ndarray:
    labels = np.asarray(y, dtype=int)
    pos_weight = positive_balance_weight(labels) if positive_weight is None else float(positive_weight)
    return np.where(labels == 1, pos_weight, 1.0).astype(float)


def balance_weight_series(y: pd.Series, *, positive_weight: float | None = None) -> pd.Series:
    return pd.Series(balance_sample_weights(y.to_numpy(), positive_weight=positive_weight), index=y.index, dtype=float)


def safe_balanced_accuracy(y_true: Any, y_pred: Any) -> float:
    labels = np.asarray(y_true, dtype=int)
    predictions = np.asarray(y_pred, dtype=int)
    if len(np.unique(labels)) < 2:
        return 0.5
    return float(balanced_accuracy_score(labels, predictions))


def weighted_resample_indices(y: Any, *, seed: int, n_samples: int | None = None, positive_weight: float | None = None) -> np.ndarray:
    weights = balance_sample_weights(y, positive_weight=positive_weight)
    probabilities = weights / weights.sum()
    sample_count = len(weights) if n_samples is None else int(n_samples)
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(len(weights)), size=sample_count, replace=True, p=probabilities)


def weighted_resample_for_knn(
    x: pd.DataFrame | np.ndarray,
    y: Any,
    *,
    seed: int,
    positive_weight: float | None = None,
) -> tuple[pd.DataFrame | np.ndarray, np.ndarray]:
    indices = weighted_resample_indices(y, seed=seed, positive_weight=positive_weight)
    if isinstance(x, pd.DataFrame):
        resampled_x: pd.DataFrame | np.ndarray = x.iloc[indices].reset_index(drop=True)
    else:
        resampled_x = np.asarray(x)[indices]
    resampled_y = np.asarray(y, dtype=int)[indices]
    return resampled_x, resampled_y
