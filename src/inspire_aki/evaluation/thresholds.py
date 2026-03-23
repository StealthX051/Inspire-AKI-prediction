from __future__ import annotations

import numpy as np
from sklearn.metrics import fbeta_score


def find_optimal_fbeta_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    beta: float = 2.0,
    threshold_min: float = 0.01,
    threshold_max: float = 0.99,
    steps: int = 200,
) -> float:
    if len(y_true) == 0:
        return 0.5
    if len(np.unique(y_true)) < 2:
        return 0.5
    thresholds = np.linspace(threshold_min, threshold_max, steps)
    scores = [fbeta_score(y_true, y_prob >= threshold, beta=beta, zero_division=0) for threshold in thresholds]
    return float(thresholds[int(np.argmax(scores))])

