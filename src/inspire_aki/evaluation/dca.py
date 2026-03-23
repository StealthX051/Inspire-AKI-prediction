from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_net_benefit_for_thresholds(y_true: np.ndarray, y_prob: np.ndarray, pt_grid: np.ndarray) -> np.ndarray:
    benefits = []
    for pt in pt_grid:
        safe_pt = min(float(pt), 0.99999)
        y_pred = y_prob >= safe_pt
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        n = len(y_true)
        benefits.append((tp / n) - (fp / n) * (safe_pt / (1 - safe_pt)))
    return np.asarray(benefits)


def decision_curve_table(predictions_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    eval_cfg = config["evaluation"]
    pt_grid = np.arange(
        eval_cfg["dca_threshold_min"],
        eval_cfg["dca_threshold_max"] + eval_cfg["dca_threshold_step"] / 2,
        eval_cfg["dca_threshold_step"],
    )

    rows: list[dict[str, object]] = []
    group_cols = ["dataset_regime", "population_id", "model_key"]
    for keys, group_df in predictions_df.groupby(group_cols, sort=False):
        y_true = group_df["y_true"].astype(int).to_numpy()
        y_prob = group_df["y_prob_calibrated"].fillna(group_df["y_prob_raw"]).astype(float).to_numpy()
        prevalence = float(np.mean(y_true))
        model_nb = calculate_net_benefit_for_thresholds(y_true, y_prob, pt_grid)
        treat_all = prevalence - (1 - prevalence) * (pt_grid / (1 - pt_grid))
        for threshold, net_benefit_model, net_benefit_all in zip(pt_grid, model_nb, treat_all):
            rows.append(
                {
                    "dataset_regime": keys[0],
                    "population_id": keys[1],
                    "model_key": keys[2],
                    "threshold_prob": float(threshold),
                    "net_benefit_model": float(net_benefit_model),
                    "net_benefit_treat_all": float(net_benefit_all),
                    "net_benefit_treat_none": 0.0,
                }
            )
    return pd.DataFrame(rows)

