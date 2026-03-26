from __future__ import annotations

import numpy as np
import pandas as pd

from inspire_aki.registry import model_display_name

RECLASSIFICATION_PAIRS = (
    ("intraop", "preop"),
    ("preop", "combined"),
)


def _paired_reclassification_rows(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    *,
    source_regime: str,
    target_regime: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    common_models = sorted(set(source_df["model_key"]) & set(target_df["model_key"]))
    join_cols = ["repeat_id", "fold_id", "op_id", "split_name"]
    for model_key in common_models:
        source_model = source_df[source_df["model_key"] == model_key].copy()
        target_model = target_df[target_df["model_key"] == model_key].copy()
        if source_model.empty or target_model.empty:
            continue
        merged = source_model.merge(
            target_model,
            on=join_cols,
            how="inner",
            suffixes=("_source", "_target"),
        )
        if merged.empty:
            continue
        for (repeat_id, fold_id), fold_df in merged.groupby(["repeat_id", "fold_id"], sort=False):
            positives_missed = fold_df[(fold_df["y_true_source"].astype(int) == 1) & (fold_df["y_pred_source"].astype(int) == 0)]
            missed_count = float(len(positives_missed))
            reclassified_count = float((positives_missed["y_pred_target"].astype(int) == 1).sum())
            correction_rate = reclassified_count / missed_count if missed_count else np.nan
            rows.append(
                {
                    "model_key": model_key,
                    "model_name": model_display_name(model_key),
                    "source_regime": source_regime,
                    "target_regime": target_regime,
                    "comparison_name": f"{source_regime}_to_{target_regime}",
                    "repeat_id": int(repeat_id),
                    "fold_id": int(fold_id),
                    "missed_positives": missed_count,
                    "reclassified_positives": reclassified_count,
                    "correction_rate": correction_rate,
                }
            )
    return rows


def compute_reclassification_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame()
    test_df = predictions_df[predictions_df["split_name"].astype(str) == "test"].copy()
    if test_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for source_regime, target_regime in RECLASSIFICATION_PAIRS:
        source_df = test_df[test_df["dataset_regime"] == source_regime]
        target_df = test_df[test_df["dataset_regime"] == target_regime]
        rows.extend(
            _paired_reclassification_rows(
                source_df,
                target_df,
                source_regime=source_regime,
                target_regime=target_regime,
            )
        )

    if not rows:
        return pd.DataFrame()

    fold_df = pd.DataFrame(rows)
    summary = (
        fold_df.groupby(["model_key", "model_name", "source_regime", "target_regime", "comparison_name"], as_index=False)
        .agg(
            correction_rate=("correction_rate", "mean"),
            reclassified_positives=("reclassified_positives", "mean"),
            missed_positives=("missed_positives", "mean"),
            n_groups=("fold_id", "count"),
        )
        .sort_values(["comparison_name", "model_name"], kind="stable")
        .reset_index(drop=True)
    )
    return summary
