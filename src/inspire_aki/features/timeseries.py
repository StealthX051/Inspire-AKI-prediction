from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from inspire_aki.features.normalization import replace_outliers


def build_clean_timeseries(vitals_df: pd.DataFrame, op_ids: pd.Series, config: dict) -> pd.DataFrame:
    feature_cfg = config["features"]
    seq_cfg = config["sequence"]
    regular_labels = feature_cfg["high_frequency_labels"] + feature_cfg["medium_frequency_labels"]

    df_vitals = vitals_df[vitals_df["op_id"].isin(op_ids.unique())].copy()
    df_vitals = df_vitals.drop_duplicates(subset=["op_id", "chart_time", "item_name"], keep="first")
    df_regular = df_vitals.loc[df_vitals["item_name"].isin(regular_labels), ["op_id", "item_name", "value", "chart_time"]]

    contained = []
    for label in regular_labels:
        label_df = df_regular.loc[df_regular["item_name"] == label].copy()
        label_df = replace_outliers(label_df, {"op_id", "item_name", "chart_time"}, config)
        contained.append(label_df)
    df_regular = pd.concat(contained, ignore_index=True) if contained else df_regular

    interpolated = []
    step = seq_cfg["interpolation_step_minutes"]
    for op_id, group in df_regular.groupby("op_id"):
        op_frame = group[["item_name", "value", "chart_time"]]
        df_complete = pd.DataFrame({"chart_time": np.arange(op_frame["chart_time"].min(), op_frame["chart_time"].max() + step, step)})
        op_frame = op_frame.pivot(index="chart_time", columns="item_name", values="value")
        df_complete = df_complete.merge(op_frame, on="chart_time", how="left")
        df_complete.fillna(df_complete.mean(), inplace=True)
        df_complete["op_id"] = op_id
        interpolated.append(df_complete)

    df_final = pd.concat(interpolated, ignore_index=True) if interpolated else pd.DataFrame(columns=["op_id", "chart_time"])
    if seq_cfg["normalize_timeseries"] and not df_final.empty:
        ignore = {"op_id", "chart_time", "aki"}
        cols_to_norm = [col for col in df_final.columns if col not in ignore]
        scaler = StandardScaler()
        df_final[cols_to_norm] = scaler.fit_transform(df_final[cols_to_norm])
    return df_final
