from __future__ import annotations

import numpy as np
import pandas as pd


def build_sequence_dataset(tabular_df: pd.DataFrame, timeseries_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    seq_cfg = config["sequence"]
    df_time = timeseries_df.copy()
    feature_mask = df_time.drop(columns=["op_id", "chart_time"]).notna().astype(int)
    regular_feature_count = len(config["features"]["high_frequency_labels"] + config["features"]["medium_frequency_labels"])
    df_time["presence"] = feature_mask.sum(axis=1) / regular_feature_count
    df_time = df_time[df_time["presence"] > seq_cfg["presence_threshold"]].drop(columns=["presence"])
    df_time = df_time.fillna(seq_cfg["fill_value"])

    df_tabular = tabular_df.copy()
    bool_cols = df_tabular.select_dtypes(include="bool").columns
    df_tabular[bool_cols] = df_tabular[bool_cols].astype(float)

    padded_tensors = []
    op_ids = []
    sequence_lengths = []
    pad_length = seq_cfg["pad_length"]
    for op_id, group in df_time.groupby("op_id"):
        mat = group.drop(columns=["op_id", "chart_time"]).to_numpy(dtype=float)
        if mat.shape[0] < pad_length:
            padded = np.pad(mat, pad_width=((0, pad_length - mat.shape[0]), (0, 0)), mode="constant", constant_values=0.0)
            padded_tensors.append(padded)
            op_ids.append(op_id)
            sequence_lengths.append(mat.shape[0])

    df_sequence = pd.DataFrame({
        "op_id": op_ids,
        "time_tensors": padded_tensors,
        "seq_len": sequence_lengths,
    })
    return df_sequence.merge(df_tabular, on="op_id", how="inner")
