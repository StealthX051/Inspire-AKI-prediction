from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import entropy, kurtosis, skew


def trend(values: np.ndarray) -> float:
    x = np.arange(len(values)).T
    x = np.vstack((np.ones(len(x)), x)).T
    return (np.linalg.pinv(x) @ values.T)[1]


def energy(values: np.ndarray) -> float:
    return float(np.inner(values, values))


def build_intraop_features(vitals_df: pd.DataFrame, preop_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    feature_cfg = config["features"]
    regular_labels = feature_cfg["high_frequency_labels"] + feature_cfg["medium_frequency_labels"]

    df_vitals = vitals_df[vitals_df["op_id"].isin(preop_df["op_id"].unique())].copy()

    df_regular = df_vitals.loc[df_vitals["item_name"].isin(regular_labels), ["op_id", "item_name", "value"]]
    df_regular = df_regular.pivot_table(
        index="op_id",
        columns="item_name",
        values="value",
        aggfunc=["mean", "max", "min", entropy, kurtosis, skew, trend, energy],
    ).reset_index()
    df_regular.columns = [f"{feature}_{vital}" for feature, vital in df_regular.columns]
    df_regular.columns.values[0] = "op_id"

    df_cs_average = df_vitals.loc[
        df_vitals["item_name"].isin(feature_cfg["cross_sec_avg_labels"]),
        ["op_id", "item_name", "value"],
    ]
    df_cs_average = df_cs_average.pivot_table(
        index="op_id",
        columns="item_name",
        values="value",
        aggfunc=["mean"],
    ).reset_index()
    df_cs_average.columns = [f"{feature}_{vital}" for feature, vital in df_cs_average.columns]
    df_cs_average.columns.values[0] = "op_id"

    df_wt_adjusted = df_vitals.loc[
        df_vitals["item_name"].isin(feature_cfg["wt_adjusted_labels"]),
        ["op_id", "item_name", "value"],
    ]
    df_wt_adjusted = df_wt_adjusted.merge(preop_df[["op_id", "weight", "op_len"]], on="op_id", how="inner")
    df_wt_adjusted["value"] = df_wt_adjusted["value"] / (df_wt_adjusted["weight"] * df_wt_adjusted["op_len"])
    df_wt_adjusted = df_wt_adjusted[["op_id", "item_name", "value"]].pivot_table(
        index="op_id",
        columns="item_name",
        values="value",
        aggfunc=["sum"],
    ).reset_index()
    df_wt_adjusted.columns = [f"{feature}_{vital}" for feature, vital in df_wt_adjusted.columns]
    df_wt_adjusted.columns.values[0] = "op_id"

    df_time_adjusted = df_vitals.loc[
        df_vitals["item_name"].isin(feature_cfg["time_adjusted_labels"]),
        ["op_id", "item_name", "value"],
    ]
    df_time_adjusted = df_time_adjusted.merge(preop_df[["op_id", "op_len"]], on="op_id", how="inner")
    df_time_adjusted["value"] = df_time_adjusted["value"] / df_time_adjusted["op_len"]
    df_time_adjusted = df_time_adjusted[["op_id", "item_name", "value"]].pivot_table(
        index="op_id",
        columns="item_name",
        values="value",
        aggfunc=["sum"],
    ).reset_index()
    df_time_adjusted.columns = [f"{feature}_{vital}" for feature, vital in df_time_adjusted.columns]
    df_time_adjusted.columns.values[0] = "op_id"

    df_fluids_agg = df_vitals.loc[
        df_vitals["item_name"].isin(feature_cfg["fluids_agg_labels"]),
        ["op_id", "item_name", "value"],
    ]
    df_fluids_agg = df_fluids_agg.groupby("op_id")["value"].sum().reset_index()
    df_fluids_agg = df_fluids_agg.merge(preop_df[["op_id", "op_len"]], on="op_id", how="inner")
    df_fluids_agg["fluids_agg"] = df_fluids_agg.pop("value") / df_fluids_agg.pop("op_len")

    df_anesthetic = df_vitals.loc[
        df_vitals["item_name"].isin(feature_cfg["anesthetic_labels"]),
        ["op_id", "item_name", "value", "chart_time"],
    ]
    anesth_op_ids: list[int] = []
    anesth_means: list[float] = []
    for op_id, group in df_anesthetic.groupby("op_id"):
        end = group["chart_time"].max()
        start = group["chart_time"].min()
        times = pd.DataFrame({"chart_time": np.arange(start, end + 5, 5)})
        df_complete = pd.merge(times, group.loc[group["item_name"] == "etdes", ["chart_time", "value"]], on="chart_time", how="left")
        df_complete = pd.merge(df_complete, group.loc[group["item_name"] == "etsevo", ["chart_time", "value"]], on="chart_time", how="left")
        df_complete.ffill(inplace=True)
        df_complete.fillna(0, inplace=True)
        df_complete["equiv_MAC"] = (df_complete["value_x"] / 6.0) + (df_complete["value_y"] / 2.0)
        anesth_op_ids.append(op_id)
        anesth_means.append(float(df_complete["equiv_MAC"].mean()))
    df_anesthetic = pd.DataFrame({"op_id": anesth_op_ids, "equiv_MAC_totals": anesth_means})

    df_final = pd.DataFrame({"op_id": sorted(df_vitals["op_id"].unique())})
    for frame in [df_regular, df_cs_average, df_wt_adjusted, df_time_adjusted, df_fluids_agg, df_anesthetic]:
        df_final = df_final.merge(frame, on="op_id", how="left")
    return df_final
