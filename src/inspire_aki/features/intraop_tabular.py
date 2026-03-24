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


def _finite_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def safe_entropy(values: np.ndarray) -> float:
    arr = _finite_array(values)
    if arr.size == 0:
        return np.nan
    if np.allclose(arr, arr[0]):
        return 0.0
    if np.any(arr < 0):
        return np.nan
    total = arr.sum()
    if np.isclose(total, 0.0):
        return 0.0
    probs = arr / total
    probs = probs[probs > 0]
    if probs.size == 0:
        return 0.0
    return float(-(probs * np.log(probs)).sum())


def safe_kurtosis(values: np.ndarray) -> float:
    arr = _finite_array(values)
    if arr.size < 4 or np.allclose(arr, arr[0]):
        return 0.0
    return float(kurtosis(arr))


def safe_skew(values: np.ndarray) -> float:
    arr = _finite_array(values)
    if arr.size < 3 or np.allclose(arr, arr[0]):
        return 0.0
    return float(skew(arr))


def safe_trend(values: np.ndarray) -> float:
    arr = _finite_array(values)
    if arr.size < 2 or np.allclose(arr, arr[0]):
        return 0.0
    return trend(arr)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    output = pd.Series(np.nan, index=numerator.index, dtype=float)
    valid = denominator.notna() & (denominator > 0)
    output.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return output


def _build_anesthetic_feature(df_vitals: pd.DataFrame, anesthetic_labels: list[str]) -> pd.DataFrame:
    df_anesthetic = df_vitals.loc[
        df_vitals["item_name"].isin(anesthetic_labels),
        ["op_id", "item_name", "value", "chart_time"],
    ].copy()
    if df_anesthetic.empty:
        return pd.DataFrame(columns=["op_id", "equiv_MAC_totals"])

    wide = (
        df_anesthetic.pivot_table(
            index=["op_id", "chart_time"],
            columns="item_name",
            values="value",
            aggfunc="first",
        )
        .reindex(columns=anesthetic_labels)
        .sort_index()
    )

    rows: list[dict[str, float | int]] = []
    for op_id, group in wide.groupby(level=0, sort=False):
        op_frame = group.droplevel(0)
        full_index = np.arange(op_frame.index.min(), op_frame.index.max() + 5, 5)
        op_frame = op_frame.reindex(full_index).ffill().fillna(0.0)
        etdes = op_frame["etdes"] if "etdes" in op_frame.columns else 0.0
        etsevo = op_frame["etsevo"] if "etsevo" in op_frame.columns else 0.0
        equiv_mac = (etdes / 6.0) + (etsevo / 2.0)
        rows.append({"op_id": int(op_id), "equiv_MAC_totals": float(np.asarray(equiv_mac).mean())})
    return pd.DataFrame(rows)


def build_intraop_features(vitals_df: pd.DataFrame, preop_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    feature_cfg = config["features"]
    regular_labels = feature_cfg["high_frequency_labels"] + feature_cfg["medium_frequency_labels"]

    df_vitals = vitals_df[vitals_df["op_id"].isin(preop_df["op_id"].unique())].copy()

    df_regular = df_vitals.loc[df_vitals["item_name"].isin(regular_labels), ["op_id", "item_name", "value"]]
    df_regular = df_regular.pivot_table(
        index="op_id",
        columns="item_name",
        values="value",
        aggfunc=["mean", "max", "min", safe_entropy, safe_kurtosis, safe_skew, safe_trend, energy],
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
    df_wt_adjusted["value"] = _safe_divide(
        df_wt_adjusted["value"],
        df_wt_adjusted["weight"] * df_wt_adjusted["op_len"],
    )
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
    df_time_adjusted["value"] = _safe_divide(df_time_adjusted["value"], df_time_adjusted["op_len"])
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
    fluid_values = df_fluids_agg.pop("value")
    op_len = df_fluids_agg.pop("op_len")
    df_fluids_agg["fluids_agg"] = _safe_divide(fluid_values, op_len)

    df_anesthetic = _build_anesthetic_feature(df_vitals, feature_cfg["anesthetic_labels"])

    df_final = pd.DataFrame({"op_id": sorted(df_vitals["op_id"].unique())})
    for frame in [df_regular, df_cs_average, df_wt_adjusted, df_time_adjusted, df_fluids_agg, df_anesthetic]:
        df_final = df_final.merge(frame, on="op_id", how="left")
    df_final = df_final.replace([np.inf, -np.inf], np.nan)
    return df_final
