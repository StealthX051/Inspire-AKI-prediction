from __future__ import annotations

import numpy as np
import pandas as pd

from inspire_aki.features.missingness import impute_with_current_behavior
from inspire_aki.features.normalization import fit_and_apply_standard_scaler, replace_outliers


def build_tabular_datasets(preop_df: pd.DataFrame, intraop_df: pd.DataFrame, config: dict) -> dict[str, pd.DataFrame]:
    missing_sources = []
    if "op_id" not in preop_df.columns:
        missing_sources.append("preop_df")
    if "op_id" not in intraop_df.columns:
        missing_sources.append("intraop_df")
    if missing_sources:
        raise ValueError(
            "build_tabular_datasets requires an 'op_id' column in both upstream inputs. "
            f"Missing in: {', '.join(missing_sources)}."
        )

    df = pd.merge(preop_df, intraop_df, on="op_id", how="inner")

    if df.empty:
        raise ValueError("Merged tabular dataset is empty.")

    df = df.replace([np.inf, -np.inf], np.nan)
    cols_to_pop = [
        "postop_creatinine",
        "subject_id",
        "opstart_time",
        "opend_time",
        "inhosp_death_time",
        "allcause_death_time",
    ]
    existing_cols = [col for col in cols_to_pop if col in df.columns]
    df = df.drop(columns=existing_cols)

    int_columns = df.select_dtypes(include=["int"]).columns
    df[int_columns] = df[int_columns].astype(float)

    ignore_cols = set(config["features"]["base_ignore_cols"])
    for col in df.columns:
        if ("department" in col) or ("_isna" in col) or ("aki" in col):
            ignore_cols.add(col)

    df_unnormalized = replace_outliers(df, ignore_cols, config)
    cols_to_norm = [col for col in df_unnormalized.columns if col not in ignore_cols and pd.api.types.is_numeric_dtype(df_unnormalized[col])]
    df_normalized, stats = fit_and_apply_standard_scaler(df_unnormalized, cols_to_norm, config)
    df_imputed, fill_rates = impute_with_current_behavior(df_normalized, config)

    preop_cols_final = [col for col in df_imputed.columns if col in preop_df.columns and col != "subject_id"]
    intraop_cols_final = [col for col in df_imputed.columns if col in intraop_df.columns]
    if "op_id" not in preop_cols_final:
        preop_cols_final.insert(0, "op_id")
    if "op_id" not in intraop_cols_final:
        intraop_cols_final.insert(0, "op_id")

    return {
        "combined": df_imputed,
        "preop": df_imputed[list(dict.fromkeys(preop_cols_final))].copy(),
        "intraop": df_imputed[list(dict.fromkeys(intraop_cols_final))].copy(),
        "combined_unnormalized": df_unnormalized,
        "normalization_stats": stats.reset_index().rename(columns={"index": "feature"}),
        "fill_rates": fill_rates,
    }
