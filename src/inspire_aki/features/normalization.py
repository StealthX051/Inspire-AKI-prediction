from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from inspire_aki.runtime import build_stage_runtime_plan, thread_limited_context


def _fit_outlier_series_plan(
    series: pd.Series,
    *,
    column_name: str,
    outlier_cfg: dict[str, float],
) -> tuple[str, dict[str, float | bool]]:
    lower_1 = series.quantile(outlier_cfg["lower_extreme"])
    upper_1 = series.quantile(outlier_cfg["upper_extreme"])
    if pd.isna(lower_1) or pd.isna(upper_1):
        return column_name, {"enabled": False}
    lower_05 = series.quantile(outlier_cfg["lower_fill_low"])
    lower_5 = series.quantile(outlier_cfg["lower_fill_high"])
    upper_95 = series.quantile(outlier_cfg["upper_fill_low"])
    upper_995 = series.quantile(outlier_cfg["upper_fill_high"])
    return column_name, {
        "enabled": True,
        "lower_1": float(lower_1),
        "upper_1": float(upper_1),
        "lower_05": float(lower_05),
        "lower_5": float(lower_5),
        "upper_95": float(upper_95),
        "upper_995": float(upper_995),
    }


def fit_outlier_replacement_plan(
    df: pd.DataFrame,
    columns: list[str],
    config: dict,
) -> dict[str, dict[str, float | bool]]:
    if not columns:
        return {}
    outlier_cfg = config["features"]["outlier_quantiles"]
    runtime_plan = build_stage_runtime_plan(config, "preprocess_tabular")
    with thread_limited_context(runtime_plan.nested_blas_threads):
        results = Parallel(n_jobs=max(1, runtime_plan.tabular_column_workers), prefer="threads")(
            delayed(_fit_outlier_series_plan)(
                df[column],
                column_name=column,
                outlier_cfg=outlier_cfg,
            )
            for column in columns
        )
    return {column_name: plan for column_name, plan in results}


def _apply_outlier_series_plan(
    series: pd.Series,
    *,
    column_name: str,
    plan: dict[str, float | bool],
    seed: int,
) -> tuple[str, pd.Series]:
    output = series.copy()
    if not bool(plan.get("enabled", False)):
        return column_name, output
    rng = np.random.default_rng(seed)
    lower_1 = float(plan["lower_1"])
    upper_1 = float(plan["upper_1"])
    lower_05 = float(plan["lower_05"])
    lower_5 = float(plan["lower_5"])
    upper_95 = float(plan["upper_95"])
    upper_995 = float(plan["upper_995"])
    mask_lower = output < lower_1
    mask_upper = output > upper_1
    if int(mask_lower.sum()) > 0:
        output.loc[mask_lower] = rng.uniform(lower_05, lower_5, size=int(mask_lower.sum()))
    if int(mask_upper.sum()) > 0:
        output.loc[mask_upper] = rng.uniform(upper_95, upper_995, size=int(mask_upper.sum()))
    return column_name, output


def apply_outlier_replacement_plan(
    df: pd.DataFrame,
    *,
    columns: list[str],
    plan: dict[str, dict[str, float | bool]],
    config: dict,
    seed: int = 42,
) -> pd.DataFrame:
    output = df.copy()
    if not columns:
        return output
    runtime_plan = build_stage_runtime_plan(config, "preprocess_tabular")
    with thread_limited_context(runtime_plan.nested_blas_threads):
        results = Parallel(n_jobs=max(1, runtime_plan.tabular_column_workers), prefer="threads")(
            delayed(_apply_outlier_series_plan)(
                output[column],
                column_name=column,
                plan=plan.get(column, {"enabled": False}),
                seed=seed + (idx * 104_729),
            )
            for idx, column in enumerate(columns)
        )
    for column_name, series in results:
        output[column_name] = series
    return output


def replace_outliers(df: pd.DataFrame, ignore_cols: set[str], config: dict, seed: int = 42) -> pd.DataFrame:
    numeric_cols = [
        col
        for col in df.columns
        if col not in ignore_cols and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_cols:
        return df.copy()
    plan = fit_outlier_replacement_plan(df, numeric_cols, config)
    return apply_outlier_replacement_plan(df, columns=numeric_cols, plan=plan, config=config, seed=seed)


def fit_and_apply_standard_scaler(df: pd.DataFrame, columns: list[str], config: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    limit = build_stage_runtime_plan(config, "preprocess_tabular").tabular_column_workers if isinstance(config, dict) else 1
    with thread_limited_context(limit):
        scaler.fit(df[columns])
    output = df.copy()
    with thread_limited_context(limit):
        output[columns] = scaler.transform(df[columns])
    stats = pd.DataFrame({"mean": scaler.mean_, "var": scaler.var_}, index=columns)
    return output, stats
