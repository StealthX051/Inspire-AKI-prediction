from __future__ import annotations

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from inspire_aki.runtime import build_stage_runtime_plan, thread_limited_context


def _replace_outlier_series(
    series: pd.Series,
    *,
    column_name: str,
    outlier_cfg: dict[str, float],
    seed: int,
) -> tuple[str, pd.Series]:
    output = series.copy()
    rng = np.random.default_rng(seed)
    lower_1 = output.quantile(outlier_cfg["lower_extreme"])
    upper_1 = output.quantile(outlier_cfg["upper_extreme"])
    if pd.isna(lower_1) or pd.isna(upper_1):
        return column_name, output
    lower_05 = output.quantile(outlier_cfg["lower_fill_low"])
    lower_5 = output.quantile(outlier_cfg["lower_fill_high"])
    upper_95 = output.quantile(outlier_cfg["upper_fill_low"])
    upper_995 = output.quantile(outlier_cfg["upper_fill_high"])
    mask_lower = output < lower_1
    mask_upper = output > upper_1
    if int(mask_lower.sum()) > 0:
        output.loc[mask_lower] = rng.uniform(lower_05, lower_5, size=int(mask_lower.sum()))
    if int(mask_upper.sum()) > 0:
        output.loc[mask_upper] = rng.uniform(upper_95, upper_995, size=int(mask_upper.sum()))
    return column_name, output


def replace_outliers(df: pd.DataFrame, ignore_cols: set[str], config: dict, seed: int = 42) -> pd.DataFrame:
    outlier_cfg = config["features"]["outlier_quantiles"]
    output = df.copy()
    runtime_plan = build_stage_runtime_plan(config, "preprocess_tabular")
    numeric_cols = [
        col
        for col in output.columns
        if col not in ignore_cols and pd.api.types.is_numeric_dtype(output[col])
    ]
    if not numeric_cols:
        return output

    with thread_limited_context(runtime_plan.nested_blas_threads):
        results = Parallel(n_jobs=max(1, runtime_plan.tabular_column_workers), prefer="threads")(
            delayed(_replace_outlier_series)(
                output[col],
                column_name=col,
                outlier_cfg=outlier_cfg,
                seed=seed + (idx * 104_729),
            )
            for idx, col in enumerate(numeric_cols)
        )
    for col, series in results:
        output[col] = series
    return output


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
