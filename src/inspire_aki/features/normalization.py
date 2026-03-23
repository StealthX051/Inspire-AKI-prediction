from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def replace_outliers(df: pd.DataFrame, ignore_cols: set[str], config: dict, seed: int = 42) -> pd.DataFrame:
    outlier_cfg = config["features"]["outlier_quantiles"]
    rng = np.random.default_rng(seed)
    output = df.copy()
    for col in output.columns:
        if col in ignore_cols or not pd.api.types.is_numeric_dtype(output[col]):
            continue
        lower_1 = output[col].quantile(outlier_cfg["lower_extreme"])
        upper_1 = output[col].quantile(outlier_cfg["upper_extreme"])
        if pd.isna(lower_1) or pd.isna(upper_1):
            continue
        lower_05 = output[col].quantile(outlier_cfg["lower_fill_low"])
        lower_5 = output[col].quantile(outlier_cfg["lower_fill_high"])
        upper_95 = output[col].quantile(outlier_cfg["upper_fill_low"])
        upper_995 = output[col].quantile(outlier_cfg["upper_fill_high"])
        mask_lower = output[col] < lower_1
        mask_upper = output[col] > upper_1
        output.loc[mask_lower, col] = rng.uniform(lower_05, lower_5, size=int(mask_lower.sum()))
        output.loc[mask_upper, col] = rng.uniform(upper_95, upper_995, size=int(mask_upper.sum()))
    return output


def fit_and_apply_standard_scaler(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scaler = StandardScaler()
    scaler.fit(df[columns])
    output = df.copy()
    output[columns] = scaler.transform(df[columns])
    stats = pd.DataFrame({"mean": scaler.mean_, "var": scaler.var_}, index=columns)
    return output, stats
