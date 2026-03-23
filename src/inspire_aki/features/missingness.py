from __future__ import annotations

import pandas as pd
from sklearn.impute import KNNImputer


def impute_with_current_behavior(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    output = df.copy()
    nan_percentage = output.isna().mean() * 100
    fill_rates = (1.0 - output.isna().mean()).rename("fill_rate").reset_index().rename(columns={"index": "feature"})

    threshold = config["features"]["high_missing_threshold_pct"]
    high_missing_cols = nan_percentage[nan_percentage >= threshold].index.tolist()
    output[high_missing_cols] = output[high_missing_cols].fillna(-99)

    low_missing_cols = nan_percentage[(nan_percentage > 0) & (nan_percentage < threshold)].index.tolist()
    if low_missing_cols:
        imputer = KNNImputer(n_neighbors=config["features"]["knn_neighbors"])
        output[low_missing_cols] = imputer.fit_transform(output[low_missing_cols])
    return output, fill_rates
