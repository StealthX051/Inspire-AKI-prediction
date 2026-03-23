from __future__ import annotations

import pandas as pd

from inspire_aki.cohort.labels import derive_aki_labels
from inspire_aki.config import load_config


def test_dialysis_positive_case_is_labeled(synthetic_config) -> None:
    config = load_config(synthetic_config)
    raw_dir = synthetic_config.parent / "raw"
    preop_df = pd.read_csv(synthetic_config.parent / "artifacts" / "features" / "preop" / "preop_features.csv") if (synthetic_config.parent / "artifacts" / "features" / "preop" / "preop_features.csv").exists() else None
    if preop_df is None:
        from inspire_aki.cohort.preop import build_preop_features
        from inspire_aki.datasets.tabular import build_tabular_datasets
        from inspire_aki.features.intraop_tabular import build_intraop_features

        preop_df, _ = build_preop_features(config, raw_dir)
        intraop_df = build_intraop_features(pd.read_csv(raw_dir / "vitals.csv"), preop_df, config)
        combined_df = build_tabular_datasets(preop_df, intraop_df, config)["combined"]
    else:
        combined_df = pd.read_csv(synthetic_config.parent / "artifacts" / "datasets" / "tabular" / "tabular_combined.csv")

    labels_df, _ = derive_aki_labels(
        config=config,
        raw_inspire_dir=raw_dir,
        preop_df=preop_df,
        tabular_combined_df=combined_df,
    )
    dialysis_op_id = 12
    assert int(labels_df.loc[labels_df["op_id"] == dialysis_op_id, "aki_boolean"].iloc[0]) == 1

