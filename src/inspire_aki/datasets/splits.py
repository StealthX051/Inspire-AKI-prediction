from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def build_bootstrap_split_manifest(
    df: pd.DataFrame,
    *,
    target: str,
    dataset_regime: str,
    population_id: str,
    random_state: int,
    n_iterations: int,
    n_cv_folds: int = 5,
    use_bootstrapping: bool = True,
) -> pd.DataFrame:
    records: list[dict] = []
    if not use_bootstrapping:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=random_state, stratify=df[target])
        for split_name, split_df in [("train", train_df), ("test", test_df)]:
            for op_id in split_df["op_id"].tolist():
                records.append({
                    "op_id": op_id,
                    "dataset_regime": dataset_regime,
                    "population_id": population_id,
                    "repeat_id": 0,
                    "fold_id": 0,
                    "split_name": split_name,
                })
        return pd.DataFrame(records)

    n_repeats = max(1, n_iterations // n_cv_folds)
    for repeat_id in range(n_repeats):
        df_remainder = df.copy()
        folds: list[pd.DataFrame] = []
        for remaining_folds in range(n_cv_folds, 1, -1):
            rest_df, fold_df = train_test_split(
                df_remainder,
                test_size=(1.0 / remaining_folds),
                random_state=random_state + repeat_id,
                stratify=df_remainder[target],
            )
            folds.append(fold_df)
            df_remainder = rest_df
        folds.append(df_remainder)

        for fold_id, test_df in enumerate(folds):
            train_dfs = [fold for idx, fold in enumerate(folds) if idx != fold_id]
            train_df = pd.concat(train_dfs, ignore_index=False)
            for split_name, split_df in [("train", train_df), ("test", test_df)]:
                for op_id in split_df["op_id"].tolist():
                    records.append({
                        "op_id": op_id,
                        "dataset_regime": dataset_regime,
                        "population_id": population_id,
                        "repeat_id": repeat_id,
                        "fold_id": fold_id,
                        "split_name": split_name,
                    })
    return pd.DataFrame(records)


def build_hpo_split_manifest(
    df: pd.DataFrame,
    *,
    target: str,
    dataset_regime: str,
    population_id: str,
    random_state: int,
    holdout_fraction: float,
    validation_fraction_within_train: float,
) -> pd.DataFrame:
    records: list[dict] = []
    train_val_df, holdout_df = train_test_split(
        df,
        test_size=holdout_fraction,
        random_state=random_state,
        stratify=df[target],
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=validation_fraction_within_train,
        random_state=random_state,
        stratify=train_val_df[target],
    )
    for split_name, split_df in [("train", train_df), ("val", val_df), ("holdout", holdout_df)]:
        for op_id in split_df["op_id"].tolist():
            records.append({
                "op_id": op_id,
                "dataset_regime": dataset_regime,
                "population_id": population_id,
                "repeat_id": 0,
                "fold_id": 0,
                "split_name": split_name,
            })
    return pd.DataFrame(records)


def subset_from_manifest(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    repeat_id: int,
    fold_id: int,
    split_name: str,
) -> pd.DataFrame:
    op_ids = manifest[
        (manifest["repeat_id"] == repeat_id)
        & (manifest["fold_id"] == fold_id)
        & (manifest["split_name"] == split_name)
    ]["op_id"]
    return df[df["op_id"].isin(op_ids)].copy()
