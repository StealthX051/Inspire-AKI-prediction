from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from inspire_aki.evaluation.split_manager import grouped_patient_train_test_split

_LEGACY_SPLIT_COLUMNS = ["op_id", "repeat_id", "fold_id", "split_name"]
_HPO_SPLIT_COLUMNS = ["op_id", "dataset_regime", "population_id", "repeat_id", "fold_id", "split_name"]


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


def build_grouped_hpo_split_manifest(
    df: pd.DataFrame,
    *,
    target: str,
    dataset_regime: str,
    population_id: str,
    random_state: int,
    holdout_fraction: float,
    validation_fraction_within_train: float,
    patient_col: str = "patient_id",
) -> pd.DataFrame:
    records: list[dict] = []
    train_val_df, holdout_df = grouped_patient_train_test_split(
        df,
        target=target,
        test_size=holdout_fraction,
        random_state=random_state,
        patient_col=patient_col,
    )
    train_df, val_df = grouped_patient_train_test_split(
        train_val_df,
        target=target,
        test_size=validation_fraction_within_train,
        random_state=random_state,
        patient_col=patient_col,
    )
    for split_name, split_df in [("train", train_df), ("val", val_df), ("holdout", holdout_df)]:
        for op_id in split_df["op_id"].tolist():
            records.append(
                {
                    "op_id": op_id,
                    "dataset_regime": dataset_regime,
                    "population_id": population_id,
                    "repeat_id": 0,
                    "fold_id": 0,
                    "split_name": split_name,
                }
            )
    return pd.DataFrame(records)


def grouped_manifest_to_training_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if "split_scope" not in manifest.columns and set(_LEGACY_SPLIT_COLUMNS).issubset(manifest.columns):
        adapted = manifest[_LEGACY_SPLIT_COLUMNS].copy()
        return adapted.sort_values(["repeat_id", "fold_id", "split_name", "op_id"], kind="stable").reset_index(drop=True)

    required = {"op_id", "outer_repeat_id", "outer_fold_id", "split_scope", "split_name"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Grouped training manifest adaptation requires columns {missing}.")

    outer = manifest[manifest["split_scope"] == "outer"].copy()
    if outer.empty:
        raise ValueError("Grouped training manifest adaptation requires outer train/test rows.")

    adapted = outer[["op_id", "outer_repeat_id", "outer_fold_id", "split_name"]].rename(
        columns={"outer_repeat_id": "repeat_id", "outer_fold_id": "fold_id"}
    )
    adapted["repeat_id"] = adapted["repeat_id"].astype(int)
    adapted["fold_id"] = adapted["fold_id"].astype(int)
    return adapted.sort_values(["repeat_id", "fold_id", "split_name", "op_id"], kind="stable").reset_index(drop=True)


def grouped_manifest_to_hpo_manifest(
    manifest: pd.DataFrame,
    *,
    dataset_regime: str,
    population_id: str,
) -> pd.DataFrame:
    if "split_scope" not in manifest.columns and set(_LEGACY_SPLIT_COLUMNS).issubset(manifest.columns):
        split_names = set(manifest["split_name"].astype(str))
        if {"train", "val", "holdout"}.issubset(split_names):
            adapted = manifest.copy()
            if "dataset_regime" not in adapted.columns:
                adapted["dataset_regime"] = dataset_regime
            if "population_id" not in adapted.columns:
                adapted["population_id"] = population_id
            return adapted[_HPO_SPLIT_COLUMNS].sort_values(["repeat_id", "fold_id", "split_name", "op_id"], kind="stable").reset_index(drop=True)
        raise ValueError(
            "Grouped HPO manifest adaptation requires grouped manifests with inner train/val rows or a legacy "
            "HPO manifest with train/val/holdout splits."
        )

    required = {
        "op_id",
        "outer_repeat_id",
        "outer_fold_id",
        "inner_repeat_id",
        "inner_fold_id",
        "split_scope",
        "split_name",
    }
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Grouped HPO manifest adaptation requires columns {missing}.")

    outer = manifest[manifest["split_scope"] == "outer"].copy()
    inner = manifest[manifest["split_scope"] == "inner"].copy()
    if outer.empty or inner.empty:
        raise ValueError("Grouped HPO manifest adaptation requires both outer and inner split rows.")

    outer_keys = (
        outer[["outer_repeat_id", "outer_fold_id"]]
        .drop_duplicates()
        .sort_values(["outer_repeat_id", "outer_fold_id"], kind="stable")
        .reset_index(drop=True)
    )
    selected_outer = outer_keys.iloc[0]
    outer_repeat_id = int(selected_outer["outer_repeat_id"])
    outer_fold_id = int(selected_outer["outer_fold_id"])

    outer_selected = outer[
        (outer["outer_repeat_id"] == outer_repeat_id)
        & (outer["outer_fold_id"] == outer_fold_id)
    ]
    inner_candidates = inner[
        (inner["outer_repeat_id"] == outer_repeat_id)
        & (inner["outer_fold_id"] == outer_fold_id)
    ]
    inner_keys = (
        inner_candidates[["inner_repeat_id", "inner_fold_id"]]
        .drop_duplicates()
        .sort_values(["inner_repeat_id", "inner_fold_id"], kind="stable")
        .reset_index(drop=True)
    )
    if inner_keys.empty:
        raise ValueError(
            f"Grouped HPO manifest adaptation found no inner folds for outer_repeat_id={outer_repeat_id}, "
            f"outer_fold_id={outer_fold_id}."
        )
    selected_inner = inner_keys.iloc[0]
    inner_repeat_id = int(selected_inner["inner_repeat_id"])
    inner_fold_id = int(selected_inner["inner_fold_id"])
    inner_selected = inner_candidates[
        (inner_candidates["inner_repeat_id"] == inner_repeat_id)
        & (inner_candidates["inner_fold_id"] == inner_fold_id)
    ]

    split_frames = {
        "train": inner_selected[inner_selected["split_name"] == "train"],
        "val": inner_selected[inner_selected["split_name"] == "val"],
        "holdout": outer_selected[outer_selected["split_name"] == "test"],
    }
    records: list[dict[str, object]] = []
    for split_name, split_df in split_frames.items():
        if split_df.empty:
            raise ValueError(f"Grouped HPO manifest adaptation requires a non-empty '{split_name}' split.")
        for op_id in split_df["op_id"].drop_duplicates().tolist():
            records.append(
                {
                    "op_id": int(op_id),
                    "dataset_regime": dataset_regime,
                    "population_id": population_id,
                    "repeat_id": 0,
                    "fold_id": 0,
                    "split_name": split_name,
                }
            )
    adapted = pd.DataFrame(records, columns=_HPO_SPLIT_COLUMNS)
    return adapted.sort_values(["repeat_id", "fold_id", "split_name", "op_id"], kind="stable").reset_index(drop=True)


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
