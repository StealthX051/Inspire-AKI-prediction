from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split

from inspire_aki.evaluation.leakage_checks import assert_group_integrity, assert_outer_test_coverage_once, assert_split_pair_has_no_overlap


EVALUATION_MODES = {"legacy_repeated_cv", "grouped_holdout", "grouped_nested_cv"}
MANIFEST_COLUMNS = [
    "evaluation_mode",
    "dataset_family",
    "op_id",
    "patient_id",
    "y_true",
    "outer_repeat_id",
    "outer_fold_id",
    "inner_repeat_id",
    "inner_fold_id",
    "split_scope",
    "split_name",
]


@dataclass(frozen=True)
class GroupedSplitBundle:
    manifest: pd.DataFrame
    overlap_audit: pd.DataFrame


@dataclass(frozen=True)
class EvaluationRun:
    run_id: int
    repeat_id: int
    fold_id: int


def _validate_split_inputs(df: pd.DataFrame, *, target: str, patient_col: str) -> pd.DataFrame:
    required = {"op_id", target, patient_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Grouped evaluation requires columns {missing}.")
    frame = df[["op_id", patient_col, target]].copy()
    frame = frame.rename(columns={patient_col: "patient_id", target: "y_true"})
    if frame["op_id"].duplicated().any():
        raise ValueError("Grouped evaluation requires unique op_id rows.")
    return frame


def patient_level_stratification_table(df: pd.DataFrame, *, target: str, patient_col: str = "patient_id") -> pd.DataFrame:
    frame = _validate_split_inputs(df, target=target, patient_col=patient_col)
    patient_table = (
        frame.groupby("patient_id", as_index=False)
        .agg(
            patient_target=("y_true", "max"),
            n_operations=("op_id", "size"),
            positive_operations=("y_true", "sum"),
        )
    )
    return patient_table


def _can_stratify(series: pd.Series, *, n_splits: int | None = None) -> bool:
    counts = series.value_counts(dropna=False)
    if len(counts) < 2:
        return False
    if n_splits is None:
        return (counts >= 2).all()
    return (counts >= n_splits).all()


def _ensure_binary_support(df: pd.DataFrame, *, target_col: str, split_name: str) -> None:
    if len(pd.unique(df[target_col])) < 2:
        raise ValueError(f"Insufficient class support for split '{split_name}'.")


def grouped_patient_train_test_split(
    df: pd.DataFrame,
    *,
    target: str,
    test_size: float,
    random_state: int,
    patient_col: str = "patient_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = _validate_split_inputs(df, target=target, patient_col=patient_col)
    patient_table = patient_level_stratification_table(df, target=target, patient_col=patient_col)
    stratify = patient_table["patient_target"] if _can_stratify(patient_table["patient_target"]) else None
    patient_train, patient_test = train_test_split(
        patient_table["patient_id"],
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    train_patients = set(patient_train.tolist())
    test_patients = set(patient_test.tolist())
    train_df = df[df[patient_col].isin(train_patients)].copy()
    test_df = df[df[patient_col].isin(test_patients)].copy()
    _ensure_binary_support(train_df, target_col=target, split_name="train")
    _ensure_binary_support(test_df, target_col=target, split_name="test")
    return train_df, test_df


def train_validation_split(
    df: pd.DataFrame,
    *,
    target: str,
    validation_fraction: float,
    random_state: int,
    evaluation_mode: str,
    patient_col: str = "patient_id",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if evaluation_mode != "legacy_repeated_cv" and patient_col in df.columns:
        return grouped_patient_train_test_split(
            df,
            target=target,
            test_size=validation_fraction,
            random_state=random_state,
            patient_col=patient_col,
        )
    train_df, val_df = train_test_split(
        df,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=df[target],
    )
    _ensure_binary_support(train_df, target_col=target, split_name="train")
    _ensure_binary_support(val_df, target_col=target, split_name="val")
    return train_df.copy(), val_df.copy()


def _outer_rows(
    frame: pd.DataFrame,
    *,
    evaluation_mode: str,
    dataset_family: str,
    outer_fold_id: int,
    train_op_ids: set[int],
    test_op_ids: set[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, op_ids in [("train", train_op_ids), ("test", test_op_ids)]:
        split_df = frame[frame["op_id"].isin(op_ids)]
        for record in split_df.itertuples(index=False):
            rows.append(
                {
                    "evaluation_mode": evaluation_mode,
                    "dataset_family": dataset_family,
                    "op_id": int(record.op_id),
                    "patient_id": record.patient_id,
                    "y_true": int(record.y_true),
                    "outer_repeat_id": 0,
                    "outer_fold_id": outer_fold_id,
                    "inner_repeat_id": pd.NA,
                    "inner_fold_id": pd.NA,
                    "split_scope": "outer",
                    "split_name": split_name,
                }
            )
    return rows


def _inner_rows(
    frame: pd.DataFrame,
    *,
    evaluation_mode: str,
    dataset_family: str,
    outer_fold_id: int,
    inner_fold_id: int,
    train_op_ids: set[int],
    val_op_ids: set[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_name, op_ids in [("train", train_op_ids), ("val", val_op_ids)]:
        split_df = frame[frame["op_id"].isin(op_ids)]
        for record in split_df.itertuples(index=False):
            rows.append(
                {
                    "evaluation_mode": evaluation_mode,
                    "dataset_family": dataset_family,
                    "op_id": int(record.op_id),
                    "patient_id": record.patient_id,
                    "y_true": int(record.y_true),
                    "outer_repeat_id": 0,
                    "outer_fold_id": outer_fold_id,
                    "inner_repeat_id": 0,
                    "inner_fold_id": inner_fold_id,
                    "split_scope": "inner",
                    "split_name": split_name,
                }
            )
    return rows


def _build_overlap_audit(manifest: pd.DataFrame) -> pd.DataFrame:
    return (
        manifest.groupby(
            ["evaluation_mode", "dataset_family", "split_scope", "outer_fold_id", "inner_fold_id", "split_name"],
            dropna=False,
        )
        .agg(
            n_operations=("op_id", "nunique"),
            n_patients=("patient_id", "nunique"),
            prevalence=("y_true", "mean"),
        )
        .reset_index()
        .sort_values(["split_scope", "outer_fold_id", "inner_fold_id", "split_name"], kind="stable")
        .reset_index(drop=True)
    )


def _finalize_grouped_manifest(manifest_rows: list[dict[str, Any]], *, evaluation_mode: str) -> GroupedSplitBundle:
    manifest = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    if manifest.empty:
        raise ValueError(f"{evaluation_mode} manifest is empty.")
    manifest["outer_repeat_id"] = manifest["outer_repeat_id"].astype("Int64")
    manifest["outer_fold_id"] = manifest["outer_fold_id"].astype("Int64")
    manifest["inner_repeat_id"] = manifest["inner_repeat_id"].astype("Int64")
    manifest["inner_fold_id"] = manifest["inner_fold_id"].astype("Int64")

    assert_group_integrity(manifest, split_scope="outer")
    if (manifest["split_scope"] == "inner").any():
        assert_group_integrity(manifest, split_scope="inner")
    assert_split_pair_has_no_overlap(manifest, split_scope="outer", split_name_left="train", split_name_right="test")
    if (manifest["split_scope"] == "inner").any():
        assert_split_pair_has_no_overlap(manifest, split_scope="inner", split_name_left="train", split_name_right="val")
    if evaluation_mode == "grouped_nested_cv":
        assert_outer_test_coverage_once(manifest)

    return GroupedSplitBundle(manifest=manifest, overlap_audit=_build_overlap_audit(manifest))


def manifest_keys(
    manifest: pd.DataFrame,
    *,
    repeat_id: int | None = None,
    fold_id: int | None = None,
) -> list[dict[str, int]]:
    if {"outer_repeat_id", "outer_fold_id", "split_scope"}.issubset(manifest.columns):
        scoped = manifest[manifest["split_scope"] == "outer"]
        repeat_column = "outer_repeat_id"
        fold_column = "outer_fold_id"
    else:
        scoped = manifest
        repeat_column = "repeat_id"
        fold_column = "fold_id"

    if repeat_id is not None:
        scoped = scoped[scoped[repeat_column] == repeat_id]
    if fold_id is not None:
        scoped = scoped[scoped[fold_column] == fold_id]

    key_columns = [repeat_column, fold_column]
    keys = (
        scoped[key_columns]
        .drop_duplicates()
        .sort_values(key_columns, kind="stable")
        .to_dict("records")
    )
    return [
        {
            "repeat_id": int(row[repeat_column]),
            "fold_id": int(row[fold_column]),
        }
        for row in keys
    ]


def evaluation_runs(
    manifest: pd.DataFrame,
    *,
    repeat_id: int | None = None,
    fold_id: int | None = None,
) -> list[EvaluationRun]:
    keys = manifest_keys(manifest, repeat_id=repeat_id, fold_id=fold_id)
    return [
        EvaluationRun(
            run_id=run_id,
            repeat_id=int(key["repeat_id"]),
            fold_id=int(key["fold_id"]),
        )
        for run_id, key in enumerate(keys)
    ]


def evaluation_run_for_run_id(manifest: pd.DataFrame, run_id: int) -> EvaluationRun:
    runs = evaluation_runs(manifest)
    if run_id < 0 or run_id >= len(runs):
        raise ValueError(f"run_id={run_id} is out of range for a manifest with {len(runs)} runs.")
    return runs[run_id]


def subset_generated_manifest(
    df: pd.DataFrame,
    manifest: pd.DataFrame,
    *,
    split_name: str,
    repeat_id: int | None = None,
    fold_id: int | None = None,
    run_id: int | None = None,
) -> pd.DataFrame:
    if run_id is not None:
        run = evaluation_run_for_run_id(manifest, run_id)
        repeat_id = run.repeat_id
        fold_id = run.fold_id
    if repeat_id is None or fold_id is None:
        raise ValueError("subset_generated_manifest requires either run_id or both repeat_id and fold_id.")

    if {"outer_repeat_id", "outer_fold_id", "split_scope"}.issubset(manifest.columns):
        subset = manifest[
            (manifest["split_scope"] == "outer")
            & (manifest["split_name"] == split_name)
            & (manifest["outer_repeat_id"] == repeat_id)
            & (manifest["outer_fold_id"] == fold_id)
        ]
    else:
        subset = manifest[
            (manifest["split_name"] == split_name)
            & (manifest["repeat_id"] == repeat_id)
            & (manifest["fold_id"] == fold_id)
        ]
    op_ids = subset["op_id"]
    return df[df["op_id"].isin(op_ids)].copy()


def build_grouped_holdout_manifest(
    df: pd.DataFrame,
    *,
    target: str,
    dataset_family: str,
    holdout_fraction: float,
    inner_n_splits: int,
    random_state: int,
    patient_col: str = "patient_id",
) -> GroupedSplitBundle:
    frame = _validate_split_inputs(df, target=target, patient_col=patient_col)
    outer_train_frame, outer_test_frame = grouped_patient_train_test_split(
        df,
        target=target,
        test_size=holdout_fraction,
        random_state=random_state,
        patient_col=patient_col,
    )
    train_op_ids = set(outer_train_frame["op_id"].tolist())
    test_op_ids = set(outer_test_frame["op_id"].tolist())

    manifest_rows = _outer_rows(
        frame,
        evaluation_mode="grouped_holdout",
        dataset_family=dataset_family,
        outer_fold_id=0,
        train_op_ids=train_op_ids,
        test_op_ids=test_op_ids,
    )

    outer_train = frame[frame["op_id"].isin(train_op_ids)].reset_index(drop=True)
    inner_splitter = _build_grouped_kfold_splitter(outer_train, n_splits=inner_n_splits, random_state=random_state)
    for inner_fold_id, (train_idx, val_idx) in enumerate(inner_splitter):
        inner_train_ids = set(outer_train.iloc[train_idx]["op_id"].tolist())
        inner_val_ids = set(outer_train.iloc[val_idx]["op_id"].tolist())
        manifest_rows.extend(
            _inner_rows(
                outer_train,
                evaluation_mode="grouped_holdout",
                dataset_family=dataset_family,
                outer_fold_id=0,
                inner_fold_id=inner_fold_id,
                train_op_ids=inner_train_ids,
                val_op_ids=inner_val_ids,
            )
        )

    return _finalize_grouped_manifest(manifest_rows, evaluation_mode="grouped_holdout")


def _build_grouped_kfold_splitter(df: pd.DataFrame, *, n_splits: int, random_state: int) -> list[tuple[Any, Any]]:
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(splitter.split(df[["op_id"]], df["y_true"], groups=df["patient_id"]))
    except ValueError:
        patient_table = (
            df.groupby("patient_id", as_index=False)
            .agg(patient_target=("y_true", "max"))
            .sort_values("patient_id", kind="stable")
            .reset_index(drop=True)
        )
        if not _can_stratify(patient_table["patient_target"], n_splits=n_splits):
            raise ValueError("Insufficient grouped class support for the requested number of folds.")
        patient_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        op_index = pd.Series(df.index.to_numpy(), index=df["op_id"])
        patient_to_op_ids = df.groupby("patient_id")["op_id"].apply(list).to_dict()
        splits: list[tuple[Any, Any]] = []
        for train_patient_idx, val_patient_idx in patient_splitter.split(patient_table["patient_id"], patient_table["patient_target"]):
            train_patients = set(patient_table.iloc[train_patient_idx]["patient_id"].tolist())
            val_patients = set(patient_table.iloc[val_patient_idx]["patient_id"].tolist())
            train_ops = [op_id for patient in train_patients for op_id in patient_to_op_ids[patient]]
            val_ops = [op_id for patient in val_patients for op_id in patient_to_op_ids[patient]]
            splits.append((op_index.loc[train_ops].to_numpy(), op_index.loc[val_ops].to_numpy()))
        return splits


def build_grouped_nested_cv_manifest(
    df: pd.DataFrame,
    *,
    target: str,
    dataset_family: str,
    outer_n_splits: int,
    inner_n_splits: int,
    random_state: int,
    patient_col: str = "patient_id",
) -> GroupedSplitBundle:
    frame = _validate_split_inputs(df, target=target, patient_col=patient_col)
    manifest_rows: list[dict[str, Any]] = []
    outer_splits = _build_grouped_kfold_splitter(frame, n_splits=outer_n_splits, random_state=random_state)

    for outer_fold_id, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        outer_train = frame.iloc[outer_train_idx].reset_index(drop=True)
        outer_test = frame.iloc[outer_test_idx].reset_index(drop=True)
        manifest_rows.extend(
            _outer_rows(
                frame,
                evaluation_mode="grouped_nested_cv",
                dataset_family=dataset_family,
                outer_fold_id=outer_fold_id,
                train_op_ids=set(outer_train["op_id"].tolist()),
                test_op_ids=set(outer_test["op_id"].tolist()),
            )
        )

        inner_splits = _build_grouped_kfold_splitter(outer_train, n_splits=inner_n_splits, random_state=random_state + outer_fold_id)
        for inner_fold_id, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):
            inner_train = outer_train.iloc[inner_train_idx]
            inner_val = outer_train.iloc[inner_val_idx]
            manifest_rows.extend(
                _inner_rows(
                    outer_train,
                    evaluation_mode="grouped_nested_cv",
                    dataset_family=dataset_family,
                    outer_fold_id=outer_fold_id,
                    inner_fold_id=inner_fold_id,
                    train_op_ids=set(inner_train["op_id"].tolist()),
                    val_op_ids=set(inner_val["op_id"].tolist()),
                )
            )

    return _finalize_grouped_manifest(manifest_rows, evaluation_mode="grouped_nested_cv")


def adapt_legacy_manifest(
    manifest: pd.DataFrame,
    df: pd.DataFrame,
    *,
    target: str,
    dataset_family: str,
    patient_col: str = "patient_id",
) -> pd.DataFrame:
    frame = _validate_split_inputs(df, target=target, patient_col=patient_col)
    lookup = frame.set_index("op_id")
    rows: list[dict[str, Any]] = []
    for record in manifest.itertuples(index=False):
        lookup_row = lookup.loc[int(record.op_id)]
        rows.append(
            {
                "evaluation_mode": "legacy_repeated_cv",
                "dataset_family": dataset_family,
                "op_id": int(record.op_id),
                "patient_id": lookup_row["patient_id"],
                "y_true": int(lookup_row["y_true"]),
                "outer_repeat_id": int(record.repeat_id),
                "outer_fold_id": int(record.fold_id),
                "inner_repeat_id": pd.NA,
                "inner_fold_id": pd.NA,
                "split_scope": "outer",
                "split_name": record.split_name,
            }
        )
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
