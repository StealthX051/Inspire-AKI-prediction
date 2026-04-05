from __future__ import annotations

import pandas as pd
import pytest

from inspire_aki.datasets.splits import build_bootstrap_split_manifest, build_grouped_hpo_split_manifest
from inspire_aki.evaluation.leakage_checks import assert_group_integrity, assert_outer_test_coverage_once, assert_split_pair_has_no_overlap
from inspire_aki.evaluation.split_manager import (
    build_grouped_holdout_manifest,
    build_grouped_nested_cv_manifest,
    patient_level_stratification_table,
    train_validation_split,
)


def test_bootstrap_split_manifest_is_reproducible() -> None:
    df = pd.DataFrame(
        {
            "op_id": list(range(1, 13)),
            "aki_boolean": [0, 1] * 6,
        }
    )
    manifest_a = build_bootstrap_split_manifest(
        df,
        target="aki_boolean",
        dataset_regime="preop",
        population_id="preop",
        random_state=42,
        n_iterations=4,
        n_cv_folds=2,
        use_bootstrapping=True,
    )
    manifest_b = build_bootstrap_split_manifest(
        df,
        target="aki_boolean",
        dataset_regime="preop",
        population_id="preop",
        random_state=42,
        n_iterations=4,
        n_cv_folds=2,
        use_bootstrapping=True,
    )
    pd.testing.assert_frame_equal(manifest_a, manifest_b)


def _grouped_df() -> pd.DataFrame:
    rows = []
    patient_specs = {
        101: [0, 0],
        102: [0, 1],
        103: [0, 0],
        104: [1, 1],
        105: [0, 0],
        106: [1, 0],
        107: [0, 0],
        108: [1, 1],
        109: [0, 0],
        110: [1, 0],
    }
    op_id = 1
    for patient_id, labels in patient_specs.items():
        for label in labels:
            rows.append({"op_id": op_id, "patient_id": patient_id, "aki_boolean": label})
            op_id += 1
    return pd.DataFrame(rows)


def test_patient_level_stratification_table_uses_ever_positive_target() -> None:
    patient_table = patient_level_stratification_table(_grouped_df(), target="aki_boolean", patient_col="patient_id")

    assert set(patient_table.columns) == {"patient_id", "patient_target", "n_operations", "positive_operations"}
    assert int(patient_table.loc[patient_table["patient_id"] == 102, "patient_target"].iloc[0]) == 1
    assert int(patient_table.loc[patient_table["patient_id"] == 101, "patient_target"].iloc[0]) == 0


def test_grouped_holdout_manifest_has_zero_patient_overlap() -> None:
    bundle = build_grouped_holdout_manifest(
        _grouped_df(),
        target="aki_boolean",
        dataset_family="tabular_common",
        holdout_fraction=0.2,
        inner_n_splits=3,
        random_state=42,
    )
    manifest = bundle.manifest

    assert set(manifest["evaluation_mode"]) == {"grouped_holdout"}
    assert set(manifest["split_scope"]) == {"outer", "inner"}
    assert_group_integrity(manifest, split_scope="outer")
    assert_group_integrity(manifest, split_scope="inner")
    assert_split_pair_has_no_overlap(manifest, split_scope="outer", split_name_left="train", split_name_right="test")
    assert_split_pair_has_no_overlap(manifest, split_scope="inner", split_name_left="train", split_name_right="val")
    assert bundle.overlap_audit["n_patients"].gt(0).all()


def test_grouped_nested_manifest_assigns_each_operation_once_to_outer_test() -> None:
    bundle = build_grouped_nested_cv_manifest(
        _grouped_df(),
        target="aki_boolean",
        dataset_family="tabular_common",
        outer_n_splits=5,
        inner_n_splits=3,
        random_state=7,
    )
    manifest = bundle.manifest

    assert set(manifest["evaluation_mode"]) == {"grouped_nested_cv"}
    assert set(manifest["split_scope"]) == {"outer", "inner"}
    assert_outer_test_coverage_once(manifest)
    assert_group_integrity(manifest, split_scope="outer")
    assert_group_integrity(manifest, split_scope="inner")


def test_grouped_nested_manifest_assigns_each_outer_train_operation_once_to_inner_validation() -> None:
    bundle = build_grouped_nested_cv_manifest(
        _grouped_df(),
        target="aki_boolean",
        dataset_family="tabular_common",
        outer_n_splits=5,
        inner_n_splits=3,
        random_state=7,
    )
    manifest = bundle.manifest
    inner_val = manifest[(manifest["split_scope"] == "inner") & (manifest["split_name"] == "val")].copy()

    op_counts = inner_val.groupby(["outer_fold_id", "op_id"]).size()
    patient_fold_counts = inner_val.groupby(["outer_fold_id", "patient_id"])["inner_fold_id"].nunique()

    assert op_counts.eq(1).all()
    assert patient_fold_counts.eq(1).all()


def test_grouped_leakage_checks_fail_closed_on_patient_overlap() -> None:
    manifest = pd.DataFrame(
        [
            {
                "evaluation_mode": "grouped_holdout",
                "dataset_family": "tabular_common",
                "op_id": 1,
                "patient_id": 10,
                "y_true": 0,
                "outer_repeat_id": 0,
                "outer_fold_id": 0,
                "inner_repeat_id": pd.NA,
                "inner_fold_id": pd.NA,
                "split_scope": "outer",
                "split_name": "train",
            },
            {
                "evaluation_mode": "grouped_holdout",
                "dataset_family": "tabular_common",
                "op_id": 2,
                "patient_id": 10,
                "y_true": 1,
                "outer_repeat_id": 0,
                "outer_fold_id": 0,
                "inner_repeat_id": pd.NA,
                "inner_fold_id": pd.NA,
                "split_scope": "outer",
                "split_name": "test",
            },
        ]
    )

    with pytest.raises(ValueError, match="Patient grouping violated"):
        assert_group_integrity(manifest, split_scope="outer")


def test_grouped_hpo_split_manifest_has_zero_patient_overlap() -> None:
    df = _grouped_df()
    manifest = build_grouped_hpo_split_manifest(
        df,
        target="aki_boolean",
        dataset_regime="combined",
        population_id="combined",
        random_state=42,
        holdout_fraction=0.2,
        validation_fraction_within_train=0.25,
    )

    lookup = df.set_index("op_id")["patient_id"]
    split_patients = {
        split_name: {int(lookup.loc[op_id]) for op_id in split_df["op_id"].tolist()}
        for split_name, split_df in manifest.groupby("split_name", sort=False)
    }

    assert set(manifest["split_name"]) == {"train", "val", "holdout"}
    assert not split_patients["train"] & split_patients["val"]
    assert not split_patients["train"] & split_patients["holdout"]
    assert not split_patients["val"] & split_patients["holdout"]


def test_train_validation_split_groups_patients_for_grouped_holdout() -> None:
    train_df, val_df = train_validation_split(
        _grouped_df(),
        target="aki_boolean",
        validation_fraction=0.25,
        random_state=42,
        evaluation_mode="grouped_holdout",
    )

    assert set(train_df["patient_id"]) and set(val_df["patient_id"])
    assert not set(train_df["patient_id"]) & set(val_df["patient_id"])
    assert train_df["aki_boolean"].nunique() == 2
    assert val_df["aki_boolean"].nunique() == 2


def test_train_validation_split_fails_fast_for_infeasible_grouped_support() -> None:
    df = pd.DataFrame(
        [
            {"op_id": 1, "patient_id": 101, "aki_boolean": 1},
            {"op_id": 2, "patient_id": 102, "aki_boolean": 0},
            {"op_id": 3, "patient_id": 103, "aki_boolean": 0},
            {"op_id": 4, "patient_id": 104, "aki_boolean": 0},
        ]
    )

    with pytest.raises(ValueError, match="Insufficient class support"):
        train_validation_split(
            df,
            target="aki_boolean",
            validation_fraction=0.5,
            random_state=42,
            evaluation_mode="grouped_holdout",
        )
