from __future__ import annotations

import pandas as pd


def assert_no_identifier_overlap(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    *,
    column: str,
    left_name: str,
    right_name: str,
) -> None:
    overlap = set(left_df[column].dropna().tolist()) & set(right_df[column].dropna().tolist())
    if overlap:
        preview = sorted(list(overlap))[:5]
        raise ValueError(
            f"{column} overlap detected between {left_name} and {right_name}; "
            f"found {len(overlap)} overlapping values, first few: {preview}"
        )


def assert_split_pair_has_no_overlap(
    manifest: pd.DataFrame,
    *,
    split_scope: str,
    split_name_left: str,
    split_name_right: str,
    patient_col: str = "patient_id",
    op_col: str = "op_id",
) -> None:
    scoped = manifest[manifest["split_scope"] == split_scope]
    grouping_cols = [col for col in ["outer_repeat_id", "outer_fold_id", "inner_repeat_id", "inner_fold_id"] if col in scoped.columns]
    keys = scoped[grouping_cols].drop_duplicates() if grouping_cols else pd.DataFrame([{}])
    for key in keys.to_dict("records"):
        subset = scoped.copy()
        label_parts = [split_scope]
        for column, value in key.items():
            subset = subset[subset[column] == value]
            label_parts.append(f"{column}={value}")
        left_df = subset[subset["split_name"] == split_name_left]
        right_df = subset[subset["split_name"] == split_name_right]
        if left_df.empty or right_df.empty:
            continue
        label = ", ".join(label_parts)
        assert_no_identifier_overlap(left_df, right_df, column=patient_col, left_name=f"{label}:{split_name_left}", right_name=f"{label}:{split_name_right}")
        assert_no_identifier_overlap(left_df, right_df, column=op_col, left_name=f"{label}:{split_name_left}", right_name=f"{label}:{split_name_right}")


def assert_group_integrity(
    manifest: pd.DataFrame,
    *,
    split_scope: str,
    patient_col: str = "patient_id",
) -> None:
    scoped = manifest[manifest["split_scope"] == split_scope]
    grouping_cols = [col for col in ["outer_repeat_id", "outer_fold_id", "inner_repeat_id", "inner_fold_id"] if col in scoped.columns]
    if not grouping_cols:
        grouping_cols = ["split_scope"]
        scoped = scoped.assign(split_scope=split_scope)
    counts = (
        scoped.groupby(grouping_cols + [patient_col], dropna=False)["split_name"]
        .nunique()
        .reset_index(name="n_split_names")
    )
    offenders = counts[counts["n_split_names"] > 1]
    if not offenders.empty:
        raise ValueError(
            f"Patient grouping violated for split_scope={split_scope}; "
            f"found {len(offenders)} patient/fold combinations assigned to multiple split names."
        )


def assert_outer_test_coverage_once(manifest: pd.DataFrame, *, op_col: str = "op_id") -> None:
    outer_test = manifest[(manifest["split_scope"] == "outer") & (manifest["split_name"] == "test")]
    counts = outer_test.groupby(op_col).size()
    if counts.empty:
        raise ValueError("No outer test assignments found in manifest.")
    if not (counts == 1).all():
        raise ValueError("Each operation must appear exactly once in the outer test role.")
