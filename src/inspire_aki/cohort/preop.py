from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from inspire_aki.cohort.audit import record_count
from inspire_aki.cohort.filters import apply_preop_filters
from inspire_aki.io.csv import read_csv_optimized
from inspire_aki.runtime import build_stage_runtime_plan, thread_limited_context


def _extract_preop_item_feature(
    preop_base: pd.DataFrame,
    item_df: pd.DataFrame,
    *,
    tolerance: float,
    feature_name: str,
) -> pd.DataFrame:
    merged = pd.merge_asof(
        preop_base.sort_values("opstart_time"),
        item_df.sort_values("chart_time"),
        left_on="opstart_time",
        right_on="chart_time",
        by="subject_id",
        tolerance=tolerance,
        suffixes=("", "_"),
    )
    column_map = {"value": feature_name}
    keep_cols = [col for col in ["op_id", "value"] if col in merged.columns]
    return merged[keep_cols].rename(columns=column_map)


def load_preop_sources(raw_inspire_dir: Path, config: dict, csv_read_threads: int) -> dict[str, pd.DataFrame]:
    jobs = {
        "labs": {
            "path": raw_inspire_dir / "labs.csv",
            "usecols": ["subject_id", "chart_time", "item_name", "value"],
        },
        "ward_vitals": {
            "path": raw_inspire_dir / "ward_vitals.csv",
            "usecols": ["subject_id", "chart_time", "item_name", "value"],
        },
        "operations": {
            "path": raw_inspire_dir / "operations.csv",
            "usecols": [
                "op_id",
                "subject_id",
                "age",
                "sex",
                "height",
                "weight",
                "asa",
                "emop",
                "opstart_time",
                "opend_time",
                "inhosp_death_time",
                "allcause_death_time",
                "orin_time",
                "orout_time",
                "department",
                "antype",
                "icd10_pcs",
            ],
        },
        "diagnosis": {
            "path": raw_inspire_dir / "diagnosis.csv",
            "usecols": ["subject_id", "chart_time", "icd10_cm"],
        },
    }
    outputs: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max(1, csv_read_threads)) as executor:
        futures = {
            key: executor.submit(
                read_csv_optimized,
                job["path"],
                config=config,
                usecols=job["usecols"],
                large=True,
            )
            for key, job in jobs.items()
        }
        for key, future in futures.items():
            outputs[key] = future.result()
    return outputs


def build_preop_features(config: dict, raw_inspire_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cfg = config["features"]
    cohort_cfg = config["cohort"]
    runtime_plan = build_stage_runtime_plan(config, "preprocess_preop")
    sources = load_preop_sources(raw_inspire_dir, config, runtime_plan.csv_read_threads)

    df_labs = sources["labs"]
    df_labs["chart_time"] = df_labs["chart_time"].astype(float)
    df_ward = sources["ward_vitals"]
    df_ward["chart_time"] = df_ward["chart_time"].astype(float)
    df_ops = sources["operations"]
    df_diags = sources["diagnosis"]

    audit: list[dict] = []
    record_count(audit, "raw_operations", df_ops)

    desired_columns = [
        "op_id",
        "subject_id",
        "age",
        "sex",
        "height",
        "weight",
        "asa",
        "emop",
        "opstart_time",
        "opend_time",
        "inhosp_death_time",
        "allcause_death_time",
        "orin_time",
        "orout_time",
    ]
    df_preop = df_ops[desired_columns].copy()
    for column in ["opstart_time", "opend_time", "orin_time", "orout_time"]:
        df_preop[column] = df_preop[column].astype(float)

    valid_mask = df_preop["height"].notna() & df_preop["weight"].notna()
    df_preop["BSA"] = np.nan
    df_preop["BMI"] = np.nan
    df_preop.loc[valid_mask, "BSA"] = np.sqrt((df_preop.loc[valid_mask, "height"] * df_preop.loc[valid_mask, "weight"]) / 3600.0)
    df_preop.loc[valid_mask, "BMI"] = df_preop.loc[valid_mask, "weight"] / ((df_preop.loc[valid_mask, "height"] / 100.0) ** 2)

    valid_mask = df_preop["orin_time"].notna() & df_preop["orout_time"].notna()
    df_preop["booking_case_length"] = np.nan
    df_preop.loc[valid_mask, "booking_case_length"] = df_preop.loc[valid_mask, "orout_time"] - df_preop.loc[valid_mask, "orin_time"]
    df_preop = df_preop.drop(columns=["orin_time", "orout_time"])

    df_diags_cvd = df_diags[df_diags["icd10_cm"].str.startswith(cohort_cfg["cardiovascular_prefix"], na=False)]
    merged = pd.merge(
        df_preop[["op_id", "subject_id", "opstart_time"]],
        df_diags_cvd[["subject_id", "chart_time"]],
        on="subject_id",
        how="inner",
    )
    merged = merged[merged["chart_time"] < merged["opstart_time"]]
    num_card_events = merged.groupby("op_id").size().reset_index(name="num_card_events")
    df_preop = pd.merge(df_preop, num_card_events, on="op_id", how="left")
    df_preop["num_card_events"] = df_preop["num_card_events"].fillna(0).astype(int)

    df_preop, df_ops_for_merge, audit = apply_preop_filters(df_preop, df_ops.copy(), config, audit)

    cols_to_keep = ["op_id", "subject_id", "antype"]
    cols_to_keep.extend(col for col in df_ops_for_merge.columns if "department_" in col)
    df_preop = pd.merge(df_preop, df_ops_for_merge[cols_to_keep], on=["op_id", "subject_id"], how="inner")
    record_count(audit, "after_antype_department_merge", df_preop)

    include_prefixes = tuple(cohort_cfg.get("include_icd10_prefixes", []))
    exclude_prefixes = tuple(cohort_cfg.get("exclude_icd10_prefixes", []))
    if include_prefixes:
        include_mask = df_ops["icd10_pcs"].astype(str).str.startswith(include_prefixes)
        included_ops = df_ops.loc[include_mask, "op_id"]
        df_preop = df_preop[df_preop["op_id"].isin(included_ops)]
    if exclude_prefixes:
        exclude_mask = df_ops["icd10_pcs"].astype(str).str.startswith(exclude_prefixes)
        ops_to_exclude = df_ops.loc[exclude_mask, "op_id"]
        df_preop = df_preop[~df_preop["op_id"].isin(ops_to_exclude)]
    record_count(audit, "after_prefix_exclusions", df_preop)

    tolerance = cohort_cfg["preop_window_days"] * 24 * 60
    preop_base = df_preop[["op_id", "subject_id", "opstart_time"]]
    feature_jobs = [
        ("labs", item_name, f"preop_{item_name}")
        for item_name in feature_cfg["preop_lab_items"]
    ] + [
        ("ward_vitals", item_name, f"ward_{item_name}")
        for item_name in feature_cfg["ward_items"]
    ]

    extracted_features: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=max(1, runtime_plan.preop_feature_workers)) as executor:
        future_to_feature = {}
        for source_name, item_name, feature_name in feature_jobs:
            source_df = df_labs if source_name == "labs" else df_ward
            item_df = source_df.loc[source_df["item_name"] == item_name, ["subject_id", "chart_time", "value"]].copy()
            future = executor.submit(
                _extract_preop_item_feature,
                preop_base,
                item_df,
                tolerance=tolerance,
                feature_name=feature_name,
            )
            future_to_feature[feature_name] = future
        for _, _, feature_name in feature_jobs:
            extracted_features.append(future_to_feature[feature_name].result())

    with thread_limited_context(runtime_plan.nested_blas_threads):
        for feature_frame in extracted_features:
            df_preop = df_preop.merge(feature_frame, on="op_id", how="left")

    return df_preop.reset_index(drop=True), pd.DataFrame(audit)
