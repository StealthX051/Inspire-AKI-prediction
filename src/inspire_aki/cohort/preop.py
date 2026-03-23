from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inspire_aki.cohort.audit import record_count
from inspire_aki.cohort.filters import apply_preop_filters


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_preop_sources(raw_inspire_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "labs": _read_csv(raw_inspire_dir / "labs.csv"),
        "ward_vitals": _read_csv(raw_inspire_dir / "ward_vitals.csv"),
        "operations": _read_csv(raw_inspire_dir / "operations.csv"),
        "diagnosis": _read_csv(raw_inspire_dir / "diagnosis.csv"),
    }


def build_preop_features(config: dict, raw_inspire_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cfg = config["features"]
    cohort_cfg = config["cohort"]
    sources = load_preop_sources(raw_inspire_dir)

    df_labs = sources["labs"].copy()
    df_labs["chart_time"] = df_labs["chart_time"].astype(float)
    df_ward = sources["ward_vitals"].copy()
    df_ward["chart_time"] = df_ward["chart_time"].astype(float)
    df_ops = sources["operations"].copy()
    df_diags = sources["diagnosis"].copy()

    audit: list[dict] = []
    df_preop = df_ops.copy()
    record_count(audit, "raw_operations", df_preop)

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
    df_preop = df_preop[desired_columns]
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

    tolerance = cohort_cfg["preop_window_days"] * 24 * 60
    for item_name in feature_cfg["preop_lab_items"]:
        df_preop = pd.merge_asof(
            df_preop.sort_values("opstart_time"),
            df_labs.loc[df_labs["item_name"] == item_name].sort_values("chart_time"),
            left_on="opstart_time",
            right_on="chart_time",
            by="subject_id",
            tolerance=tolerance,
            suffixes=("", "_"),
        )
        df_preop.drop(columns=["chart_time", "item_name"], inplace=True)
        df_preop.rename(columns={"value": f"preop_{item_name}"}, inplace=True)

    for item_name in feature_cfg["ward_items"]:
        df_preop = pd.merge_asof(
            df_preop.sort_values("opstart_time"),
            df_ward.loc[df_ward["item_name"] == item_name].sort_values("chart_time"),
            left_on="opstart_time",
            right_on="chart_time",
            by="subject_id",
            tolerance=tolerance,
            suffixes=("", "_"),
        )
        df_preop.drop(columns=["chart_time", "item_name"], inplace=True)
        df_preop.rename(columns={"value": f"ward_{item_name}"}, inplace=True)

    mask = df_ops["icd10_pcs"].astype(str).str.startswith(tuple(cohort_cfg["exclude_icd10_prefixes"]))
    ops_to_exclude = df_ops.loc[mask, "op_id"]
    df_preop = df_preop[~df_preop["op_id"].isin(ops_to_exclude)]
    record_count(audit, "after_prefix_exclusions", df_preop)

    return df_preop.reset_index(drop=True), pd.DataFrame(audit)
