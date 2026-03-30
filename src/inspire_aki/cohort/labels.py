from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from inspire_aki.cohort.audit import audit_frame, record_count
from inspire_aki.config import active_outcome_config, active_outcome_key
from inspire_aki.io.csv import read_csv_optimized
from inspire_aki.runtime import build_stage_runtime_plan


def _base_label_frame(preop_df: pd.DataFrame, tabular_combined_df: pd.DataFrame) -> pd.DataFrame:
    base = preop_df[["op_id", "subject_id", "opend_time"]].copy()
    base = base.merge(tabular_combined_df[["op_id"]].drop_duplicates(), on="op_id", how="inner")
    base["patient_id"] = base["subject_id"]
    return base.drop_duplicates(subset="op_id", keep="last").reset_index(drop=True)


def _read_diagnosis_frame(raw_inspire_dir: Path, config: dict) -> pd.DataFrame:
    diagnosis_df = read_csv_optimized(
        raw_inspire_dir / "diagnosis.csv",
        config=config,
        usecols=["subject_id", "chart_time", "icd10_cm"],
        large=True,
    )
    diagnosis_df["chart_time"] = pd.to_numeric(diagnosis_df["chart_time"], errors="coerce")
    diagnosis_df["icd10_cm"] = diagnosis_df["icd10_cm"].astype(str)
    return diagnosis_df


def _read_operations_frame(raw_inspire_dir: Path, config: dict, *, usecols: list[str]) -> pd.DataFrame:
    operations_df = read_csv_optimized(
        raw_inspire_dir / "operations.csv",
        config=config,
        usecols=usecols,
        large=True,
    )
    for column in [column for column in operations_df.columns if "time" in column]:
        operations_df[column] = pd.to_numeric(operations_df[column], errors="coerce")
    return operations_df


def _matched_icd_codes(
    base_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
    *,
    prefixes: list[str],
    flag_column: str,
    window_days: int,
) -> pd.DataFrame:
    filtered_diag = diagnosis_df[diagnosis_df["icd10_cm"].str.startswith(tuple(prefixes), na=False)].copy()
    if filtered_diag.empty:
        return pd.DataFrame(columns=["op_id", flag_column, f"{flag_column}_event_codes"])
    merged = base_df[["op_id", "subject_id", "opend_time"]].merge(filtered_diag, on="subject_id", how="left")
    window_minutes = int(window_days) * 24 * 60
    merged = merged[
        (merged["chart_time"] >= merged["opend_time"])
        & (merged["chart_time"] <= (merged["opend_time"] + window_minutes))
    ].copy()
    if merged.empty:
        return pd.DataFrame(columns=["op_id", flag_column, f"{flag_column}_event_codes"])
    matched = (
        merged.groupby("op_id")["icd10_cm"]
        .apply(lambda codes: ";".join(sorted(set(codes.dropna().astype(str)))))
        .reset_index(name=f"{flag_column}_event_codes")
    )
    matched[flag_column] = True
    return matched


def _collapse_event_codes(frame: pd.DataFrame, event_columns: list[str], *, output_column: str) -> pd.DataFrame:
    collapsed = frame.copy()
    if not event_columns:
        collapsed[output_column] = ""
        return collapsed
    collapsed[output_column] = collapsed[event_columns].apply(
        lambda row: ";".join(
            sorted(
                {
                    code
                    for value in row
                    if isinstance(value, str) and value
                    for code in value.split(";")
                    if code
                }
            )
        ),
        axis=1,
    )
    return collapsed


def _derive_diagnosis_window_labels(
    *,
    base_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
    outcome_cfg: dict,
) -> pd.DataFrame:
    target = outcome_cfg["target_column"]
    matched = _matched_icd_codes(
        base_df,
        diagnosis_df,
        prefixes=list(outcome_cfg["diagnosis_prefixes"]),
        flag_column=target,
        window_days=int(outcome_cfg["window_days"]),
    )
    labels = base_df[["op_id", "subject_id", "patient_id"]].copy()
    labels = labels.merge(matched, on="op_id", how="left")
    labels[target] = labels[target].fillna(False).astype(bool)
    labels[f"{target}_event_codes"] = labels[f"{target}_event_codes"].fillna("")
    return labels


def _derive_composite_labels(
    *,
    base_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
    outcome_cfg: dict,
) -> pd.DataFrame:
    target = outcome_cfg["target_column"]
    labels = base_df[["op_id", "subject_id", "patient_id"]].copy()
    event_columns: list[str] = []
    for component_key in outcome_cfg["component_keys"]:
        matched = _matched_icd_codes(
            base_df,
            diagnosis_df,
            prefixes=list(outcome_cfg["component_diagnosis_prefixes"][component_key]),
            flag_column=component_key,
            window_days=int(outcome_cfg["window_days"]),
        )
        labels = labels.merge(matched, on="op_id", how="left")
        labels[component_key] = labels[component_key].fillna(False).astype(bool)
        event_column = f"{component_key}_event_codes"
        labels[event_column] = labels[event_column].fillna("")
        event_columns.append(event_column)
    labels[target] = labels[list(outcome_cfg["component_keys"])].any(axis=1).astype(bool)
    labels = _collapse_event_codes(labels, event_columns, output_column=f"{target}_event_codes")
    return labels


def _derive_time_comparison_labels(
    *,
    base_df: pd.DataFrame,
    operations_df: pd.DataFrame,
    outcome_cfg: dict,
) -> pd.DataFrame:
    target = outcome_cfg["target_column"]
    source_column = outcome_cfg["source_column"]
    labels = base_df[["op_id", "subject_id", "patient_id", "opend_time"]].copy()
    labels = labels.merge(operations_df[["op_id", source_column]], on="op_id", how="left")
    event_time = pd.to_numeric(labels[source_column], errors="coerce")
    opend_time = pd.to_numeric(labels["opend_time"], errors="coerce")
    positive_mask = event_time.notna() & (event_time > opend_time)
    if outcome_cfg["comparison_rule"] == "strictly_after_within_window":
        window_minutes = int(outcome_cfg["window_days"]) * 24 * 60
        positive_mask &= event_time <= (opend_time + window_minutes)
    labels[target] = positive_mask.astype(bool)
    return labels.drop(columns=["opend_time"])


def derive_aki_labels(
    *,
    config: dict,
    raw_inspire_dir: Path,
    preop_df: pd.DataFrame,
    tabular_combined_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_cfg = config["cohort"]
    labs_path = raw_inspire_dir / "labs.csv"
    ward_vitals_path = raw_inspire_dir / "ward_vitals.csv"
    runtime_plan = build_stage_runtime_plan(config, "preprocess_labels")

    with ThreadPoolExecutor(max_workers=min(2, max(1, runtime_plan.csv_read_threads))) as executor:
        ward_future = executor.submit(
            read_csv_optimized,
            ward_vitals_path,
            config=config,
            usecols=["subject_id", "chart_time", "item_name", "value"],
            large=True,
        )
        labs_future = executor.submit(
            read_csv_optimized,
            labs_path,
            config=config,
            usecols=["subject_id", "chart_time", "item_name", "value"],
            large=True,
        )
        df_ward = ward_future.result()
        df_labs = labs_future.result()
    df_dialysis = df_ward[df_ward["item_name"] == cohort_cfg["dialysis_item_name"]][["subject_id", "value"]]
    df_dialysis = df_dialysis.rename(columns={"value": "dialysis"})
    df_dialysis = df_dialysis.drop_duplicates(subset="subject_id", keep="first").reset_index(drop=True)

    df_preop = preop_df[["op_id", "subject_id", "preop_creatinine", "opend_time"]].copy()
    df_preop = df_preop.merge(tabular_combined_df[["op_id"]].drop_duplicates(), on="op_id", how="inner")
    df_preop["patient_id"] = df_preop["subject_id"]

    audit: list[dict] = []
    record_count(audit, "tabular_ops_before_labels", df_preop)
    df_preop = df_preop[df_preop["preop_creatinine"].notna()]
    record_count(audit, "has_preop_creatinine", df_preop)
    df_preop = df_preop[df_preop["preop_creatinine"] < cohort_cfg["max_preop_creatinine"]]
    record_count(audit, "preop_creatinine_lt_threshold", df_preop)

    df_labs["chart_time"] = pd.to_numeric(df_labs["chart_time"], errors="coerce")
    df_creatinine = df_labs[df_labs["item_name"] == cohort_cfg["creatinine_item_name"]]
    df_merge = pd.merge(df_preop, df_creatinine, on="subject_id")

    df_aki = df_preop.copy().merge(df_dialysis, on="subject_id", how="left")
    for n_days in cohort_cfg["postop_windows_days"]:
        n_minutes = n_days * 24 * 60
        df_merge_filtered = df_merge[
            (df_merge["chart_time"] > df_merge["opend_time"])
            & (df_merge["chart_time"] <= (df_merge["opend_time"] + n_minutes))
        ]
        max_creatinine = (
            df_merge_filtered.groupby("op_id")["value"]
            .max()
            .reset_index()
            .rename(columns={"value": f"postop_creatinine_{n_days}_days"})
        )
        df_aki = pd.merge(df_aki, max_creatinine, on="op_id", how="outer")

    cols = ["postop_creatinine_2_days", "postop_creatinine_7_days", "dialysis"]
    df_aki = df_aki[~df_aki[cols].isna().all(axis=1)]
    df_aki = df_aki.fillna({"dialysis": 0})
    record_count(audit, "has_postop_creatinine_or_dialysis", df_aki)

    df_aki["crt_7_day_ratio"] = df_aki["postop_creatinine_7_days"] / df_aki["preop_creatinine"]
    df_aki["aki_1"] = (
        ((df_aki["crt_7_day_ratio"] > 1.5) & (df_aki["crt_7_day_ratio"] < 2))
        | ((df_aki["postop_creatinine_2_days"] - df_aki["preop_creatinine"]) > 0.3)
    )
    df_aki["aki_2"] = (df_aki["crt_7_day_ratio"] >= 2) & (df_aki["crt_7_day_ratio"] < 3)
    df_aki["aki_3"] = (
        (df_aki["crt_7_day_ratio"] >= 3)
        | (df_aki["postop_creatinine_7_days"] > 4)
        | (df_aki["dialysis"] > 0)
    )
    df_aki["aki_boolean"] = df_aki[["aki_2", "aki_3"]].any(axis=1).astype(bool)

    labels = df_aki[
        [
            "op_id",
            "subject_id",
            "patient_id",
            "preop_creatinine",
            "postop_creatinine_2_days",
            "postop_creatinine_7_days",
            "dialysis",
            "crt_7_day_ratio",
            "aki_1",
            "aki_2",
            "aki_3",
            "aki_boolean",
        ]
    ].copy()
    record_count(audit, "final_labeled_ops", labels)
    return labels.reset_index(drop=True), audit_frame(audit)


def derive_active_labels(
    *,
    config: dict,
    raw_inspire_dir: Path,
    preop_df: pd.DataFrame,
    tabular_combined_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if active_outcome_key(config) == "aki":
        return derive_aki_labels(
            config=config,
            raw_inspire_dir=raw_inspire_dir,
            preop_df=preop_df,
            tabular_combined_df=tabular_combined_df,
        )

    base_df = _base_label_frame(preop_df, tabular_combined_df)
    outcome_cfg = active_outcome_config(config)
    audit: list[dict] = []
    record_count(audit, "tabular_ops_before_labels", base_df)

    if outcome_cfg["kind"] == "diagnosis_window":
        diagnosis_df = _read_diagnosis_frame(raw_inspire_dir, config)
        labels = _derive_diagnosis_window_labels(base_df=base_df, diagnosis_df=diagnosis_df, outcome_cfg=outcome_cfg)
    elif outcome_cfg["kind"] == "composite":
        diagnosis_df = _read_diagnosis_frame(raw_inspire_dir, config)
        labels = _derive_composite_labels(base_df=base_df, diagnosis_df=diagnosis_df, outcome_cfg=outcome_cfg)
    elif outcome_cfg["kind"] == "time_comparison":
        operations_df = _read_operations_frame(
            raw_inspire_dir,
            config,
            usecols=["op_id", outcome_cfg["source_column"]],
        )
        labels = _derive_time_comparison_labels(base_df=base_df, operations_df=operations_df, outcome_cfg=outcome_cfg)
    else:
        raise ValueError(f"Unsupported active outcome kind: {outcome_cfg['kind']}")

    record_count(audit, "final_labeled_ops", labels)
    return labels.reset_index(drop=True), audit_frame(audit)
