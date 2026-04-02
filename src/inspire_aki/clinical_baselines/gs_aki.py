from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inspire_aki.cohort.audit import audit_frame, record_count
from inspire_aki.config import REPO_ROOT
from inspire_aki.io.csv import read_csv_optimized
from inspire_aki.runtime import build_stage_runtime_plan


GS_AKI_FACTOR_COLUMNS = [
    "gs_aki_age_ge_56",
    "gs_aki_male",
    "gs_aki_emergency",
    "gs_aki_intraperitoneal",
    "gs_aki_diabetes",
    "gs_aki_chf_30d",
    "gs_aki_ascites_30d",
    "gs_aki_hypertension",
    "gs_aki_renal_insufficiency",
]
GS_AKI_AUDIT_COLUMNS = [
    "gs_aki_renal_mild",
    "gs_aki_renal_moderate",
]
GS_AKI_DATASET_COLUMNS = [
    "op_id",
    "subject_id",
    *GS_AKI_FACTOR_COLUMNS,
    "gs_aki_count",
    "gs_aki_class",
]
_INTRAPERITONEAL_MAP_COLUMNS = [
    "icd10_pcs_5char",
    "approach",
    "nhsn_category",
    "intraperitoneal_proxy",
    "source",
    "rationale",
]


def gs_aki_enabled(config: dict[str, Any]) -> bool:
    return (
        str(config.get("study", {}).get("outcome_key", "")) == "aki"
        and "gs_aki_rule" in config.get("models", {}).get("tabular_enabled", [])
    )


def _gs_aki_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["clinical_baselines"]["gs_aki"]


def gs_aki_score_max(config: dict[str, Any]) -> int:
    return int(_gs_aki_config(config)["score_max"])


def gs_aki_high_risk_count_threshold(config: dict[str, Any]) -> int:
    class_three_bounds = _gs_aki_config(config)["class_cutpoints"]["III"]
    return int(class_three_bounds[0])


def gs_aki_high_risk_probability_threshold(config: dict[str, Any]) -> float:
    return float(gs_aki_high_risk_count_threshold(config) / gs_aki_score_max(config))


def _resolve_map_path(config: dict[str, Any]) -> Path:
    path = Path(str(_gs_aki_config(config)["intraperitoneal_map_path"]))
    return path if path.is_absolute() else REPO_ROOT / path


def _normalize_icd10_pcs(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.upper()
    normalized = normalized.where(series.notna(), pd.NA)
    return normalized


def _normalize_diagnosis_prefix(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.upper()
    normalized = normalized.where(series.notna(), pd.NA)
    return normalized.str[:3]


def _male_indicator(series: pd.Series) -> pd.Series:
    clean = series.copy()
    if clean.map(lambda value: isinstance(value, (bool, np.bool_)) or pd.isna(value)).all():
        return clean.astype("boolean").fillna(False).astype(int)

    numeric = pd.to_numeric(clean, errors="coerce")
    if numeric.notna().sum() == clean.notna().sum() and set(numeric.dropna().unique()).issubset({0, 1}):
        return numeric.fillna(0).astype(int)

    normalized = clean.astype(str).str.strip().str.upper()
    male_tokens = {"M", "MALE", "TRUE", "1"}
    return normalized.isin(male_tokens).astype(int)


def load_intraperitoneal_proxy_map(config: dict[str, Any]) -> pd.DataFrame:
    map_path = _resolve_map_path(config)
    mapping = pd.read_csv(map_path)
    missing_columns = [column for column in _INTRAPERITONEAL_MAP_COLUMNS if column not in mapping.columns]
    if missing_columns:
        raise ValueError(f"Intraperitoneal proxy map is missing required columns: {missing_columns}")
    mapping = mapping[_INTRAPERITONEAL_MAP_COLUMNS].copy()
    mapping["icd10_pcs_5char"] = _normalize_icd10_pcs(mapping["icd10_pcs_5char"])
    if mapping["icd10_pcs_5char"].isna().any():
        raise ValueError("Intraperitoneal proxy map contains missing icd10_pcs_5char values.")
    invalid_lengths = mapping.loc[mapping["icd10_pcs_5char"].str.len() != 5, "icd10_pcs_5char"].unique().tolist()
    if invalid_lengths:
        raise ValueError(f"Intraperitoneal proxy map contains non-5-character codes: {invalid_lengths[:10]}")
    duplicated = mapping.loc[mapping["icd10_pcs_5char"].duplicated(keep=False), "icd10_pcs_5char"].unique().tolist()
    if duplicated:
        raise ValueError(f"Intraperitoneal proxy map contains duplicate icd10_pcs_5char values: {duplicated[:10]}")
    mapping["intraperitoneal_proxy"] = pd.to_numeric(mapping["intraperitoneal_proxy"], errors="raise").astype(int)
    invalid_values = sorted(set(mapping["intraperitoneal_proxy"].unique()) - {0, 1})
    if invalid_values:
        raise ValueError(f"Intraperitoneal proxy map contains invalid intraperitoneal_proxy values: {invalid_values}")
    return mapping.reset_index(drop=True)


def derive_gs_aki_diagnosis_features(
    *,
    base_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    gs_aki_cfg = _gs_aki_config(config)
    diag = diagnosis_df.copy()
    diag["chart_time"] = pd.to_numeric(diag["chart_time"], errors="coerce")
    diag["icd10_prefix"] = _normalize_diagnosis_prefix(diag["icd10_cm"])
    relevant_prefixes = set(gs_aki_cfg["diabetes_prefixes"]) | set(gs_aki_cfg["hypertension_prefixes"]) | set(
        gs_aki_cfg["chf_prefixes"]
    ) | set(gs_aki_cfg["ascites_prefixes"])
    diag = diag[diag["subject_id"].isin(base_df["subject_id"]) & diag["icd10_prefix"].isin(relevant_prefixes)].copy()

    base = base_df[["op_id", "subject_id", "opstart_time"]].copy()
    if diag.empty:
        return base.assign(
            gs_aki_diabetes=0,
            gs_aki_hypertension=0,
            gs_aki_chf_30d=0,
            gs_aki_ascites_30d=0,
        )[["op_id", "gs_aki_diabetes", "gs_aki_hypertension", "gs_aki_chf_30d", "gs_aki_ascites_30d"]]

    merged = base.merge(diag[["subject_id", "chart_time", "icd10_prefix"]], on="subject_id", how="left")
    merged = merged[merged["chart_time"].notna() & (merged["chart_time"] < merged["opstart_time"])].copy()
    if merged.empty:
        return base.assign(
            gs_aki_diabetes=0,
            gs_aki_hypertension=0,
            gs_aki_chf_30d=0,
            gs_aki_ascites_30d=0,
        )[["op_id", "gs_aki_diabetes", "gs_aki_hypertension", "gs_aki_chf_30d", "gs_aki_ascites_30d"]]

    recent_window_minutes = int(gs_aki_cfg["recent_window_days"]) * 24 * 60
    recent_mask = merged["chart_time"] >= (merged["opstart_time"] - recent_window_minutes)

    def _flag_rows(mask: pd.Series, column_name: str) -> pd.DataFrame:
        flagged = merged.loc[mask, ["op_id"]].drop_duplicates()
        flagged[column_name] = 1
        return flagged

    dm_rows = _flag_rows(merged["icd10_prefix"].isin(gs_aki_cfg["diabetes_prefixes"]), "gs_aki_diabetes")
    htn_rows = _flag_rows(merged["icd10_prefix"].isin(gs_aki_cfg["hypertension_prefixes"]), "gs_aki_hypertension")
    chf_rows = _flag_rows(recent_mask & merged["icd10_prefix"].isin(gs_aki_cfg["chf_prefixes"]), "gs_aki_chf_30d")
    ascites_rows = _flag_rows(
        recent_mask & merged["icd10_prefix"].isin(gs_aki_cfg["ascites_prefixes"]),
        "gs_aki_ascites_30d",
    )

    features = base[["op_id"]].drop_duplicates().copy()
    for feature_df in [dm_rows, htn_rows, chf_rows, ascites_rows]:
        features = features.merge(feature_df, on="op_id", how="left")
    for column in ["gs_aki_diabetes", "gs_aki_hypertension", "gs_aki_chf_30d", "gs_aki_ascites_30d"]:
        features[column] = features[column].fillna(0).astype(int)
    return features


def score_gs_aki_counts(counts: pd.Series, config: dict[str, Any]) -> pd.Series:
    class_cutpoints = _gs_aki_config(config)["class_cutpoints"]
    labels = pd.Series(index=counts.index, dtype="object")
    numeric = pd.to_numeric(counts, errors="coerce").astype("Int64")
    for class_name, bounds in class_cutpoints.items():
        lower, upper = int(bounds[0]), int(bounds[1])
        labels.loc[numeric.between(lower, upper, inclusive="both")] = class_name
    if labels.isna().any():
        missing_scores = numeric.loc[labels.isna()].dropna().unique().tolist()
        raise ValueError(f"Encountered GS-AKI counts without a configured class mapping: {missing_scores}")
    return labels.astype(str)


def build_gs_aki_features(
    config: dict[str, Any],
    raw_inspire_dir: Path,
    preop_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    runtime_plan = build_stage_runtime_plan(config, "preprocess_preop")
    with ThreadPoolExecutor(max_workers=min(2, max(1, runtime_plan.csv_read_threads))) as executor:
        operations_future = executor.submit(
            read_csv_optimized,
            raw_inspire_dir / "operations.csv",
            config=config,
            usecols=["op_id", "subject_id", "icd10_pcs"],
            large=True,
        )
        diagnosis_future = executor.submit(
            read_csv_optimized,
            raw_inspire_dir / "diagnosis.csv",
            config=config,
            usecols=["subject_id", "chart_time", "icd10_cm"],
            large=True,
        )
        operations_df = operations_future.result()
        diagnosis_df = diagnosis_future.result()

    audit: list[dict[str, Any]] = []
    base = preop_df[
        ["op_id", "subject_id", "age", "sex", "emop", "opstart_time", "preop_creatinine"]
    ].copy()
    if base["op_id"].duplicated().any():
        raise ValueError("GS-AKI feature derivation requires unique op_id values in the retained preop cohort.")
    record_count(audit, "gs_aki_preop_rows", base)

    ops_subset = operations_df[["op_id", "subject_id", "icd10_pcs"]].drop_duplicates(subset=["op_id"], keep="last")
    base = base.merge(ops_subset, on=["op_id", "subject_id"], how="left")
    base["icd10_pcs_5char"] = _normalize_icd10_pcs(base["icd10_pcs"])
    missing_required_mask = (
        base["age"].isna()
        | base["sex"].isna()
        | base["emop"].isna()
        | base["preop_creatinine"].isna()
        | base["icd10_pcs_5char"].isna()
    )
    if missing_required_mask.any():
        missing_required = {
            "age": int(base["age"].isna().sum()),
            "sex": int(base["sex"].isna().sum()),
            "emop": int(base["emop"].isna().sum()),
            "preop_creatinine": int(base["preop_creatinine"].isna().sum()),
            "icd10_pcs": int(base["icd10_pcs_5char"].isna().sum()),
        }
        missing_required = {key: value for key, value in missing_required.items() if value > 0}
        record_count(audit, "gs_aki_missing_required_inputs_excluded", base.loc[missing_required_mask])
        base = base.loc[~missing_required_mask].copy()
        record_count(audit, "gs_aki_complete_case_rows", base)
        if base.empty:
            raise ValueError(
                "GS-AKI preprocessing encountered missing required inputs for every retained operation. "
                f"Missing counts: {missing_required}"
            )
    invalid_code_lengths = (
        base.loc[base["icd10_pcs_5char"].str.len() != 5, "icd10_pcs_5char"].dropna().unique().tolist()
    )
    if invalid_code_lengths:
        raise ValueError(f"GS-AKI requires 5-character ICD-10-PCS codes. Invalid values: {invalid_code_lengths[:10]}")

    proxy_map = load_intraperitoneal_proxy_map(config)
    base = base.merge(proxy_map[["icd10_pcs_5char", "intraperitoneal_proxy"]], on="icd10_pcs_5char", how="left")
    if base["intraperitoneal_proxy"].isna().any():
        unmapped = (
            base.loc[base["intraperitoneal_proxy"].isna(), "icd10_pcs_5char"]
            .value_counts()
            .head(10)
            .to_dict()
        )
        raise ValueError(
            "Intraperitoneal proxy map did not cover every retained operation code in the GS-AKI cohort. "
            f"Top missing codes: {unmapped}"
        )

    diag_features = derive_gs_aki_diagnosis_features(base_df=base, diagnosis_df=diagnosis_df, config=config)
    base = base.merge(diag_features, on="op_id", how="left")
    for column in ["gs_aki_diabetes", "gs_aki_hypertension", "gs_aki_chf_30d", "gs_aki_ascites_30d"]:
        base[column] = base[column].fillna(0).astype(int)

    base["gs_aki_age_ge_56"] = (pd.to_numeric(base["age"], errors="coerce") >= 56).astype(int)
    base["gs_aki_male"] = _male_indicator(base["sex"])
    base["gs_aki_emergency"] = (pd.to_numeric(base["emop"], errors="coerce") == 1).astype(int)
    base["gs_aki_intraperitoneal"] = pd.to_numeric(base["intraperitoneal_proxy"], errors="raise").astype(int)
    creatinine = pd.to_numeric(base["preop_creatinine"], errors="coerce")
    base["gs_aki_renal_mild"] = ((creatinine >= 1.2) & (creatinine < 2.0)).astype(int)
    base["gs_aki_renal_moderate"] = (creatinine >= 2.0).astype(int)
    base["gs_aki_renal_insufficiency"] = (creatinine >= 1.2).astype(int)
    base["gs_aki_count"] = base[GS_AKI_FACTOR_COLUMNS].sum(axis=1).astype(int)
    base["gs_aki_class"] = score_gs_aki_counts(base["gs_aki_count"], config)

    record_count(audit, "gs_aki_intraperitoneal_positive", base.loc[base["gs_aki_intraperitoneal"] == 1])
    record_count(audit, "gs_aki_diabetes_positive", base.loc[base["gs_aki_diabetes"] == 1])
    record_count(audit, "gs_aki_hypertension_positive", base.loc[base["gs_aki_hypertension"] == 1])
    record_count(audit, "gs_aki_chf_30d_positive", base.loc[base["gs_aki_chf_30d"] == 1])
    record_count(audit, "gs_aki_ascites_30d_positive", base.loc[base["gs_aki_ascites_30d"] == 1])
    record_count(audit, "gs_aki_renal_insufficiency_positive", base.loc[base["gs_aki_renal_insufficiency"] == 1])
    record_count(audit, "gs_aki_class_I", base.loc[base["gs_aki_class"] == "I"])
    record_count(audit, "gs_aki_class_II", base.loc[base["gs_aki_class"] == "II"])
    record_count(audit, "gs_aki_class_III", base.loc[base["gs_aki_class"] == "III"])
    record_count(audit, "gs_aki_class_IV", base.loc[base["gs_aki_class"] == "IV"])
    record_count(audit, "gs_aki_class_V", base.loc[base["gs_aki_class"] == "V"])

    ordered_columns = [
        "op_id",
        "subject_id",
        "age",
        "sex",
        "emop",
        "opstart_time",
        "preop_creatinine",
        "icd10_pcs_5char",
        *GS_AKI_FACTOR_COLUMNS,
        *GS_AKI_AUDIT_COLUMNS,
        "gs_aki_count",
        "gs_aki_class",
    ]
    return base[ordered_columns].sort_values("op_id", kind="stable").reset_index(drop=True), audit_frame(audit)
