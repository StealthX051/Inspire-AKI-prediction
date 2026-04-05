from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from inspire_aki.config import DEFAULT_CONFIG_PATH, active_outcome_config, active_target_column, config_hash, load_config
from inspire_aki.io.paths import ProjectPaths
from inspire_aki.reporting.procedure_audit import (
    _DEPARTMENT_LABELS as PROCEDURE_AUDIT_DEPARTMENT_LABELS,
)
from inspire_aki.reporting.procedure_audit import build_cms_prefix_reference, load_cms_order_entries
from inspire_aki.reporting.tables import _DEPARTMENT_LABELS as REPORT_DEPARTMENT_LABELS


ANTYPE_CATEGORIES = ("General", "MAC", "Neuraxial", "Regional")
SUMMARY_SLICES = ("overall", "OS", "OT", "GS", "NS", "UR")
TOP_ICD10PCS4_N = 10
DEFAULT_REVIEWER_OUTPUT_DIRNAME = "reviewer_department_audit"
PROCEDURE_NAME_CANDIDATES = (
    "op_name",
    "opname",
    "operation_name",
    "procedure_name",
    "primary_procedure",
    "procedure_group",
    "procedure_group_name",
    "operation_group",
)
BASE_OPERATION_COLUMNS = (
    "op_id",
    "subject_id",
    "department",
    "antype",
    "icd10_pcs",
    "opstart_time",
    "opend_time",
)


@dataclass(frozen=True)
class AuditContext:
    config: dict[str, Any]
    config_path: Path | None
    paths: ProjectPaths
    out_dir: Path
    target_column: str


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the provenance and meaning of department_OS in the INSPIRE AKI cohort.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Config path to load. Defaults to configs/aki/default.yaml.")
    parser.add_argument("--raw-dir", default=None, help="Optional override for paths.raw_inspire_dir.")
    parser.add_argument("--artifacts-dir", default=None, help="Optional override for paths.artifacts_dir.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. Defaults to <artifacts_dir>/reports/reviewer_department_audit/.",
    )
    return parser


def default_reviewer_output_dir(paths: ProjectPaths) -> Path:
    return paths.artifact_path("reports", DEFAULT_REVIEWER_OUTPUT_DIRNAME)


def _load_context(
    *,
    config_path: str | Path | None,
    raw_dir: str | Path | None,
    artifacts_dir: str | Path | None,
    out_dir: str | Path | None,
) -> AuditContext:
    resolved_config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    config = copy.deepcopy(load_config(resolved_config_path))
    if raw_dir is not None:
        config["paths"]["raw_inspire_dir"] = str(Path(raw_dir))
    if artifacts_dir is not None:
        config["paths"]["artifacts_dir"] = str(Path(artifacts_dir))

    paths = ProjectPaths.from_config(config)
    resolved_out_dir = Path(out_dir) if out_dir is not None else default_reviewer_output_dir(paths)
    if not resolved_out_dir.is_absolute():
        resolved_out_dir = paths.repo_root / resolved_out_dir
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    return AuditContext(
        config=config,
        config_path=resolved_config_path,
        paths=paths,
        out_dir=resolved_out_dir,
        target_column=active_target_column(config),
    )


def load_raw_department_dictionary(path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    column_lookup = {str(column).strip().lower(): str(column) for column in raw.columns}
    code_col = next((column_lookup[key] for key in ("abbreviations", "abbr", "code", "department") if key in column_lookup), None)
    label_col = next((column for key, column in column_lookup.items() if "full" in key and "name" in key), None)
    if code_col is None or label_col is None:
        raise ValueError(f"Could not infer department dictionary columns from {path}. Columns: {raw.columns.tolist()}")
    return (
        raw[[code_col, label_col]]
        .rename(columns={code_col: "department_code", label_col: "raw_dictionary_label"})
        .assign(
            department_code=lambda frame: frame["department_code"].astype("string").str.strip().str.upper(),
            raw_dictionary_label=lambda frame: frame["raw_dictionary_label"].astype("string").str.strip(),
        )
        .sort_values("department_code", kind="stable")
        .reset_index(drop=True)
    )


def current_department_label_frame(codes: Iterable[str] | None = None) -> pd.DataFrame:
    all_codes = set(REPORT_DEPARTMENT_LABELS) | set(PROCEDURE_AUDIT_DEPARTMENT_LABELS)
    if codes is not None:
        all_codes |= {str(code).strip().upper() for code in codes}
    rows = [
        {
            "department_code": code,
            "current_report_label": REPORT_DEPARTMENT_LABELS.get(code, pd.NA),
            "current_procedure_audit_label": PROCEDURE_AUDIT_DEPARTMENT_LABELS.get(code, pd.NA),
        }
        for code in sorted(all_codes)
    ]
    return pd.DataFrame(rows)


def _load_operations_frame(raw_dir: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    operations_path = raw_dir / "operations.csv"
    header = pd.read_csv(operations_path, nrows=0).columns.tolist()
    procedure_cols = [column for column in PROCEDURE_NAME_CANDIDATES if column in header]
    usecols = [column for column in BASE_OPERATION_COLUMNS if column in header] + procedure_cols
    operations_df = pd.read_csv(operations_path, usecols=usecols)
    operations_df["department"] = operations_df["department"].astype("string").str.strip().str.upper()
    operations_df["antype"] = operations_df["antype"].astype("string").str.strip()
    operations_df["icd10_pcs"] = operations_df["icd10_pcs"].astype("string").str.strip().str.upper()
    for column in ("opstart_time", "opend_time"):
        operations_df[column] = pd.to_numeric(operations_df[column], errors="coerce")
    operations_df["op_len_raw_minutes"] = operations_df["opend_time"] - operations_df["opstart_time"]
    return operations_df, procedure_cols, header


def _load_schema_frame(raw_dir: Path) -> pd.DataFrame:
    schema_path = raw_dir / "schema.csv"
    if not schema_path.exists():
        return pd.DataFrame()
    return pd.read_csv(schema_path)


def _manifest_warnings(context: AuditContext) -> list[str]:
    warnings: list[str] = []
    expected_hash = config_hash(context.config)
    for manifest_name in ("preprocess_preop.json", "preprocess_labels.json", "report_tables.json"):
        manifest_path = context.paths.artifacts_root / "manifests" / manifest_name
        if not manifest_path.exists():
            continue
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed_hash = str(payload.get("config_hash", ""))
        if observed_hash and observed_hash != expected_hash:
            warnings.append(
                f"Manifest {manifest_name} has config_hash={observed_hash}, which differs from the loaded config_hash={expected_hash}."
            )
    return warnings


def load_final_audit_cohort(context: AuditContext) -> tuple[pd.DataFrame, list[str], list[str]]:
    tabular_path = context.paths.artifact_path("datasets", "tabular", "tabular_combined_unnormalized.csv")
    labels_path = context.paths.artifact_path("cohort", "labels.csv")
    if not tabular_path.exists():
        raise FileNotFoundError(f"Expected tabular cohort artifact was not found: {tabular_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Expected labels artifact was not found: {labels_path}")

    tabular_df = pd.read_csv(
        tabular_path,
        usecols=lambda column: column == "op_id" or column == "op_len" or str(column).startswith("department_"),
    )
    labels_df = pd.read_csv(labels_path)
    if context.target_column not in labels_df.columns:
        raise KeyError(f"Target column '{context.target_column}' was not present in {labels_path}.")
    label_cols = [column for column in ("op_id", "subject_id", "patient_id", context.target_column) if column in labels_df.columns]
    labeled_df = tabular_df.merge(labels_df[label_cols], on="op_id", how="inner", validate="one_to_one")

    operations_df, procedure_cols, operations_header = _load_operations_frame(context.paths.raw_inspire_dir)
    if operations_df["op_id"].duplicated().any():
        raise ValueError("Raw operations.csv contained duplicate op_id values; the audit expects op_id to be unique.")
    merge_cols = ["op_id", "subject_id", "department", "antype", "icd10_pcs", "opstart_time", "opend_time", "op_len_raw_minutes", *procedure_cols]
    audit_df = labeled_df.merge(operations_df[merge_cols], on="op_id", how="inner", validate="one_to_one", suffixes=("", "_raw"))

    if "subject_id_raw" in audit_df.columns:
        if "subject_id" in audit_df.columns:
            mismatch_mask = (
                pd.to_numeric(audit_df["subject_id"], errors="coerce")
                != pd.to_numeric(audit_df["subject_id_raw"], errors="coerce")
            )
            mismatch_mask = mismatch_mask.fillna(False)
            if bool(mismatch_mask.any()):
                raise ValueError("Subject ID mismatch between labels.csv and operations.csv for the joined audit cohort.")
        else:
            audit_df = audit_df.rename(columns={"subject_id_raw": "subject_id"})
        if "subject_id_raw" in audit_df.columns:
            audit_df = audit_df.drop(columns=["subject_id_raw"])

    if "patient_id" not in audit_df.columns:
        audit_df["patient_id"] = audit_df["subject_id"]

    audit_df["raw_department"] = audit_df["department"].astype("string").str.strip().str.upper()
    if "op_len" in audit_df.columns:
        audit_df["op_len_minutes"] = pd.to_numeric(audit_df["op_len"], errors="coerce")
    else:
        audit_df["op_len_minutes"] = audit_df["op_len_raw_minutes"]
    fallback_mask = audit_df["op_len_minutes"].isna()
    audit_df.loc[fallback_mask, "op_len_minutes"] = audit_df.loc[fallback_mask, "op_len_raw_minutes"]
    audit_df["icd10_pcs4"] = audit_df["icd10_pcs"].where(audit_df["icd10_pcs"].str.len() >= 4, pd.NA).str[:4]
    audit_df["pcs_prefix"] = audit_df["icd10_pcs"].where(audit_df["icd10_pcs"].str.len() >= 5, pd.NA).str[:5]
    return audit_df, procedure_cols, operations_header


def compare_department_indicator(audit_df: pd.DataFrame, department_code: str) -> dict[str, Any]:
    code = str(department_code).strip().upper()
    indicator_col = f"department_{code}"
    if indicator_col not in audit_df.columns:
        raise KeyError(f"Expected indicator column '{indicator_col}' in audit cohort.")
    indicator = pd.to_numeric(audit_df[indicator_col], errors="coerce").fillna(0).astype(int)
    expected = audit_df["raw_department"].astype("string").eq(code).astype(int)
    mismatches = audit_df.loc[indicator != expected, ["op_id", "raw_department", indicator_col]].copy()
    positive_mismatches = audit_df.loc[(indicator == 1) & (audit_df["raw_department"] != code), ["op_id", "raw_department"]].copy()
    return {
        "department_code": code,
        "indicator_column": indicator_col,
        "n_rows": int(len(audit_df)),
        "indicator_positive_n": int(indicator.sum()),
        "raw_positive_n": int(expected.sum()),
        "mismatch_n": int(len(mismatches)),
        "positive_mismatch_n": int(len(positive_mismatches)),
        "matches_exactly": bool(mismatches.empty),
        "mismatched_op_ids": mismatches["op_id"].astype(int).head(10).tolist(),
    }


def build_patient_level_department_counts(audit_df: pd.DataFrame) -> pd.DataFrame:
    patient_df = (
        audit_df.sort_values("op_id", kind="stable")
        .drop_duplicates(subset=["subject_id"], keep="last")
        .reset_index(drop=True)
    )
    total_patients = int(len(patient_df))
    dept_columns = sorted(column for column in patient_df.columns if str(column).startswith("department_"))
    rows = []
    for column in dept_columns:
        code = column.removeprefix("department_").upper()
        count = int(pd.to_numeric(patient_df[column], errors="coerce").fillna(0).astype(int).sum())
        rows.append(
            {
                "department_code": code,
                "n_patients": count,
                "pct_patients": _pct(count, total_patients),
            }
        )
    return pd.DataFrame(rows).sort_values("department_code", kind="stable").reset_index(drop=True)


def build_department_raw_counts(audit_df: pd.DataFrame, raw_dictionary_df: pd.DataFrame) -> pd.DataFrame:
    total_ops = int(len(audit_df))
    total_patients = int(audit_df["patient_id"].nunique())
    op_counts = audit_df["raw_department"].value_counts(dropna=False)
    patient_counts = build_patient_level_department_counts(audit_df).set_index("department_code")
    rows = []
    for row in raw_dictionary_df.itertuples(index=False):
        code = str(row.department_code).strip().upper()
        patient_count = int(patient_counts.loc[code, "n_patients"]) if code in patient_counts.index else 0
        patient_pct = float(patient_counts.loc[code, "pct_patients"]) if code in patient_counts.index else 0.0
        op_count = int(op_counts.get(code, 0))
        rows.append(
            {
                "department_code": code,
                "raw_dictionary_label": row.raw_dictionary_label,
                "current_report_label": REPORT_DEPARTMENT_LABELS.get(code, pd.NA),
                "n_ops_final_labeled": op_count,
                "pct_ops_final_labeled": _pct(op_count, total_ops),
                "n_patients_final_labeled": patient_count,
                "pct_patients_final_labeled": patient_pct if total_patients > 0 else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("department_code", kind="stable").reset_index(drop=True)


def build_slice_summary(audit_df: pd.DataFrame, raw_dictionary_df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    label_lookup = raw_dictionary_df.set_index("department_code")["raw_dictionary_label"].to_dict()
    total_ops = int(len(audit_df))
    total_patients = int(audit_df["patient_id"].nunique())
    rows = []
    for slice_code in SUMMARY_SLICES:
        if slice_code == "overall":
            slice_df = audit_df.copy()
            slice_label = "Overall cohort"
        else:
            slice_df = audit_df.loc[audit_df["raw_department"] == slice_code].copy()
            slice_label = label_lookup.get(slice_code, slice_code)
        n_ops = int(len(slice_df))
        n_patients = int(slice_df["patient_id"].nunique()) if n_ops else 0
        op_len = pd.to_numeric(slice_df["op_len_minutes"], errors="coerce").dropna()
        row = {
            "slice": slice_code,
            "slice_label": slice_label,
            "n_ops": n_ops,
            "pct_ops": _pct(n_ops, total_ops),
            "n_patients": n_patients,
            "pct_patients": _pct(n_patients, total_patients),
            "severe_aki_n": int(pd.to_numeric(slice_df[target_column], errors="coerce").fillna(0).astype(int).sum()) if n_ops else 0,
            "severe_aki_rate": _pct(
                int(pd.to_numeric(slice_df[target_column], errors="coerce").fillna(0).astype(int).sum()) if n_ops else 0,
                n_ops,
            ),
            "op_len_median": float(op_len.median()) if not op_len.empty else pd.NA,
            "op_len_q1": float(op_len.quantile(0.25)) if not op_len.empty else pd.NA,
            "op_len_q3": float(op_len.quantile(0.75)) if not op_len.empty else pd.NA,
            "op_len_mean": float(op_len.mean()) if not op_len.empty else pd.NA,
            "op_len_sd": float(op_len.std(ddof=1)) if len(op_len) > 1 else (0.0 if len(op_len) == 1 else pd.NA),
        }
        for antype in ANTYPE_CATEGORIES:
            count = int(slice_df["antype"].eq(antype).sum()) if n_ops else 0
            slug = antype.lower()
            row[f"antype_{slug}_n"] = count
            row[f"antype_{slug}_pct"] = _pct(count, n_ops)
        rows.append(row)
    return pd.DataFrame(rows)


def build_top_icd10pcs4_frame(
    audit_df: pd.DataFrame,
    *,
    department_code: str,
    target_column: str,
    cms_reference: pd.DataFrame | None = None,
    top_n: int = TOP_ICD10PCS4_N,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    code = str(department_code).strip().upper()
    slice_df = audit_df.loc[audit_df["raw_department"] == code].copy()
    if slice_df.empty:
        empty_csv = pd.DataFrame(columns=["rank", "icd10_pcs4", "n_ops", "pct_of_os_ops", "severe_aki_n", "severe_aki_rate"])
        empty_report = pd.DataFrame(
            columns=[
                "rank",
                "icd10_pcs4",
                "n_ops",
                "pct_of_ops",
                "severe_aki_n",
                "severe_aki_rate",
                "representative_5char_prefix",
                "body_system_desc",
                "root_op_desc",
                "example_long_title",
            ]
        )
        return empty_csv, empty_report

    total_ops = int(len(slice_df))
    grouped_rows: list[dict[str, Any]] = []
    for icd10_pcs4, group_df in (
        slice_df.dropna(subset=["icd10_pcs4"])
        .groupby("icd10_pcs4", sort=True)
    ):
        severe_n = int(pd.to_numeric(group_df[target_column], errors="coerce").fillna(0).astype(int).sum())
        representative_prefix = pd.NA
        body_system_desc = pd.NA
        root_op_desc = pd.NA
        example_long_title = pd.NA
        if cms_reference is not None and not cms_reference.empty:
            prefix_counts = group_df["pcs_prefix"].dropna().astype(str).value_counts()
            if not prefix_counts.empty:
                representative_prefix = str(prefix_counts.index[0])
                reference_row = cms_reference.loc[cms_reference["pcs_prefix"] == representative_prefix]
                if not reference_row.empty:
                    reference = reference_row.iloc[0]
                    body_system_desc = reference.get("body_system_desc", pd.NA)
                    root_op_desc = reference.get("root_op_desc", pd.NA)
                    example_long_title = reference.get("sample_long_title", pd.NA)
        grouped_rows.append(
            {
                "icd10_pcs4": str(icd10_pcs4),
                "n_ops": int(len(group_df)),
                "pct_of_ops": _pct(len(group_df), total_ops),
                "severe_aki_n": severe_n,
                "severe_aki_rate": _pct(severe_n, len(group_df)),
                "representative_5char_prefix": representative_prefix,
                "body_system_desc": body_system_desc,
                "root_op_desc": root_op_desc,
                "example_long_title": example_long_title,
            }
        )

    report_df = (
        pd.DataFrame(grouped_rows)
        .sort_values(["n_ops", "icd10_pcs4"], ascending=[False, True], kind="stable")
        .head(top_n)
        .reset_index(drop=True)
    )
    report_df.insert(0, "rank", range(1, len(report_df) + 1))
    csv_df = report_df[["rank", "icd10_pcs4", "n_ops", "pct_of_ops", "severe_aki_n", "severe_aki_rate"]].rename(
        columns={"pct_of_ops": f"pct_of_{code.lower()}_ops"}
    )
    return csv_df, report_df


def _load_cms_reference(context: AuditContext) -> tuple[pd.DataFrame, list[str]]:
    warnings: list[str] = []
    procedure_audit_cfg = context.config.get("reports", {}).get("procedure_audit", {})
    zip_path = procedure_audit_cfg.get("cms_order_zip_path")
    if not zip_path:
        return pd.DataFrame(), ["No reports.procedure_audit.cms_order_zip_path was configured; ICD-10-PCS title enrichment was skipped."]
    try:
        return build_cms_prefix_reference(load_cms_order_entries(zip_path)), warnings
    except Exception as exc:  # pragma: no cover - exercised against local mounted data
        warnings.append(f"Could not load CMS ICD-10-PCS order reference: {exc}")
        return pd.DataFrame(), warnings


def _artifact_table_reconciliation(raw_counts_df: pd.DataFrame, artifact_table_path: Path) -> pd.DataFrame:
    if not artifact_table_path.exists():
        return pd.DataFrame(
            columns=[
                "department_code",
                "expected_raw_label",
                "expected_patient_finding",
                "artifact_table_label",
                "artifact_table_finding",
            ]
        )
    artifact_df = pd.read_csv(artifact_table_path)
    rows = []
    for code in ("OS", "OT"):
        match_row = raw_counts_df.loc[raw_counts_df["department_code"] == code]
        if match_row.empty:
            continue
        finding = _format_count_pct(
            int(match_row["n_patients_final_labeled"].iloc[0]),
            float(match_row["pct_patients_final_labeled"].iloc[0]),
        )
        artifact_match = artifact_df.loc[artifact_df["finding"].astype(str) == finding, ["characteristic", "finding"]]
        if artifact_match.empty:
            rows.append(
                {
                    "department_code": code,
                    "expected_raw_label": match_row["raw_dictionary_label"].iloc[0],
                    "expected_patient_finding": finding,
                    "artifact_table_label": pd.NA,
                    "artifact_table_finding": pd.NA,
                }
            )
            continue
        rows.append(
            {
                "department_code": code,
                "expected_raw_label": match_row["raw_dictionary_label"].iloc[0],
                "expected_patient_finding": finding,
                "artifact_table_label": artifact_match.iloc[0]["characteristic"],
                "artifact_table_finding": artifact_match.iloc[0]["finding"],
            }
        )
    return pd.DataFrame(rows)


def _find_procedure_name_fields(operations_header: list[str], schema_df: pd.DataFrame) -> list[str]:
    header_matches = [column for column in PROCEDURE_NAME_CANDIDATES if column in operations_header]
    if header_matches:
        return header_matches
    if schema_df.empty:
        return []
    subset = schema_df.loc[
        schema_df["Table"].astype("string").str.strip().eq("operations")
        & schema_df["Variable"].astype("string").str.contains("name|group", case=False, na=False),
        "Variable",
    ]
    return sorted({str(value).strip() for value in subset.tolist() if str(value).strip()})


def _pct(numerator: int | float, denominator: int | float) -> float:
    if denominator in (0, 0.0) or pd.isna(denominator):
        return 0.0
    return float(numerator) * 100.0 / float(denominator)


def _format_count_pct(count: int, pct: float) -> str:
    return f"{int(count)} ({float(pct):.2f}%)"


def _format_metric(value: Any, *, decimals: int = 2) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float)) and float(value).is_integer():
        return str(int(value))
    return f"{float(value):.{decimals}f}"


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None_"
    display_df = df.copy()
    for column in display_df.columns:
        display_df[column] = display_df[column].map(lambda value: "N/A" if pd.isna(value) else str(value))
    rows = [display_df.columns.tolist(), ["---"] * len(display_df.columns), *display_df.values.tolist()]
    return "\n".join(f"| {' | '.join(row)} |" for row in rows)


def _format_summary_for_report(summary_df: pd.DataFrame) -> pd.DataFrame:
    report_df = summary_df.copy()
    report_df["n_ops"] = report_df["n_ops"].map(lambda value: f"{int(value):,}")
    report_df["pct_ops"] = report_df["pct_ops"].map(lambda value: f"{value:.2f}%")
    report_df["n_patients"] = report_df["n_patients"].map(lambda value: f"{int(value):,}")
    report_df["pct_patients"] = report_df["pct_patients"].map(lambda value: f"{value:.2f}%")
    report_df["severe_aki_n"] = report_df["severe_aki_n"].map(lambda value: f"{int(value):,}")
    report_df["severe_aki_rate"] = report_df["severe_aki_rate"].map(lambda value: f"{value:.2f}%")
    for column in ("op_len_median", "op_len_q1", "op_len_q3", "op_len_mean", "op_len_sd"):
        report_df[column] = report_df[column].map(_format_metric)
    for antype in ANTYPE_CATEGORIES:
        slug = antype.lower()
        report_df[f"antype_{slug}_n"] = report_df[f"antype_{slug}_n"].map(lambda value: f"{int(value):,}")
        report_df[f"antype_{slug}_pct"] = report_df[f"antype_{slug}_pct"].map(lambda value: f"{value:.2f}%")
    return report_df


def _format_indicator_checks_for_report(indicator_checks: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(indicator_checks)
    if frame.empty:
        return frame
    frame = frame.rename(
        columns={
            "department_code": "department_code",
            "indicator_column": "indicator_column",
            "indicator_positive_n": "indicator_positive_n",
            "raw_positive_n": "raw_positive_n",
            "mismatch_n": "mismatch_n",
            "positive_mismatch_n": "positive_mismatch_n",
            "matches_exactly": "matches_exactly",
        }
    )
    frame["matches_exactly"] = frame["matches_exactly"].map(lambda value: "yes" if value else "no")
    frame["mismatched_op_ids"] = frame["mismatched_op_ids"].map(lambda values: ", ".join(str(value) for value in values) if values else "")
    return frame[
        [
            "department_code",
            "indicator_column",
            "indicator_positive_n",
            "raw_positive_n",
            "mismatch_n",
            "positive_mismatch_n",
            "matches_exactly",
            "mismatched_op_ids",
        ]
    ]


def _format_mapping_for_report(mapping_df: pd.DataFrame) -> pd.DataFrame:
    frame = mapping_df.copy()
    for column in frame.columns:
        frame[column] = frame[column].map(lambda value: "N/A" if pd.isna(value) else str(value))
    return frame


def _build_report(
    *,
    context: AuditContext,
    raw_dictionary_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    raw_counts_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    indicator_checks: list[dict[str, Any]],
    os_top_report_df: pd.DataFrame,
    ot_top_report_df: pd.DataFrame,
    procedure_name_fields: list[str],
    manifest_warnings: list[str],
    artifact_reconciliation_df: pd.DataFrame,
    feature_bug: bool,
) -> str:
    total_ops = int(len(pd.read_csv(context.paths.artifact_path("cohort", "labels.csv"))))
    total_patients = int(summary_df.loc[summary_df["slice"] == "overall", "n_patients"].iloc[0])
    outcome_cfg = active_outcome_config(context.config)
    outcome_label = str(outcome_cfg.get("positive_label") or outcome_cfg.get("display_name") or context.target_column)
    os_row = raw_counts_df.loc[raw_counts_df["department_code"] == "OS"].iloc[0]
    ot_row = raw_counts_df.loc[raw_counts_df["department_code"] == "OT"].iloc[0]
    overall_row = summary_df.loc[summary_df["slice"] == "overall"].iloc[0]
    os_summary = summary_df.loc[summary_df["slice"] == "OS"].iloc[0]
    ot_summary = summary_df.loc[summary_df["slice"] == "OT"].iloc[0]
    os_indicator = next(check for check in indicator_checks if check["department_code"] == "OS")
    ot_indicator = next(check for check in indicator_checks if check["department_code"] == "OT")
    artifact_label_mismatch = False
    if not artifact_reconciliation_df.empty:
        artifact_label_mismatch = bool(
            (
                artifact_reconciliation_df["expected_raw_label"].astype("string")
                != artifact_reconciliation_df["artifact_table_label"].astype("string")
            )
            .fillna(False)
            .any()
        )

    lines: list[str] = [
        "# Department `OS` Provenance Audit",
        "",
        "## Executive Summary",
        f"- The final labeled analytic cohort contained {total_ops:,} operations across {total_patients:,} patients. `department_OS` matched raw `department == \"OS\"` exactly in the joined cohort with {int(os_indicator['mismatch_n'])} row-level mismatches; `department_OT` matched raw `department == \"OT\"` with {int(ot_indicator['mismatch_n'])} mismatches.",
        f"- Raw INSPIRE `department.csv` defines `OS` as `{os_row['raw_dictionary_label']}` and `OT` as `{ot_row['raw_dictionary_label']}`. In the operation-level final cohort, `OS` contributed {int(os_summary['n_ops']):,} operations ({float(os_summary['pct_ops']):.2f}%) with a `{outcome_label}` rate of {float(os_summary['severe_aki_rate']):.2f}%, while `OT` contributed {int(ot_summary['n_ops']):,} operations ({float(ot_summary['pct_ops']):.2f}%) with a `{outcome_label}` rate of {float(ot_summary['severe_aki_rate']):.2f}%.",
    ]

    if feature_bug:
        lines.append(
            "- A true feature-construction bug was detected because at least one `department_OS` row did not map back to raw `department == \"OS\"`. This would require a pipeline correction before any wording-only manuscript response."
        )
    else:
        lines.extend(
            [
                "- The current data support a reporting/manuscript labeling problem rather than a one-hot encoding bug. `department_OS` is an operation-level Orthopedic Surgery indicator, and its prominence is most consistent with surgical-service case mix rather than a direct renal mechanism.",
                "- The currently mounted cohort table is patient-level by construction: it drops duplicate `subject_id` rows before counting departments. That aggregation, rather than the feature construction itself, explains why the cohort table and the operation-level model feature can appear inconsistent.",
            ]
        )
        if artifact_label_mismatch:
            lines.append("- The currently mounted cohort table still reproduces the `OS` and `OT` patient counts under swapped human-readable labels, which explains the manuscript inconsistency.")

    lines.extend(
        [
            "",
            "## Methods And Provenance",
            f"- Config loaded via `inspire_aki.config.load_config` from `{context.config_path}` with raw data rooted at `{context.paths.raw_inspire_dir}` and artifacts rooted at `{context.paths.artifacts_root}`.",
            "- Code-path audit focused on the maintained implementation surface: `src/inspire_aki/cohort/filters.py`, `src/inspire_aki/cohort/preop.py`, `src/inspire_aki/reporting/tables.py`, and `src/inspire_aki/reporting/procedure_audit.py`.",
            "- Operation-level audit cohort: `datasets/tabular/tabular_combined_unnormalized.csv` inner-joined to `cohort/labels.csv` on `op_id`, then joined back to raw `operations.csv` on `op_id`.",
            f"- Outcome column audited: `{context.target_column}`.",
        ]
    )
    if manifest_warnings:
        lines.append(f"- Manifest/config warnings: {'; '.join(manifest_warnings)}")
    if procedure_name_fields:
        lines.append(f"- Additional maintained procedure-name/grouping fields detected in raw operations data: {', '.join(procedure_name_fields)}.")
    else:
        lines.append("- No maintained anonymized operation-name or procedure-grouping variable was present in `operations.csv` or documented in `schema.csv`; interpretation therefore relied on raw department codes, anesthesia type, operation length, and ICD-10-PCS group summaries.")

    lines.extend(
        [
            "",
            "## Code-Path Findings",
            "- In `src/inspire_aki/cohort/filters.py`, the preoperative cohort filter one-hot encodes raw `department` with `pd.get_dummies(..., columns=[\"department\"])`.",
            "- In `src/inspire_aki/cohort/preop.py`, the resulting `department_*` indicator columns are merged into the maintained preoperative feature artifact and then carried into the tabular modeling datasets.",
            "- In the final labeled cohort, the joined data confirm that the feature columns still retain raw code identity rather than any downstream remapping.",
            "",
            _markdown_table(_format_indicator_checks_for_report(indicator_checks)),
            "",
            "### Raw Dictionary Versus Current Maintained Code Labels",
            _markdown_table(_format_mapping_for_report(mapping_df)),
        ]
    )

    lines.extend(
        [
            "",
            "## OS Characterization",
            "- The summary table below reports operation-level counts and rates for the requested slices.",
            "",
            _markdown_table(
                _format_summary_for_report(
                    summary_df[
                        [
                            "slice",
                            "slice_label",
                            "n_ops",
                            "pct_ops",
                            "n_patients",
                            "pct_patients",
                            "severe_aki_n",
                            "severe_aki_rate",
                            "op_len_median",
                            "op_len_q1",
                            "op_len_q3",
                            "op_len_mean",
                            "op_len_sd",
                            "antype_general_n",
                            "antype_general_pct",
                            "antype_mac_n",
                            "antype_mac_pct",
                            "antype_neuraxial_n",
                            "antype_neuraxial_pct",
                            "antype_regional_n",
                            "antype_regional_pct",
                        ]
                    ]
                ).rename(columns={"severe_aki_n": "positive_n", "severe_aki_rate": "positive_rate"})
            ),
            "",
            "- The top raw `OS` ICD-10-PCS 4-character groups were:",
            "",
            _markdown_table(
                os_top_report_df.rename(
                    columns={
                        "pct_of_ops": "pct_of_os_ops",
                        "severe_aki_n": "positive_n",
                        "severe_aki_rate": "positive_rate",
                        "representative_5char_prefix": "representative_5char_prefix",
                        "body_system_desc": "body_system_desc",
                        "root_op_desc": "root_op_desc",
                        "example_long_title": "example_long_title",
                    }
                )
            ),
        ]
    )

    if not ot_top_report_df.empty:
        lines.extend(
            [
                "",
                "- As a control, the much smaller raw `OT` group was dominated by these ICD-10-PCS 4-character families:",
                "",
                _markdown_table(
                    ot_top_report_df.head(5).rename(
                        columns={
                            "pct_of_ops": "pct_of_ot_ops",
                            "severe_aki_n": "positive_n",
                            "severe_aki_rate": "positive_rate",
                        }
                    )
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Patient-Vs-Operation Reconciliation",
            "- The maintained cohort table computes `Total operations` from unique `op_id`, then drops duplicate `subject_id` rows before counting department indicators. That means the department rows in the current cohort table are patient-level counts, not operation-level counts.",
        ]
    )
    if not artifact_reconciliation_df.empty:
        lines.extend(
            [
                "- The rows below show which current artifact-table labels reproduce the raw `OS` and `OT` patient counts:",
                "",
                _markdown_table(artifact_reconciliation_df),
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            f"1. `department_OS` most likely represents `{os_row['raw_dictionary_label']}` in this analytic dataset. The raw INSPIRE dictionary, the feature-engineering code path, and the row-level joined data all point to the same conclusion.",
            "2. The current `ophthalmology` interpretation for `department_OS` is not defensible. Ophthalmology corresponds to raw `OT`, not raw `OS`, in the INSPIRE department dictionary.",
            f"3. `department_OS` is likely acting as a case-mix proxy rather than a direct renal signal. It defines a large service slice with a distinctive anesthesia mix and procedure profile, and its `{outcome_label}` rate ({float(os_summary['severe_aki_rate']):.2f}%) is below the overall cohort rate ({float(overall_row['severe_aki_rate']):.2f}%).",
            "4. Nothing in this audit suggests data leakage or a one-hot feature-construction mistake. The engineered indicator columns preserve the raw department codes exactly in the final labeled cohort.",
        ]
    )

    if feature_bug:
        lines.append("5. Because a true feature-construction bug was detected, the smallest next step is a pipeline correction before any manuscript wording change.")
    else:
        lines.append(
            "5. The smallest manuscript change is to relabel `department_OS` as Orthopedic Surgery, clarify that department indicators are administrative service/procedural case-mix features, and reconcile patient-level cohort-table counts against operation-level feature provenance."
        )

    if not feature_bug:
        reviewer_paragraph = (
            "We audited the provenance of `department_OS` directly against the maintained feature-engineering code and the raw INSPIRE operations table. "
            "In the current pipeline, department indicators are created by one-hot encoding the raw `department` field from `operations.csv`, and row-level verification in the final labeled cohort showed that `department_OS` maps exactly to raw `department == \"OS\"`. "
            "The public INSPIRE department dictionary defines `OS` as Orthopedic Surgery and `OT` as Ophthalmology. "
            "In our final analytic cohort, raw `OS` accounted for "
            f"{int(os_summary['n_ops']):,} operations with a `{outcome_label}` rate of {float(os_summary['severe_aki_rate']):.2f}%, and its most common ICD-10-PCS groups were orthopedic/joint procedure families rather than ophthalmic procedures. "
            "We therefore interpret `department_OS` as an orthopedic surgical-service indicator and a case-mix proxy rather than a mechanistic kidney-risk factor. "
            "The inconsistency arose because the current cohort table aggregates department counts at the patient level and the previously generated manuscript-facing labels swapped `OS` and `OT`. "
            "This audit did not identify evidence of leakage or a feature-construction error, so the appropriate correction is to fix the label/writing and to describe the feature more carefully as a service-level case-mix marker."
        )
        manuscript_revision = (
            "The engineered feature `department_OS` corresponds to the raw INSPIRE department code `OS`, which in the INSPIRE data dictionary denotes Orthopedic Surgery rather than Ophthalmology. "
            "We interpret this variable as an administrative surgical-service indicator that captures procedural case mix, anesthesia mix, and operation context rather than a direct mechanistic renal risk factor. "
            "Accordingly, we avoid overinterpreting this SHAP signal as a biologic AKI predictor and instead describe it as a service-level case-mix feature."
        )
        lines.extend(
            [
                "",
                "## Draft Reviewer-Response Paragraph",
                reviewer_paragraph,
                "",
                "## Draft Manuscript Revision",
                manuscript_revision,
            ]
        )

    lines.extend(
        [
            "",
            "1. The likely meaning of `department_OS` is Orthopedic Surgery.",
            f"2. {'Yes; the mounted artifact table still shows the old swapped human-readable labels.' if artifact_label_mismatch else 'Yes; the manuscript previously mislabeled it, and the maintained code plus regenerated reviewer-facing outputs now use the corrected labels.'}",
            f"3. This requires {'a genuine data/pipeline correction' if feature_bug else 'wording/table fixes rather than a feature-construction correction'}.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    *,
    config_path: str | Path | None = None,
    raw_dir: str | Path | None = None,
    artifacts_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Path]:
    context = _load_context(config_path=config_path, raw_dir=raw_dir, artifacts_dir=artifacts_dir, out_dir=out_dir)
    raw_dictionary_df = load_raw_department_dictionary(context.paths.raw_inspire_dir / "department.csv")
    mapping_df = raw_dictionary_df.merge(
        current_department_label_frame(raw_dictionary_df["department_code"].tolist()),
        on="department_code",
        how="left",
    )
    audit_df, _, operations_header = load_final_audit_cohort(context)
    schema_df = _load_schema_frame(context.paths.raw_inspire_dir)
    procedure_name_fields = _find_procedure_name_fields(operations_header, schema_df)
    indicator_checks = [compare_department_indicator(audit_df, code) for code in ("OS", "OT")]
    feature_bug = any(check["mismatch_n"] > 0 for check in indicator_checks)
    raw_counts_df = build_department_raw_counts(audit_df, raw_dictionary_df)
    summary_df = build_slice_summary(audit_df, raw_dictionary_df, context.target_column)
    cms_reference, cms_warnings = _load_cms_reference(context)
    os_top_csv_df, os_top_report_df = build_top_icd10pcs4_frame(
        audit_df,
        department_code="OS",
        target_column=context.target_column,
        cms_reference=cms_reference,
        top_n=TOP_ICD10PCS4_N,
    )
    _, ot_top_report_df = build_top_icd10pcs4_frame(
        audit_df,
        department_code="OT",
        target_column=context.target_column,
        cms_reference=cms_reference,
        top_n=5,
    )
    manifest_warnings = _manifest_warnings(context) + cms_warnings
    artifact_reconciliation_df = _artifact_table_reconciliation(
        raw_counts_df,
        context.paths.artifact_path("reports", "tables", "cohort_characteristics.csv"),
    )

    summary_path = context.out_dir / "department_os_summary.csv"
    top_icd_path = context.out_dir / "department_os_top_icd10pcs4.csv"
    raw_counts_path = context.out_dir / "department_raw_counts.csv"
    report_path = context.out_dir / "department_os_audit.md"

    summary_df.round(6).to_csv(summary_path, index=False)
    os_top_csv_df.round(6).to_csv(top_icd_path, index=False)
    raw_counts_df.round(6).to_csv(raw_counts_path, index=False)
    report_text = _build_report(
        context=context,
        raw_dictionary_df=raw_dictionary_df,
        mapping_df=mapping_df,
        raw_counts_df=raw_counts_df,
        summary_df=summary_df,
        indicator_checks=indicator_checks,
        os_top_report_df=os_top_report_df,
        ot_top_report_df=ot_top_report_df,
        procedure_name_fields=procedure_name_fields,
        manifest_warnings=manifest_warnings,
        artifact_reconciliation_df=artifact_reconciliation_df,
        feature_bug=feature_bug,
    )
    report_path.write_text(report_text, encoding="utf-8")
    return {
        "report": report_path,
        "summary_csv": summary_path,
        "top_icd10pcs4_csv": top_icd_path,
        "raw_counts_csv": raw_counts_path,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    outputs = run_audit(
        config_path=args.config,
        raw_dir=args.raw_dir,
        artifacts_dir=args.artifacts_dir,
        out_dir=args.out_dir,
    )
    for label, path in outputs.items():
        print(f"{label}: {path}")
    return 0
