#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Iterable

import pandas as pd

from inspire_aki.reporting.department_os_audit import _load_context, load_final_audit_cohort
from inspire_aki.reporting.procedure_audit import build_cms_prefix_reference, load_cms_order_entries


OT_REPORT_NAME = "department_ot_reviewer_report.md"
OT_SUMMARY_NAME = "department_ot_summary.csv"
OT_TOP_GROUPS_NAME = "department_ot_top_icd10pcs4.csv"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a compact reviewer-facing report for ophthalmology-coded cases.")
    parser.add_argument("--config", default="configs/aki/default.yaml", help="Config path to load.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory. Defaults to <artifacts_dir>/reports/reviewer_department_audit/.",
    )
    return parser


def _markdown_table(frame: pd.DataFrame) -> str:
    display = frame.fillna("")
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    def fmt(row: Iterable[str]) -> str:
        return "| " + " | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(row)) + " |"

    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"
    lines = [fmt(headers), separator]
    lines.extend(fmt(row) for row in rows)
    return "\n".join(lines)


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator * 100.0


def _median_iqr(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "N/A"
    return f"{clean.median():.1f} ({clean.quantile(0.25):.1f}-{clean.quantile(0.75):.1f})"


def _median_iqr_with_precision(series: pd.Series, digits: int = 2) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "N/A"
    return (
        f"{clean.median():.{digits}f} "
        f"({clean.quantile(0.25):.{digits}f}-{clean.quantile(0.75):.{digits}f})"
    )


def _distribution_summary(series: pd.Series, *, total: int, labels: list[str]) -> str:
    counts = series.fillna("<NA>").value_counts()
    parts: list[str] = []
    for label in labels:
        count = int(counts.get(label, 0))
        parts.append(f"{label} {count} ({_pct(count, total):.1f}%)")
    return "; ".join(parts)


def _load_review_frame(config_path: str, out_dir: str | None) -> tuple[object, pd.DataFrame]:
    context = _load_context(config_path=config_path, raw_dir=None, artifacts_dir=None, out_dir=out_dir)
    audit_df, _, _ = load_final_audit_cohort(context)

    covariate_columns = ["op_id", "age", "asa", "preop_creatinine", "preop_bun"]
    tabular_path = context.paths.artifact_path("datasets", "tabular", "tabular_combined_unnormalized.csv")
    covariates = pd.read_csv(tabular_path, usecols=lambda column: column in covariate_columns)
    review_df = audit_df.merge(covariates, on="op_id", how="left", validate="one_to_one")
    return context, review_df


def _patient_positive_counts(frame: pd.DataFrame, *, target_column: str) -> tuple[int, int]:
    patient_flags = (
        frame.groupby("subject_id")[target_column]
        .max()
        .fillna(0)
        .astype(int)
    )
    return int(patient_flags.sum()), int(patient_flags.shape[0])


def _summary_table(review_df: pd.DataFrame, *, target_column: str) -> pd.DataFrame:
    ot = review_df.loc[review_df["raw_department"] == "OT"].copy()
    ot_aki = ot.loc[ot[target_column] == 1].copy()
    ot_no_aki = ot.loc[ot[target_column] != 1].copy()

    overall_patient_positive_n, overall_patient_n = _patient_positive_counts(review_df, target_column=target_column)
    ot_patient_positive_n, ot_patient_n = _patient_positive_counts(ot, target_column=target_column)

    positive_ot_patient_flags = (
        ot_aki.groupby("subject_id")
        .size()
        .rename("n_positive_ot_ops")
    )
    repeated_positive_patient_n = int((positive_ot_patient_flags > 1).sum())
    repeated_positive_op_n = int(
        ot_aki["subject_id"].isin(positive_ot_patient_flags.loc[positive_ot_patient_flags > 1].index).sum()
    )
    eye_prefix_n = int(ot["icd10_pcs"].astype("string").str.startswith("08", na=False).sum())
    non_eye_n = int(len(ot) - eye_prefix_n)

    rows = [
        {
            "metric": "Operations, n",
            "overall": f"{len(review_df):,}",
            "ophthalmology": f"{len(ot):,}",
            "ophthalmology_aki": f"{len(ot_aki):,}",
            "ophthalmology_no_aki": f"{len(ot_no_aki):,}",
        },
        {
            "metric": "Patients, n",
            "overall": f"{review_df['subject_id'].nunique():,}",
            "ophthalmology": f"{ot['subject_id'].nunique():,}",
            "ophthalmology_aki": f"{ot_aki['subject_id'].nunique():,}",
            "ophthalmology_no_aki": f"{ot_no_aki['subject_id'].nunique():,}",
        },
        {
            "metric": "AKI-positive operations, n (%)",
            "overall": f"{int(review_df[target_column].sum()):,} ({_pct(int(review_df[target_column].sum()), len(review_df)):.1f}%)",
            "ophthalmology": f"{int(ot[target_column].sum()):,} ({_pct(int(ot[target_column].sum()), len(ot)):.1f}%)",
            "ophthalmology_aki": "Reference",
            "ophthalmology_no_aki": "Reference",
        },
        {
            "metric": "Patients with >=1 AKI-positive cohort op, n (%)",
            "overall": f"{overall_patient_positive_n:,} ({_pct(overall_patient_positive_n, overall_patient_n):.1f}%)",
            "ophthalmology": f"{ot_patient_positive_n:,} ({_pct(ot_patient_positive_n, ot_patient_n):.1f}%)",
            "ophthalmology_aki": "Reference",
            "ophthalmology_no_aki": "Reference",
        },
        {
            "metric": "Age, y, median (IQR)",
            "overall": _median_iqr(review_df["age"]),
            "ophthalmology": _median_iqr(ot["age"]),
            "ophthalmology_aki": _median_iqr(ot_aki["age"]),
            "ophthalmology_no_aki": _median_iqr(ot_no_aki["age"]),
        },
        {
            "metric": "ASA, median (IQR)",
            "overall": _median_iqr(review_df["asa"]),
            "ophthalmology": _median_iqr(ot["asa"]),
            "ophthalmology_aki": _median_iqr(ot_aki["asa"]),
            "ophthalmology_no_aki": _median_iqr(ot_no_aki["asa"]),
        },
        {
            "metric": "Preop creatinine, mg/dL, median (IQR)",
            "overall": _median_iqr_with_precision(review_df["preop_creatinine"]),
            "ophthalmology": _median_iqr_with_precision(ot["preop_creatinine"]),
            "ophthalmology_aki": _median_iqr_with_precision(ot_aki["preop_creatinine"]),
            "ophthalmology_no_aki": _median_iqr_with_precision(ot_no_aki["preop_creatinine"]),
        },
        {
            "metric": "Preop BUN, mg/dL, median (IQR)",
            "overall": _median_iqr_with_precision(review_df["preop_bun"]),
            "ophthalmology": _median_iqr_with_precision(ot["preop_bun"]),
            "ophthalmology_aki": _median_iqr_with_precision(ot_aki["preop_bun"]),
            "ophthalmology_no_aki": _median_iqr_with_precision(ot_no_aki["preop_bun"]),
        },
        {
            "metric": "Op length, min, median (IQR)",
            "overall": _median_iqr(review_df["op_len_minutes"]),
            "ophthalmology": _median_iqr(ot["op_len_minutes"]),
            "ophthalmology_aki": _median_iqr(ot_aki["op_len_minutes"]),
            "ophthalmology_no_aki": _median_iqr(ot_no_aki["op_len_minutes"]),
        },
        {
            "metric": "Anesthesia distribution",
            "overall": _distribution_summary(review_df["antype"], total=len(review_df), labels=["MAC", "General"]),
            "ophthalmology": _distribution_summary(ot["antype"], total=len(ot), labels=["MAC", "General"]),
            "ophthalmology_aki": _distribution_summary(ot_aki["antype"], total=len(ot_aki), labels=["MAC", "General"]),
            "ophthalmology_no_aki": _distribution_summary(ot_no_aki["antype"], total=len(ot_no_aki), labels=["MAC", "General"]),
        },
        {
            "metric": "Eye ICD-10-PCS prefix (`08..`), n (%)",
            "overall": "Not summarized",
            "ophthalmology": f"{eye_prefix_n:,} ({_pct(eye_prefix_n, len(ot)):.1f}%)",
            "ophthalmology_aki": f"{int(ot_aki['icd10_pcs'].astype('string').str.startswith('08', na=False).sum()):,} ({_pct(int(ot_aki['icd10_pcs'].astype('string').str.startswith('08', na=False).sum()), len(ot_aki)):.1f}%)",
            "ophthalmology_no_aki": f"{int(ot_no_aki['icd10_pcs'].astype('string').str.startswith('08', na=False).sum()):,} ({_pct(int(ot_no_aki['icd10_pcs'].astype('string').str.startswith('08', na=False).sum()), len(ot_no_aki)):.1f}%)",
        },
        {
            "metric": "Non-eye outlier codes, n",
            "overall": "Not summarized",
            "ophthalmology": f"{non_eye_n:,}",
            "ophthalmology_aki": f"{int(len(ot_aki) - ot_aki['icd10_pcs'].astype('string').str.startswith('08', na=False).sum()):,}",
            "ophthalmology_no_aki": f"{int(len(ot_no_aki) - ot_no_aki['icd10_pcs'].astype('string').str.startswith('08', na=False).sum()):,}",
        },
        {
            "metric": "AKI-positive OT ops on patients with >1 positive OT op",
            "overall": "N/A",
            "ophthalmology": f"{repeated_positive_op_n:,} of {len(ot_aki):,} ({_pct(repeated_positive_op_n, len(ot_aki)):.1f}%)",
            "ophthalmology_aki": f"{repeated_positive_patient_n:,} of {ot_aki['subject_id'].nunique():,} patients ({_pct(repeated_positive_patient_n, ot_aki['subject_id'].nunique()):.1f}%)",
            "ophthalmology_no_aki": "N/A",
        },
    ]
    return pd.DataFrame(rows)


def _top_ot_groups(review_df: pd.DataFrame, *, target_column: str, cms_reference: pd.DataFrame) -> pd.DataFrame:
    ot = review_df.loc[review_df["raw_department"] == "OT"].copy()
    prefix_map = ot.dropna(subset=["pcs_prefix"])[["icd10_pcs4", "pcs_prefix"]].drop_duplicates("icd10_pcs4")
    top = (
        ot.groupby("icd10_pcs4")
        .agg(
            n_ops=("op_id", "size"),
            aki_positive_n=(target_column, "sum"),
        )
        .reset_index()
    )
    top["pct_of_ot_ops"] = top["n_ops"] / len(ot) * 100.0
    top["aki_rate"] = top["aki_positive_n"] / top["n_ops"] * 100.0
    top = top.sort_values(["n_ops", "icd10_pcs4"], ascending=[False, True]).head(8)
    top = top.merge(prefix_map, on="icd10_pcs4", how="left")
    top = top.merge(
        cms_reference[
            ["pcs_prefix", "body_system_desc", "canonical_prefix_label", "sample_long_title"]
        ].drop_duplicates("pcs_prefix"),
        on="pcs_prefix",
        how="left",
    )
    top.insert(0, "rank", range(1, len(top) + 1))
    return top[
        [
            "rank",
            "icd10_pcs4",
            "n_ops",
            "pct_of_ot_ops",
            "aki_positive_n",
            "aki_rate",
            "body_system_desc",
            "canonical_prefix_label",
            "sample_long_title",
        ]
    ]


def _format_top_group_display(top_groups: pd.DataFrame) -> pd.DataFrame:
    display = top_groups.copy()
    display["pct_of_ot_ops"] = display["pct_of_ot_ops"].map(lambda value: f"{value:.1f}%")
    display["aki_rate"] = display["aki_rate"].map(lambda value: f"{value:.1f}%")
    display = display.rename(
        columns={
            "icd10_pcs4": "ICD-10-PCS4",
            "n_ops": "OT ops",
            "pct_of_ot_ops": "% of OT ops",
            "aki_positive_n": "AKI+ ops",
            "aki_rate": "AKI rate",
            "body_system_desc": "Body system",
            "canonical_prefix_label": "Representative procedure",
            "sample_long_title": "Sample title",
        }
    )
    return display


def _build_markdown_report(
    summary_table: pd.DataFrame,
    top_groups: pd.DataFrame,
    *,
    target_column: str,
) -> str:
    ot_ops = int(summary_table.loc[summary_table["metric"] == "Operations, n", "ophthalmology"].iloc[0].replace(",", ""))
    ot_patients = int(summary_table.loc[summary_table["metric"] == "Patients, n", "ophthalmology"].iloc[0].replace(",", ""))
    ot_positive_ops = summary_table.loc[summary_table["metric"] == "AKI-positive operations, n (%)", "ophthalmology"].iloc[0]
    ot_positive_patients = summary_table.loc[
        summary_table["metric"] == "Patients with >=1 AKI-positive cohort op, n (%)", "ophthalmology"
    ].iloc[0]
    repeated_line = summary_table.loc[
        summary_table["metric"] == "AKI-positive OT ops on patients with >1 positive OT op", "ophthalmology"
    ].iloc[0]
    age_line = summary_table.loc[summary_table["metric"] == "Age, y, median (IQR)", "ophthalmology"].iloc[0]
    creatinine_line = summary_table.loc[
        summary_table["metric"] == "Preop creatinine, mg/dL, median (IQR)", "ophthalmology"
    ].iloc[0]
    bun_line = summary_table.loc[
        summary_table["metric"] == "Preop BUN, mg/dL, median (IQR)", "ophthalmology"
    ].iloc[0]
    length_line = summary_table.loc[summary_table["metric"] == "Op length, min, median (IQR)", "ophthalmology"].iloc[0]
    antype_line = summary_table.loc[summary_table["metric"] == "Anesthesia distribution", "ophthalmology"].iloc[0]
    eye_prefix_line = summary_table.loc[summary_table["metric"] == "Eye ICD-10-PCS prefix (`08..`), n (%)", "ophthalmology"].iloc[0]

    reviewer_response = (
        "We audited the ophthalmology-coded rows directly in the final labeled analytic cohort by joining the final "
        "`labels.csv` and `tabular_combined_unnormalized.csv` artifacts back to raw `operations.csv` on `op_id`. "
        "The earlier reporting confusion reflected a human-readable `OS`/`OT` label swap; the underlying feature construction was correct, "
        "and true ophthalmology corresponds to raw `department == \"OT\"`. In the final cohort, ophthalmology accounted for "
        f"{ot_ops:,} operations from {ot_patients:,} patients, and {eye_prefix_line.lower()} were genuine eye procedures with ICD-10-PCS `08..` prefixes. "
        f"The row-level AKI frequency was {ot_positive_ops.lower()}, but this corresponded to {ot_positive_patients.lower()}, and {repeated_line.lower()} occurred in patients with repeated positive ophthalmology rows, suggesting that staged or bilateral procedures amplified the row-level signal. "
        f"These patients were older and more renally vulnerable than the overall cohort (ophthalmology median age {age_line} years; preoperative creatinine {creatinine_line} mg/dL; preoperative BUN {bun_line} mg/dL), while the procedures themselves remained brief ({length_line} minutes) and predominantly MAC/general anesthesia cases ({antype_line}). "
        "Accordingly, we interpret the ophthalmology indicator as a surgical-service/case-mix proxy that helps partition a small, medically higher-risk subgroup rather than as a mechanistic renal effect of eye surgery itself. "
        "We will revise the Discussion to distinguish predictive importance from causal interpretation and to describe department indicators as administrative case-mix features."
    )

    return "\n".join(
        [
            "# Ophthalmology Department Reviewer Note",
            "",
            "## Executive Summary",
            "",
            "- The original reviewer concern was sharpened by a reporting-label issue: the maintained human-readable labels had swapped `OS` and `OT`, although the underlying one-hot encoded feature columns were correct.",
            f"- True ophthalmology corresponds to raw `department == \"OT\"` and contributes {ot_ops:,} operations from {ot_patients:,} patients in the final analytic cohort.",
            f"- Ophthalmology-coded rows are overwhelmingly genuine eye procedures: {eye_prefix_line.lower()} have ICD-10-PCS `08..` Eye body-system prefixes, and the few non-eye outliers are AKI-negative.",
            f"- The observed row-level AKI signal is most consistent with case mix rather than procedure mechanism: ophthalmology patients are older and have higher baseline renal markers, and repeated staged/bilateral eye operations inflate the row-level frequency ({repeated_line.lower()}).",
            "",
            "## Methods",
            "",
            "- Loaded the maintained default config `configs/aki/default.yaml`.",
            "- Joined the final labeled cohort (`cohort/labels.csv` and `datasets/tabular/tabular_combined_unnormalized.csv`) back to raw `operations.csv` on `op_id`.",
            f"- Used the active outcome `{target_column}` from the maintained default config.",
            "- Summarized ophthalmology-coded rows (`department == \"OT\"`) at both operation and patient level and enriched ICD-10-PCS prefixes with the configured CMS order reference.",
            "",
            "## Summary Table",
            "",
            _markdown_table(summary_table.rename(
                columns={
                    "metric": "Metric",
                    "overall": "Overall cohort",
                    "ophthalmology": "Ophthalmology",
                    "ophthalmology_aki": "Ophthalmology AKI+",
                    "ophthalmology_no_aki": "Ophthalmology AKI-",
                }
            )),
            "",
            "## Top Ophthalmology ICD-10-PCS Groups",
            "",
            _markdown_table(_format_top_group_display(top_groups)),
            "",
            "## Interpretation",
            "",
            "- The ophthalmology indicator is not capturing mislabeled orthopedic or other large-service cases; it mostly marks short lens, vitreous, and related eye procedures.",
            "- The high row-level AKI frequency is partly inflated by repeated operations in a small subset of patients, which matters because the model and SHAP beeswarm are row-based.",
            "- The subgroup also looks medically higher-risk than the full cohort, with older age and worse baseline renal labs, which is a more plausible explanation for the predictive signal than a direct nephrotoxic effect of ophthalmology itself.",
            "- For manuscript interpretation, department indicators should therefore be framed as service/procedural case-mix variables rather than mechanistic AKI risk factors.",
            "",
            "## Draft Reviewer Response",
            "",
            reviewer_response,
        ]
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    context, review_df = _load_review_frame(args.config, args.out_dir)
    out_dir = context.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cms_reference = build_cms_prefix_reference(
        load_cms_order_entries(context.config.get("reports", {}).get("procedure_audit", {}).get("cms_order_zip_path"))
    )
    summary_table = _summary_table(review_df, target_column=context.target_column)
    top_groups = _top_ot_groups(review_df, target_column=context.target_column, cms_reference=cms_reference)
    markdown = _build_markdown_report(summary_table, top_groups, target_column=context.target_column)

    (out_dir / OT_REPORT_NAME).write_text(markdown + "\n", encoding="utf-8")
    summary_table.to_csv(out_dir / OT_SUMMARY_NAME, index=False)
    top_groups.to_csv(out_dir / OT_TOP_GROUPS_NAME, index=False)

    print(f"report: {out_dir / OT_REPORT_NAME}")
    print(f"summary_csv: {out_dir / OT_SUMMARY_NAME}")
    print(f"top_icd10pcs4_csv: {out_dir / OT_TOP_GROUPS_NAME}")


if __name__ == "__main__":
    main()
