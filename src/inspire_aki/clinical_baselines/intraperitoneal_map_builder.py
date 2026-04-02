from __future__ import annotations

import argparse
import json
import logging
import re
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

import pandas as pd

from inspire_aki.config import DEFAULT_CONFIG_PATH, REPO_ROOT, _deep_merge, _normalize_config, load_yaml


LOGGER = logging.getLogger(__name__)
POSITIVE_APPROACHES = {"0", "4", "F"}
POSITIVE_CATEGORIES = {"APPY", "BILI", "CHOL", "COLO", "GAST", "HYST", "OVRY", "REC", "SB", "SPLE", "XLAP", "LTP"}
COHORT_EXCLUDED_CATEGORIES = {"CSEC", "KTP"}
EXPERT_REVIEW_OVERRIDES: dict[str, tuple[int, str]] = {
    "0UB90": (1, "expert_review_override_1_open_uterine_excision"),
    "0UB94": (1, "expert_review_override_1_laparoscopic_uterine_excision"),
    "0UQ90": (1, "expert_review_override_1_open_uterine_repair"),
    "0UQ94": (1, "expert_review_override_1_laparoscopic_uterine_repair"),
    "0UJD0": (1, "expert_review_override_1_open_uterus_cervix_inspection"),
    "0UJD4": (1, "expert_review_override_1_laparoscopic_uterus_cervix_inspection"),
    "0US94": (1, "expert_review_override_1_laparoscopic_uterine_reposition"),
    "0U990": (1, "expert_review_override_1_open_uterine_drainage"),
    "0T160": (0, "expert_review_override_0_ureter_to_bladder_diversion"),
    "0T164": (0, "expert_review_override_0_ureter_to_bladder_diversion"),
    "0T170": (0, "expert_review_override_0_ureter_to_bladder_diversion"),
    "0T174": (0, "expert_review_override_0_ureter_to_bladder_diversion"),
    "0T1B0": (0, "expert_review_override_0_bladder_diversion"),
    "0T1B4": (0, "expert_review_override_0_bladder_diversion"),
    "04100": (0, "expert_review_override_0_abdominal_vascular_bypass"),
    "04C50": (0, "expert_review_override_0_mesenteric_artery_vascular"),
    "0FYG0": (1, "expert_review_override_1_open_pancreas_transplant"),
    "0DY60": (1, "expert_review_override_1_open_stomach_transplant"),
    "001U0": (0, "expert_review_override_0_mixed_truncation_artifact"),
}
INTRAPERITONEAL_KEYWORDS = (
    "appendix",
    "stomach",
    "gastric",
    "small intestine",
    "jejun",
    "ile",
    "colon",
    "cecum",
    "sigmoid",
    "rect",
    "gallbladder",
    "bile duct",
    "liver",
    "pancre",
    "spleen",
    "ovary",
    "uterus",
    "peritone",
    "omentum",
    "mesenter",
    "laparotomy",
)
CDC_WORKBOOK_URL = "https://www.cdc.gov/nhsn/xls/icd10-pcs-pcm-nhsn-opc.xlsx"
CMS_ORDER_URL = "https://www.cms.gov/files/zip/april-1-2026-icd-10-pcs-order-file-long-abbreviated-titles.zip"
CMS_TABLES_URL = "https://www.cms.gov/files/zip/april-1-2026-icd-10-pcs-code-tables-index.zip"
CDC_DEFAULT_PATH = Path("external/cdc_nhsn/icd10-pcs-pcm-nhsn-opc-2026.xlsx")
CMS_ORDER_DEFAULT_PATH = Path("external/cms_icd10pcs/april-1-2026-icd10pcs-order.zip")
CMS_TABLES_DEFAULT_PATH = Path("external/cms_icd10pcs/april-1-2026-icd10pcs-code-tables-index.zip")

_XML_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "doc_rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkg_rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


@dataclass(frozen=True)
class IntraperitonealBuildOutputs:
    mapping_df: pd.DataFrame
    unmatched_df: pd.DataFrame
    qc: dict[str, Any]
    build_report: str


def download_if_missing(url: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        LOGGER.info("Downloading %s -> %s", url, path)
        urllib.request.urlretrieve(url, path)
    return path


def download_default_resources(include_code_tables: bool = False) -> dict[str, Path]:
    paths = {
        "cdc_workbook": download_if_missing(CDC_WORKBOOK_URL, CDC_DEFAULT_PATH),
        "cms_order_zip": download_if_missing(CMS_ORDER_URL, CMS_ORDER_DEFAULT_PATH),
    }
    if include_code_tables:
        paths["cms_tables_zip"] = download_if_missing(CMS_TABLES_URL, CMS_TABLES_DEFAULT_PATH)
    return paths


def normalize_icd10_pcs_codes(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.upper()
    normalized = normalized.where(series.notna(), pd.NA)
    return normalized


def retained_operations_from_raw(operations_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    cohort_cfg = config["cohort"]
    retained = operations_df.copy()
    retained["age"] = pd.to_numeric(retained["age"], errors="coerce")
    retained["asa"] = pd.to_numeric(retained["asa"], errors="coerce")
    retained["height"] = pd.to_numeric(retained["height"], errors="coerce")
    retained["weight"] = pd.to_numeric(retained["weight"], errors="coerce")
    retained["opstart_time"] = pd.to_numeric(retained["opstart_time"], errors="coerce")
    retained["opend_time"] = pd.to_numeric(retained["opend_time"], errors="coerce")
    retained = retained[retained["asa"] < float(cohort_cfg["max_asa_exclusive"])]
    retained = retained[retained["age"] >= float(cohort_cfg["min_age"])]
    retained = retained.dropna(subset=["opstart_time", "opend_time"])
    if bool(cohort_cfg.get("require_positive_op_len", True)):
        retained = retained[(retained["opend_time"] - retained["opstart_time"]) > 0]
    if bool(cohort_cfg.get("require_height_weight", True)):
        retained = retained.dropna(subset=["height", "weight"])
        retained = retained[(retained["height"] != 0) & (retained["weight"] != 0)]
    exclude_antype = cohort_cfg.get("exclude_antype", [])
    if exclude_antype:
        retained = retained[~retained["antype"].isin(exclude_antype)]
    department_include = cohort_cfg.get("department_include", [])
    if department_include:
        retained = retained[retained["department"].isin(department_include)]
    department_exclude = cohort_cfg.get("department_exclude", [])
    if department_exclude:
        retained = retained[~retained["department"].isin(department_exclude)]
    include_prefixes = tuple(cohort_cfg.get("include_icd10_prefixes", []))
    if include_prefixes:
        retained = retained[normalize_icd10_pcs_codes(retained["icd10_pcs"]).str.startswith(include_prefixes, na=False)]
    exclude_prefixes = tuple(cohort_cfg.get("exclude_icd10_prefixes", []))
    if exclude_prefixes:
        retained = retained[~normalize_icd10_pcs_codes(retained["icd10_pcs"]).str.startswith(exclude_prefixes, na=False)]
    return retained.reset_index(drop=True)


def observed_code_counts(operations_df: pd.DataFrame) -> pd.DataFrame:
    normalized = normalize_icd10_pcs_codes(operations_df["icd10_pcs"])
    invalid_lengths = normalized.loc[normalized.str.len() != 5].dropna().unique().tolist()
    if invalid_lengths:
        raise ValueError(f"Observed retained ICD-10-PCS codes must all be 5 characters. Invalid values: {invalid_lengths[:10]}")
    invalid_tokens = sorted({code for code in normalized.dropna().unique().tolist() if ("O" in code) or ("I" in code)})
    if invalid_tokens:
        raise ValueError(f"Observed retained ICD-10-PCS codes contain invalid ICD-10-PCS letters O/I: {invalid_tokens[:10]}")
    if normalized.str.contains(r"[^0-9A-Z]", regex=True, na=False).any():
        raise ValueError("Observed retained ICD-10-PCS codes contain non-alphanumeric characters.")
    return (
        pd.DataFrame({"icd10_pcs_5char": normalized})
        .value_counts()
        .rename("n_ops")
        .reset_index()
        .sort_values(["n_ops", "icd10_pcs_5char"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )


def _xlsx_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    shared: list[str] = []
    for si in root.findall("main:si", _XML_NS):
        texts = [node.text or "" for node in si.iterfind(".//main:t", _XML_NS)]
        shared.append("".join(texts))
    return shared


def _xlsx_sheet_targets(zf: zipfile.ZipFile) -> dict[str, str]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels.findall("pkg_rel:Relationship", _XML_NS)
    }
    targets: dict[str, str] = {}
    for sheet in workbook.find("main:sheets", _XML_NS):
        name = sheet.attrib["name"]
        rel_id = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
        targets[name] = "xl/" + rel_map[rel_id]
    return targets


def _xlsx_sheet_rows(xlsx_path: Path, sheet_name: str) -> list[list[str]]:
    with zipfile.ZipFile(xlsx_path) as zf:
        shared = _xlsx_shared_strings(zf)
        targets = _xlsx_sheet_targets(zf)
        if sheet_name not in targets:
            raise KeyError(f"Sheet '{sheet_name}' not found in {xlsx_path}. Available sheets: {sorted(targets)}")
        root = ET.fromstring(zf.read(targets[sheet_name]))
        rows: list[list[str]] = []

        def _cell_value(cell: ET.Element) -> str:
            value_node = cell.find("main:v", _XML_NS)
            cell_type = cell.attrib.get("t")
            if cell_type == "inlineStr":
                inline = cell.find("main:is", _XML_NS)
                return "".join(node.text or "" for node in inline.iterfind(".//main:t", _XML_NS)) if inline is not None else ""
            if value_node is None:
                return ""
            raw = value_node.text or ""
            if cell_type == "s":
                return shared[int(raw)]
            return raw

        for row in root.findall(".//main:sheetData/main:row", _XML_NS):
            rows.append([_cell_value(cell) for cell in row.findall("main:c", _XML_NS)])
    return rows


def load_cdc_workbook_rows(xlsx_path: Path) -> pd.DataFrame:
    sheet_rows = _xlsx_sheet_rows(xlsx_path, "ALL 2026 ICD-10-PCS CODES")
    if not sheet_rows:
        raise ValueError(f"No rows found in CDC workbook: {xlsx_path}")
    header = [str(cell).strip() for cell in sheet_rows[0]]
    normalized_header = [re.sub(r"[^a-z0-9]+", "_", cell.lower()).strip("_") for cell in header]
    records = []
    for row in sheet_rows[1:]:
        padded = row + [""] * (len(normalized_header) - len(row))
        record = dict(zip(normalized_header, padded, strict=False))
        if any(str(value).strip() for value in record.values()):
            records.append(record)
    frame = pd.DataFrame(records)
    category_column = "procedure_code_category"
    pcs_column = "icd_10_pcs_codes"
    if category_column not in frame.columns or pcs_column not in frame.columns:
        raise ValueError(f"CDC workbook did not expose expected columns. Found: {frame.columns.tolist()}")
    frame = frame.rename(columns={category_column: "nhsn_category", pcs_column: "pcs7"})
    frame["nhsn_category"] = frame["nhsn_category"].astype(str).str.strip().str.upper()
    frame["pcs7"] = normalize_icd10_pcs_codes(frame["pcs7"]).str[:7]
    frame = frame[frame["pcs7"].str.len() == 7].copy()
    return frame[["nhsn_category", "pcs7"]].reset_index(drop=True)


def collapse_cdc_map_to_code5(cdc_df: pd.DataFrame) -> pd.DataFrame:
    mapping = cdc_df.copy()
    mapping["code5"] = mapping["pcs7"].str[:5]
    mapping["approach"] = mapping["pcs7"].str[4]
    mapping["intraperitoneal_proxy"] = (
        mapping["nhsn_category"].isin(POSITIVE_CATEGORIES) & mapping["approach"].isin(POSITIVE_APPROACHES)
    ).astype(int)
    conflicts = mapping.groupby("code5")["intraperitoneal_proxy"].nunique()
    conflicting_codes = conflicts.loc[conflicts > 1].index.tolist()
    if conflicting_codes:
        raise ValueError(
            "Conflicting CDC/NHSN-derived intraperitoneal labels across 7-character expansions for 5-character codes: "
            f"{conflicting_codes[:10]}"
        )
    collapsed = (
        mapping.sort_values(["code5", "nhsn_category", "pcs7"], kind="stable")
        .groupby("code5", as_index=False)
        .agg(
            approach=("approach", "first"),
            nhsn_category=("nhsn_category", "first"),
            intraperitoneal_proxy=("intraperitoneal_proxy", "first"),
        )
        .rename(columns={"code5": "icd10_pcs_5char"})
    )
    collapsed["source"] = "cdc_nhsn_primary"
    collapsed["rationale"] = collapsed.apply(
        lambda row: f"{row['nhsn_category']} + approach {row['approach']} -> {int(row['intraperitoneal_proxy'])}",
        axis=1,
    )
    return collapsed


def load_cms_order_titles(cms_order_zip: Path) -> pd.DataFrame:
    with zipfile.ZipFile(cms_order_zip) as zf:
        candidate_names = [name for name in zf.namelist() if name.lower().startswith("icd10pcs_order_") and name.lower().endswith(".txt")]
        if not candidate_names:
            raise ValueError(f"Could not locate icd10pcs_order_*.txt in {cms_order_zip}")
        lines = zf.read(candidate_names[0]).decode("latin1").splitlines()
    pattern = re.compile(r"^\s*\d+\s+([0-9A-Z]{3,7})\s+\S\s+(.*?)\s{2,}(.*?)\s*$")
    rows = []
    for line in lines:
        match = pattern.match(line)
        if not match:
            continue
        code = match.group(1).strip().upper()
        if len(code) != 7:
            continue
        rows.append(
            {
                "pcs7": code,
                "code5": code[:5],
                "short_title": match.group(2).strip(),
                "long_title": match.group(3).strip(),
            }
        )
    frame = pd.DataFrame(rows).drop_duplicates(subset=["pcs7"], keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"Could not parse any 7-character ICD-10-PCS titles from {cms_order_zip}")
    return frame


def _positive_looking_title(title: str) -> bool:
    normalized = title.lower()
    return any(keyword in normalized for keyword in INTRAPERITONEAL_KEYWORDS)


def _expert_override_frame(observed_codes: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code in sorted(set(observed_codes) & set(EXPERT_REVIEW_OVERRIDES)):
        label, rationale = EXPERT_REVIEW_OVERRIDES[code]
        rows.append(
            {
                "icd10_pcs_5char": code,
                "approach": code[4],
                "nhsn_category": pd.NA,
                "intraperitoneal_proxy": int(label),
                "source": "expert_review_override",
                "rationale": rationale,
            }
        )
    return pd.DataFrame(rows)


def build_intraperitoneal_proxy_outputs(
    *,
    observed_counts_df: pd.DataFrame,
    cdc_collapsed_df: pd.DataFrame,
    cms_titles_df: pd.DataFrame,
    positive_keyword_fraction_threshold: float = 0.001,
) -> IntraperitonealBuildOutputs:
    observed = observed_counts_df.copy()
    merged = observed.merge(cdc_collapsed_df, on="icd10_pcs_5char", how="left")
    expert_overrides = _expert_override_frame(observed["icd10_pcs_5char"].astype(str).tolist())
    matched_categories = set(merged["nhsn_category"].dropna().astype(str))
    leaked_categories = sorted(matched_categories & COHORT_EXCLUDED_CATEGORIES)
    if leaked_categories:
        raise ValueError(
            "Retained observed ICD-10-PCS codes mapped to cohort-excluded NHSN categories. "
            f"Observed categories: {leaked_categories}"
        )

    unmatched = merged.loc[merged["intraperitoneal_proxy"].isna(), ["icd10_pcs_5char", "n_ops"]].copy()
    if not expert_overrides.empty:
        unmatched = unmatched[~unmatched["icd10_pcs_5char"].isin(expert_overrides["icd10_pcs_5char"])].copy()
    cms_lookup = cms_titles_df.groupby("code5", as_index=False).agg(
        long_titles=("long_title", lambda series: sorted(set(series))),
        short_titles=("short_title", lambda series: sorted(set(series))),
    )
    unmatched = unmatched.merge(cms_lookup, left_on="icd10_pcs_5char", right_on="code5", how="left").drop(columns=["code5"])
    unmatched["long_titles"] = unmatched["long_titles"].map(lambda value: value if isinstance(value, list) else [])
    unmatched["short_titles"] = unmatched["short_titles"].map(lambda value: value if isinstance(value, list) else [])
    unmatched["title_text"] = unmatched.apply(
        lambda row: " | ".join(list(row["long_titles"]) + list(row["short_titles"])),
        axis=1,
    )
    unmatched["approach"] = unmatched["icd10_pcs_5char"].str[4]
    unmatched["positive_keyword_match"] = unmatched["title_text"].map(_positive_looking_title) & unmatched["approach"].isin(POSITIVE_APPROACHES)
    total_ops = int(observed["n_ops"].sum())
    positive_keyword_ops = int(unmatched.loc[unmatched["positive_keyword_match"], "n_ops"].sum())
    if total_ops > 0 and (positive_keyword_ops / total_ops) > positive_keyword_fraction_threshold:
        raise ValueError(
            "Unmatched observed ICD-10-PCS codes with positive intraperitoneal keywords exceeded the configured coverage threshold. "
            f"Coverage={positive_keyword_ops / total_ops:.4%}"
        )

    defaults = unmatched[["icd10_pcs_5char"]].copy()
    defaults["approach"] = defaults["icd10_pcs_5char"].str[4]
    defaults["nhsn_category"] = pd.NA
    defaults["intraperitoneal_proxy"] = 0
    defaults["source"] = "default_zero_unmatched"
    defaults["rationale"] = "unmatched_default_0"

    mapping = pd.concat([cdc_collapsed_df, expert_overrides, defaults], ignore_index=True)
    mapping = mapping[mapping["icd10_pcs_5char"].isin(observed["icd10_pcs_5char"])].copy()
    source_priority = {"expert_review_override": 0, "cdc_nhsn_primary": 1, "default_zero_unmatched": 2}
    mapping["source_priority"] = mapping["source"].map(source_priority).fillna(99).astype(int)
    mapping = mapping.sort_values(["icd10_pcs_5char", "source_priority", "source"], kind="stable").drop_duplicates(
        subset=["icd10_pcs_5char"], keep="first"
    )
    mapping = mapping.merge(observed, on="icd10_pcs_5char", how="left")

    top_positive = (
        mapping.loc[mapping["intraperitoneal_proxy"] == 1, ["icd10_pcs_5char", "n_ops"]]
        .sort_values(["n_ops", "icd10_pcs_5char"], ascending=[False, True], kind="stable")
        .head(25)
        .to_dict(orient="records")
    )
    top_unmatched = unmatched.sort_values(["n_ops", "icd10_pcs_5char"], ascending=[False, True], kind="stable").head(25)
    positive_category_counts = Counter(
        mapping.loc[mapping["intraperitoneal_proxy"] == 1, "nhsn_category"].dropna().astype(str).tolist()
    )

    qc = {
        "n_observed_codes": int(len(observed)),
        "n_observed_operations": total_ops,
        "n_positive_operations": int(mapping.loc[mapping["intraperitoneal_proxy"] == 1, "n_ops"].sum()),
        "positive_operation_fraction": 0.0 if total_ops == 0 else float(mapping.loc[mapping["intraperitoneal_proxy"] == 1, "n_ops"].sum() / total_ops),
        "n_unmatched_codes": int(len(unmatched)),
        "n_unmatched_operations": int(unmatched["n_ops"].sum()) if not unmatched.empty else 0,
        "positive_keyword_unmatched_operations": positive_keyword_ops,
        "n_expert_override_codes": int(len(expert_overrides)),
        "n_expert_override_operations": int(
            observed.loc[observed["icd10_pcs_5char"].isin(expert_overrides["icd10_pcs_5char"]), "n_ops"].sum()
        )
        if not expert_overrides.empty
        else 0,
        "top_positive_codes": top_positive,
        "top_unmatched_codes": top_unmatched[["icd10_pcs_5char", "n_ops", "positive_keyword_match"]].to_dict(orient="records"),
        "positive_category_distribution": dict(sorted(positive_category_counts.items())),
    }

    report_lines = [
        "# Intraperitoneal Proxy Build Report",
        "",
        f"- Observed retained codes: {qc['n_observed_codes']}",
        f"- Retained operations covered: {qc['n_observed_operations']}",
        f"- Positive operations: {qc['n_positive_operations']} ({qc['positive_operation_fraction']:.2%})",
        f"- Expert override codes applied: {qc['n_expert_override_codes']}",
        f"- Operations covered by expert overrides: {qc['n_expert_override_operations']}",
        f"- Unmatched observed codes: {qc['n_unmatched_codes']}",
        f"- Unmatched operations defaulted to 0: {qc['n_unmatched_operations']}",
        f"- Positive-keyword unmatched operations: {qc['positive_keyword_unmatched_operations']}",
        "",
        "## Positive category distribution",
    ]
    if qc["positive_category_distribution"]:
        report_lines.extend(f"- {category}: {count}" for category, count in qc["positive_category_distribution"].items())
    else:
        report_lines.append("- None")
    report_lines.extend(["", "## Top positive codes"])
    if top_positive:
        report_lines.extend(f"- {row['icd10_pcs_5char']}: {row['n_ops']}" for row in top_positive)
    else:
        report_lines.append("- None")
    report_lines.extend(["", "## Top unmatched codes before default-zero"])
    if not top_unmatched.empty:
        report_lines.extend(
            f"- {row.icd10_pcs_5char}: {int(row.n_ops)} (positive_keyword_match={bool(row.positive_keyword_match)})"
            for row in top_unmatched.itertuples(index=False)
        )
    else:
        report_lines.append("- None")

    mapping = mapping[
        ["icd10_pcs_5char", "approach", "nhsn_category", "intraperitoneal_proxy", "source", "rationale"]
    ].copy()
    mapping["intraperitoneal_proxy"] = mapping["intraperitoneal_proxy"].astype(int)
    unmatched = unmatched.drop(columns=["title_text"])
    return IntraperitonealBuildOutputs(
        mapping_df=mapping.reset_index(drop=True),
        unmatched_df=unmatched.reset_index(drop=True),
        qc=qc,
        build_report="\n".join(report_lines) + "\n",
    )


def build_intraperitoneal_proxy_map(
    *,
    config_path: str | Path,
    operations_path: Path | None,
    cdc_workbook: Path,
    cms_order_zip: Path,
    output_path: Path,
    report_dir: Path,
    qc_path: Path | None = None,
) -> IntraperitonealBuildOutputs:
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    override_cfg = load_yaml(config_path)
    base_cfg = load_yaml(DEFAULT_CONFIG_PATH)
    config = _normalize_config(_deep_merge(base_cfg, override_cfg))
    if str(config["study"]["outcome_key"]) != "aki":
        raise ValueError("Intraperitoneal proxy map builder should be run against an AKI cohort config.")
    raw_operations_path = operations_path or (Path(config["paths"]["raw_inspire_dir"]) / "operations.csv")
    operations_df = pd.read_csv(
        raw_operations_path,
        usecols=[
            "op_id",
            "age",
            "asa",
            "height",
            "weight",
            "opstart_time",
            "opend_time",
            "antype",
            "department",
            "icd10_pcs",
        ],
    )
    retained_ops = retained_operations_from_raw(operations_df, config)
    observed = observed_code_counts(retained_ops)
    cdc_rows = load_cdc_workbook_rows(cdc_workbook)
    observed_codes = set(observed["icd10_pcs_5char"].astype(str))
    cdc_rows = cdc_rows[cdc_rows["pcs7"].str[:5].isin(observed_codes)].copy()
    cdc_collapsed = collapse_cdc_map_to_code5(cdc_rows)
    cms_titles = load_cms_order_titles(cms_order_zip)
    outputs = build_intraperitoneal_proxy_outputs(
        observed_counts_df=observed,
        cdc_collapsed_df=cdc_collapsed,
        cms_titles_df=cms_titles,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    outputs.mapping_df.to_csv(output_path, index=False)
    outputs.unmatched_df.to_csv(report_dir / "intraperitoneal_proxy_unmatched_codes.csv", index=False)
    (report_dir / "intraperitoneal_proxy_build_report.md").write_text(outputs.build_report, encoding="utf-8")
    qc_output_path = qc_path or (report_dir / "intraperitoneal_proxy_qc.json")
    qc_output_path.write_text(json.dumps(outputs.qc, indent=2, sort_keys=True), encoding="utf-8")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the committed 5-character intraperitoneal proxy map for adapted GS-AKI.")
    parser.add_argument("--config", default="configs/aki/default.yaml", help="Config path used to identify the retained AKI cohort.")
    parser.add_argument("--operations-path", default=None, help="Optional override for the raw operations.csv path.")
    parser.add_argument("--cdc-workbook", default=str(CDC_DEFAULT_PATH), help="CDC/NHSN workbook path.")
    parser.add_argument("--cms-order-zip", default=str(CMS_ORDER_DEFAULT_PATH), help="CMS order-file zip path.")
    parser.add_argument(
        "--output",
        default="configs/clinical_baselines/intraperitoneal_proxy_map_5char.csv",
        help="Committed mapping CSV output path.",
    )
    parser.add_argument(
        "--report-dir",
        default="reports",
        help="Directory for unmatched-code audit CSV, build report, and QC JSON.",
    )
    parser.add_argument("--download", action="store_true", help="Download the default CDC/CMS resources if they are not already cached.")
    parser.add_argument("--include-code-tables", action="store_true", help="Also download the optional CMS code tables zip.")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cdc_path = Path(args.cdc_workbook)
    cms_order_path = Path(args.cms_order_zip)
    if args.download:
        downloaded = download_default_resources(include_code_tables=bool(args.include_code_tables))
        cdc_path = downloaded["cdc_workbook"]
        cms_order_path = downloaded["cms_order_zip"]

    outputs = build_intraperitoneal_proxy_map(
        config_path=args.config,
        operations_path=Path(args.operations_path) if args.operations_path else None,
        cdc_workbook=cdc_path,
        cms_order_zip=cms_order_path,
        output_path=Path(args.output),
        report_dir=Path(args.report_dir),
    )
    LOGGER.info(
        "Wrote %s mapped codes (%s positives).",
        len(outputs.mapping_df),
        int(outputs.mapping_df["intraperitoneal_proxy"].sum()),
    )
    return 0
