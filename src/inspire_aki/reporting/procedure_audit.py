from __future__ import annotations

from dataclasses import dataclass
import re
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from inspire_aki.config import REPO_ROOT
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.csv import read_csv_optimized
from inspire_aki.reporting.department_labels import DEPARTMENT_LABELS as SHARED_DEPARTMENT_LABELS
from inspire_aki.reporting.rendering import ColumnSpec, TableSection, TableSpec, write_table_outputs


_DEPARTMENT_LABELS = SHARED_DEPARTMENT_LABELS
_ORDER_LINE_RE = re.compile(
    r"^\s*(?P<seq>\d+)\s+(?P<code>[0-9A-HJ-NP-Z]{3,7})\s+(?P<valid>[01])\s+(?P<short>.+?)\s{2,}(?P<long>.+?)\s*$"
)
_APPROACH_SUFFIX_RE = re.compile(r", (?P<approach>(?:[^,]+, )?[^,]+ Approach)(?:, Diagnostic)?$")
_TRAILING_CONNECTOR_TOKENS = {"from", "with", "using", "to", "via", "by", "for"}
_CARDIAC_KEYWORDS = (
    "coronary",
    "heart",
    "cardiac",
    "pericard",
    "aortic valve",
    "mitral valve",
    "tricuspid valve",
    "pulmonary valve",
    "pulmonic valve",
    "great vessel",
)
_CHEST_RELATED_KEYWORDS = (
    "lung",
    "pleura",
    "pleural",
    "bronch",
    "trachea",
    "mediastin",
    "diaphragm",
    "rib",
    "sternum",
    "sternal",
    "thorax",
    "thoracic",
    "chest wall",
    "chest",
)
_THORACIC_DESCRIBE_KEYWORDS = (
    "thymus",
    "thymectomy",
    "esophag",
    "sternum",
    "mediastin",
    "larynx",
    "trachea",
    "bronch",
    "pleura",
    "lung",
    "diaphragm",
    "thorax",
    "thoracic",
    "chest wall",
    "chest",
)
_VASCULAR_KEYWORDS = (
    "aorta",
    "artery",
    "vein",
    "carotid",
    "innominate",
    "femoral",
)
_VASCULAR_BODY_SYSTEMS = {"Lower Arteries", "Upper Arteries", "Upper Veins", "Lower Veins"}
_SUMMARY_CLASS_ORDER = (
    "cardiac_exclude",
    "thoracic_keep",
    "thoracic_or_chest_related_noncardiac",
    "ct_service_noncardiac_keep",
    "vascular_noncardiac_describe",
    "manual_review",
    "other_operation",
)
_SUMMARY_CLASS_NOTES = {
    "cardiac_exclude": "Official Heart and Great Vessels cardiac prefix or explicit cardiac title family.",
    "thoracic_keep": "CTS respiratory-system prefix without CPB discordance.",
    "thoracic_or_chest_related_noncardiac": "CTS thoracic, mediastinal, or foregut noncardiac family outside the respiratory prefix family.",
    "ct_service_noncardiac_keep": "CTS-labeled but clearly noncardiac nonthoracic family without CPB discordance.",
    "vascular_noncardiac_describe": "CTS vascular family without CPB; retain as noncardiac and describe explicitly.",
    "manual_review": "CTS residual gray-zone family resolved by the final operation-level keep versus exclude policy.",
    "other_operation": "Non-CTS or otherwise not part of the dedicated cardiothoracic reviewer subset.",
}
_CLINICIAN_REVIEW_BUCKET_ORDER = (
    "cpb_positive_aortic_or_vascular",
    "respiratory_plus_cpb",
    "unresolved_prefix_or_title",
    "other_cpb_discordant_nonvascular_nonrespiratory",
    "other_prefix_level_review",
)
_CLINICIAN_REVIEW_BUCKET_NOTES = {
    "cpb_positive_aortic_or_vascular": "CPB-supported aortic or vascular family requiring direct clinician review.",
    "respiratory_plus_cpb": "Respiratory family with CPB timing populated; likely atypical thoracic support case.",
    "unresolved_prefix_or_title": "Prefix/title could not be resolved from the CMS order reference and needs direct review.",
    "other_cpb_discordant_nonvascular_nonrespiratory": "Nonvascular nonrespiratory family with CPB timing populated; likely service-label or documentation discordance.",
    "other_prefix_level_review": "Residual CTS prefix-level ambiguity after rule-based classification.",
}
_DEFAULT_BENIGN_NEIGHBOR_KEYWORDS = (
    "scalp skin",
    "sternum",
    "chest wall",
    "external ear",
    "conjunctiva",
    "breast",
)
_AUTO_RETAIN_AUDIT_CLASSES = {
    "thoracic_keep",
    "thoracic_or_chest_related_noncardiac",
    "ct_service_noncardiac_keep",
    "vascular_noncardiac_describe",
    "other_operation",
}
_AUTO_EXCLUDE_AUDIT_CLASSES = {"cardiac_exclude"}
_FINAL_NONCARDIAC_ACTIONS = {"retain", "exclude"}


@dataclass(frozen=True)
class ProcedureAuditPolicy:
    ct_department_code: str
    cardiac_prefixes: tuple[str, ...]
    thoracic_prefixes: tuple[str, ...]
    manual_review_if_cpb: bool


@dataclass(frozen=True)
class ProcedureAuditResolutionPolicy:
    exclude_audit_classes: frozenset[str]
    exclude_manual_review_buckets: frozenset[str]
    retain_unresolved_cpb_negative_with_benign_neighbor: bool
    benign_neighbor_keywords: tuple[str, ...]


@dataclass(frozen=True)
class AuditClassification:
    audit_class: str
    reason_code: str
    note: str
    clinician_review_bucket: str | None = None
    exclusion_reason: str = ""


def _procedure_audit_config(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("reports", {}).get("procedure_audit", {})


def _procedure_audit_resolution_config(config: dict[str, Any]) -> dict[str, Any]:
    resolution_cfg = config.get("cohort", {}).get("procedure_audit_resolution", {})
    return resolution_cfg if isinstance(resolution_cfg, dict) else {}


def _build_procedure_audit_policy(config: dict[str, Any]) -> ProcedureAuditPolicy:
    audit_cfg = _procedure_audit_config(config)
    return ProcedureAuditPolicy(
        ct_department_code=str(audit_cfg.get("ct_department_code", "CTS")).strip().upper(),
        cardiac_prefixes=tuple(str(prefix).strip().upper() for prefix in audit_cfg.get("definite_cardiac_prefixes", ["02"])),
        thoracic_prefixes=tuple(
            str(prefix).strip().upper() for prefix in audit_cfg.get("definite_thoracic_prefixes", ["0B"])
        ),
        manual_review_if_cpb=bool(audit_cfg.get("manual_review_if_cpb_discordant", True)),
    )


def _build_resolution_policy(config: dict[str, Any]) -> ProcedureAuditResolutionPolicy:
    resolution_cfg = _procedure_audit_resolution_config(config)
    return ProcedureAuditResolutionPolicy(
        exclude_audit_classes=frozenset(
            str(value).strip()
            for value in resolution_cfg.get("exclude_audit_classes", list(_AUTO_EXCLUDE_AUDIT_CLASSES))
            if str(value).strip()
        ),
        exclude_manual_review_buckets=frozenset(
            str(value).strip()
            for value in resolution_cfg.get("exclude_manual_review_buckets", [])
            if str(value).strip()
        ),
        retain_unresolved_cpb_negative_with_benign_neighbor=bool(
            resolution_cfg.get("retain_unresolved_cpb_negative_with_benign_neighbor", True)
        ),
        benign_neighbor_keywords=tuple(
            str(value).strip().lower()
            for value in resolution_cfg.get("benign_neighbor_keywords", list(_DEFAULT_BENIGN_NEIGHBOR_KEYWORDS))
            if str(value).strip()
        ),
    )


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _normalize_string_series(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip()
    return normalized.where(series.notna(), pd.NA)


def normalize_op_ids(series: pd.Series) -> pd.Series:
    normalized = _normalize_string_series(series)
    numeric = pd.to_numeric(normalized, errors="coerce")
    numeric_mask = numeric.notna()
    normalized.loc[numeric_mask] = numeric.loc[numeric_mask].astype("Int64").astype("string")
    normalized = normalized.str.lstrip("\ufeff")
    return normalized.where(normalized.notna() & normalized.ne(""), pd.NA)


def normalize_icd10_pcs(series: pd.Series) -> pd.Series:
    normalized = _normalize_string_series(series).str.upper()
    return normalized.where(normalized.notna() & normalized.ne(""), pd.NA)


def _nonempty_mask(series: pd.Series) -> pd.Series:
    normalized = _normalize_string_series(series)
    return normalized.notna() & normalized.ne("")


def _pct(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(count / total) * 100:.2f}%"


def _department_label(code: Any) -> str:
    normalized = str(code).strip().upper()
    return _DEPARTMENT_LABELS.get(normalized, normalized)


def _split_group_title(title: str) -> tuple[str | pd.NA, str | pd.NA]:
    cleaned = " ".join(str(title).split())
    if ", " not in cleaned:
        return cleaned or pd.NA, pd.NA
    body_system_desc, root_op_desc = cleaned.rsplit(", ", 1)
    return body_system_desc.strip() or pd.NA, root_op_desc.strip() or pd.NA


def _strip_approach_suffix(title: str) -> str:
    cleaned = " ".join(str(title).split())
    return _APPROACH_SUFFIX_RE.sub("", cleaned).strip()


def _collapse_approach_desc(titles: list[str]) -> str | pd.NA:
    approaches = sorted(
        {
            match.group("approach").strip()
            for title in titles
            if (match := _APPROACH_SUFFIX_RE.search(" ".join(str(title).split()))) is not None
        }
    )
    return approaches[0] if len(approaches) == 1 else pd.NA


def _collapse_prefix_titles(titles: list[str]) -> str | pd.NA:
    cleaned_titles = sorted({_strip_approach_suffix(title) for title in titles if str(title).strip()})
    if not cleaned_titles:
        return pd.NA
    if len(cleaned_titles) == 1:
        return cleaned_titles[0]

    token_lists = [title.split() for title in cleaned_titles]
    common_tokens: list[str] = []
    for idx, token in enumerate(token_lists[0]):
        if all(len(tokens) > idx and tokens[idx] == token for tokens in token_lists[1:]):
            common_tokens.append(token)
        else:
            break
    while common_tokens and common_tokens[-1].rstrip(",").lower() in _TRAILING_CONNECTOR_TOKENS:
        common_tokens.pop()
    collapsed = " ".join(common_tokens).strip().rstrip(",;:-")
    if len(collapsed) >= 8:
        return collapsed

    first = cleaned_titles[0]
    lowered = first.lower()
    for marker in (" with ", " from ", " using ", " to ", " via ", " by ", " for "):
        idx = lowered.find(marker)
        if idx > 0:
            trimmed = first[:idx].strip().rstrip(",;:-")
            if len(trimmed) >= 8:
                return trimmed
    return first


def load_cms_order_entries(zip_path: str | Path) -> pd.DataFrame:
    resolved_path = _resolve_repo_path(zip_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"CMS ICD-10-PCS order zip was not found: {resolved_path}")

    with zipfile.ZipFile(resolved_path) as archive:
        members = archive.namelist()
        order_member = next(
            (name for name in members if name.lower().startswith("icd10pcs_order_") and name.lower().endswith(".txt")),
            None,
        )
        if order_member is None:
            raise ValueError(f"CMS order zip did not include an icd10pcs_order text file: {resolved_path}")
        lines = archive.read(order_member).decode("utf-8", errors="replace").splitlines()

    rows: list[dict[str, Any]] = []
    for line in lines:
        match = _ORDER_LINE_RE.match(line)
        if match is None:
            continue
        rows.append(
            {
                "seq": int(match.group("seq")),
                "code": match.group("code").strip(),
                "valid_flag": int(match.group("valid")),
                "short_title": " ".join(match.group("short").split()),
                "long_title": " ".join(match.group("long").split()),
            }
        )
    if not rows:
        raise ValueError(f"CMS order zip did not yield any parsable order rows: {resolved_path}")
    return pd.DataFrame(rows).sort_values("seq", kind="stable").reset_index(drop=True)


def build_cms_prefix_reference(order_entries: pd.DataFrame) -> pd.DataFrame:
    if order_entries.empty:
        return pd.DataFrame(
            columns=[
                "pcs_prefix",
                "group_code",
                "body_system_desc",
                "root_op_desc",
                "canonical_prefix_label",
                "approach_desc",
                "sample_long_title",
            ]
        )

    group_entries = order_entries.loc[order_entries["code"].astype(str).str.len() == 3, ["code", "long_title"]].copy()
    split_titles = group_entries["long_title"].map(_split_group_title)
    group_entries["body_system_desc"] = split_titles.map(lambda value: value[0])
    group_entries["root_op_desc"] = split_titles.map(lambda value: value[1])
    group_lookup = (
        group_entries.drop_duplicates(subset=["code"], keep="first")
        .set_index("code")[["body_system_desc", "root_op_desc"]]
        .to_dict(orient="index")
    )

    code_entries = order_entries.loc[order_entries["code"].astype(str).str.len() == 7, ["code", "long_title"]].copy()
    code_entries["pcs_prefix"] = code_entries["code"].astype(str).str[:5]
    code_entries["group_code"] = code_entries["code"].astype(str).str[:3]

    rows: list[dict[str, Any]] = []
    for pcs_prefix, prefix_df in code_entries.groupby("pcs_prefix", sort=True):
        titles = prefix_df["long_title"].astype(str).dropna().tolist()
        group_info = group_lookup.get(str(prefix_df["group_code"].iat[0]), {})
        sample_long_title = sorted(set(titles))[0] if titles else pd.NA
        rows.append(
            {
                "pcs_prefix": pcs_prefix,
                "group_code": str(prefix_df["group_code"].iat[0]),
                "body_system_desc": group_info.get("body_system_desc", pd.NA),
                "root_op_desc": group_info.get("root_op_desc", pd.NA),
                "canonical_prefix_label": _collapse_prefix_titles(titles),
                "approach_desc": _collapse_approach_desc(titles),
                "sample_long_title": sample_long_title,
            }
        )
    return pd.DataFrame(rows).sort_values("pcs_prefix", kind="stable").reset_index(drop=True)


def _pcs_length_counts(audit_df: pd.DataFrame) -> pd.Series:
    return audit_df["pcs_prefix"].dropna().astype(str).str.len().value_counts().sort_index()


def _validate_current_pcs_contract(audit_df: pd.DataFrame) -> None:
    length_counts = _pcs_length_counts(audit_df)
    observed_lengths = sorted(int(length) for length in length_counts.index.tolist())
    if observed_lengths != [5]:
        raise ValueError(
            "Procedure audit currently expects a 5-character icd10_pcs contract. "
            f"Observed length counts: {length_counts.to_dict()}"
        )


def _keyword_match(text: Any, keywords: tuple[str, ...]) -> bool:
    normalized = str(text or "").strip().lower()
    if not normalized:
        return False
    return any(keyword in normalized for keyword in keywords)


def _row_title_text(row: pd.Series) -> str:
    parts = [row.get("canonical_prefix_label"), row.get("sample_long_title"), row.get("body_system_desc")]
    return " | ".join(str(part) for part in parts if pd.notna(part) and str(part).strip())


def _is_vascular_row(row: pd.Series, *, title_text: str) -> bool:
    body_system_desc = str(row.get("body_system_desc", "")).strip()
    return body_system_desc in _VASCULAR_BODY_SYSTEMS or _keyword_match(title_text, _VASCULAR_KEYWORDS)


def _classification(
    audit_class: str,
    reason_code: str,
    note: str,
    *,
    clinician_review_bucket: str | None = None,
    exclusion_reason: str = "",
) -> AuditClassification:
    return AuditClassification(
        audit_class=audit_class,
        reason_code=reason_code,
        note=note,
        clinician_review_bucket=clinician_review_bucket,
        exclusion_reason=exclusion_reason,
    )


def _classify_row(row: pd.Series, *, policy: ProcedureAuditPolicy) -> AuditClassification:
    department_code = str(row.get("department", "")).strip().upper()
    is_ct_department = department_code == policy.ct_department_code
    pcs_prefix = str(row.get("pcs_prefix", "")).strip().upper()
    cpb_present = bool(row.get("cpb_present", False))
    title_text = _row_title_text(row)
    has_cardiac_terms = _keyword_match(title_text, _CARDIAC_KEYWORDS)
    has_thoracic_terms = _keyword_match(title_text, _THORACIC_DESCRIBE_KEYWORDS) or _keyword_match(title_text, _CHEST_RELATED_KEYWORDS)
    is_vascular = _is_vascular_row(row, title_text=title_text)

    if pcs_prefix and any(pcs_prefix.startswith(prefix) for prefix in policy.cardiac_prefixes):
        return _classification(
            "cardiac_exclude",
            "official_heart_and_great_vessels_prefix",
            "Official 02... Heart and Great Vessels family.",
            exclusion_reason="official_heart_and_great_vessels_prefix",
        )
    if pcs_prefix and has_cardiac_terms and not any(pcs_prefix.startswith(prefix) for prefix in policy.thoracic_prefixes):
        return _classification(
            "cardiac_exclude",
            "explicit_cardiac_title_match",
            "Decoded title matched a cardiac procedure family.",
            exclusion_reason="explicit_cardiac_title_match",
        )

    if is_ct_department:
        if not pcs_prefix or len(pcs_prefix) != 5:
            return _classification(
                "manual_review",
                "missing_or_malformed_prefix",
                "CTS case with malformed or missing ICD-10-PCS prefix.",
                clinician_review_bucket="unresolved_prefix_or_title",
            )
        if any(pcs_prefix.startswith(prefix) for prefix in policy.thoracic_prefixes):
            if cpb_present and policy.manual_review_if_cpb:
                return _classification(
                    "manual_review",
                    "respiratory_plus_cpb",
                    "CTS respiratory-system prefix with CPB timing populated.",
                    clinician_review_bucket="respiratory_plus_cpb",
                )
            return _classification(
                "thoracic_keep",
                "respiratory_prefix_without_cpb_discordance",
                "CTS respiratory-system prefix without CPB discordance.",
            )
        if pd.isna(row.get("canonical_prefix_label")) and pd.isna(row.get("sample_long_title")):
            return _classification(
                "manual_review",
                "missing_cms_title",
                "CTS case missing a resolvable CMS order-file title.",
                clinician_review_bucket="unresolved_prefix_or_title",
            )
        if has_thoracic_terms:
            if cpb_present and policy.manual_review_if_cpb:
                return _classification(
                    "manual_review",
                    "thoracic_plus_cpb",
                    "CTS thoracic or mediastinal family with CPB timing populated.",
                    clinician_review_bucket="other_cpb_discordant_nonvascular_nonrespiratory",
                )
            return _classification(
                "thoracic_or_chest_related_noncardiac",
                "thoracic_or_mediastinal_family_without_cpb_discordance",
                "CTS thoracic, mediastinal, or foregut noncardiac title family.",
            )
        if is_vascular:
            if cpb_present and policy.manual_review_if_cpb:
                return _classification(
                    "manual_review",
                    "vascular_plus_cpb",
                    "CTS aortic or vascular family with CPB timing populated.",
                    clinician_review_bucket="cpb_positive_aortic_or_vascular",
                )
            return _classification(
                "vascular_noncardiac_describe",
                "vascular_family_without_cpb_discordance",
                "CTS vascular family without CPB; retain but describe explicitly.",
            )
        if cpb_present and policy.manual_review_if_cpb:
            return _classification(
                "manual_review",
                "other_plus_cpb",
                "CTS non-02/non-0B case with CPB timing populated.",
                clinician_review_bucket="other_cpb_discordant_nonvascular_nonrespiratory",
            )
        return _classification(
            "ct_service_noncardiac_keep",
            "clearly_noncardiac_ct_service_family_without_cpb_discordance",
            "CTS-labeled but clearly noncardiac nonthoracic title family without CPB discordance.",
        )

    return _classification(
        "other_operation",
        "outside_ct_reviewer_subset",
        "Outside the dedicated CTS reviewer subset.",
    )


def _has_benign_neighbor_support(label: Any, *, keywords: tuple[str, ...]) -> bool:
    return _keyword_match(label, keywords)


def _final_noncardiac_resolution(
    row: pd.Series,
    *,
    policy: ProcedureAuditResolutionPolicy,
) -> tuple[str, str]:
    audit_class = str(row.get("audit_class", "")).strip()
    if audit_class in policy.exclude_audit_classes:
        return "exclude", "Audit class is excluded from the final fully defined noncardiac cohort."
    if audit_class in _AUTO_RETAIN_AUDIT_CLASSES:
        return "retain", "Audit class is retained in the final fully defined noncardiac cohort."
    if audit_class != "manual_review":
        return "exclude", "Residual audit state was not explicitly retained; excluded conservatively."

    bucket = str(row.get("clinician_review_bucket", "")).strip()
    if bucket in policy.exclude_manual_review_buckets:
        return "exclude", "Clinician-review bucket is excluded from the final fully defined noncardiac cohort."
    if bucket == "unresolved_prefix_or_title":
        if bool(row.get("cpb_present", False)):
            return "exclude", "Unresolved prefix with CPB support is excluded from the final fully defined noncardiac cohort."
        if policy.retain_unresolved_cpb_negative_with_benign_neighbor and _has_benign_neighbor_support(
            row.get("same4_neighbor_label"),
            keywords=policy.benign_neighbor_keywords,
        ):
            return "retain", "Unresolved CPB-negative prefix retained because a same-4 neighbor supports a benign noncardiac family."
        return "exclude", "Unresolved CPB-negative prefix excluded because benign same-4 neighbor support was absent."
    return "exclude", "Residual manual-review case excluded from the final fully defined noncardiac cohort."


def annotate_operations_with_procedure_audit(
    operations_df: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_policy = _build_procedure_audit_policy(config)
    resolution_policy = _build_resolution_policy(config)
    audit_df = operations_df.copy()
    for column in ["subject_id", "department", "icd10_pcs", "cpbon_time", "cpboff_time"]:
        if column not in audit_df.columns:
            audit_df[column] = pd.NA

    audit_df["op_id_norm"] = normalize_op_ids(audit_df["op_id"])
    audit_df["subject_id_norm"] = normalize_op_ids(audit_df["subject_id"])
    audit_df["department"] = _normalize_string_series(audit_df["department"]).str.upper()
    audit_df["pcs_prefix"] = normalize_icd10_pcs(audit_df["icd10_pcs"])
    audit_df["pcs_len"] = audit_df["pcs_prefix"].astype("string").str.len().astype("Int64")
    audit_df["cpbon_time"] = _normalize_string_series(audit_df["cpbon_time"])
    audit_df["cpboff_time"] = _normalize_string_series(audit_df["cpboff_time"])
    audit_df["cpb_present"] = _nonempty_mask(audit_df["cpbon_time"]) | _nonempty_mask(audit_df["cpboff_time"])
    audit_df = audit_df.dropna(subset=["op_id_norm"]).drop_duplicates(subset=["op_id_norm"], keep="last")

    _validate_current_pcs_contract(audit_df)

    cms_reference = build_cms_prefix_reference(load_cms_order_entries(_procedure_audit_config(config)["cms_order_zip_path"]))
    audit_df = audit_df.merge(cms_reference, on="pcs_prefix", how="left")
    audit_df["op_id"] = audit_df["op_id_norm"]
    audit_df["subject_id"] = audit_df["subject_id_norm"]
    audit_df["department_display"] = audit_df["department"].map(_department_label)
    classified = audit_df.apply(lambda row: _classify_row(row, policy=audit_policy), axis=1)
    audit_df["audit_class"] = classified.map(lambda value: value.audit_class)
    audit_df["audit_reason_code"] = classified.map(lambda value: value.reason_code)
    audit_df["exclusion_reason"] = classified.map(lambda value: value.exclusion_reason)
    audit_df["audit_note"] = classified.map(lambda value: value.note)
    audit_df["clinician_review_bucket"] = classified.map(
        lambda value: value.clinician_review_bucket if value.clinician_review_bucket is not None else pd.NA
    )
    neighbor_lookup = _same4_neighbor_lookup(audit_df)
    audit_df["same4_neighbor_prefix"] = audit_df["pcs_prefix"].astype("string").map(
        lambda prefix: neighbor_lookup.get(str(prefix), ("", ""))[0] if pd.notna(prefix) else ""
    )
    audit_df["same4_neighbor_label"] = audit_df["pcs_prefix"].astype("string").map(
        lambda prefix: neighbor_lookup.get(str(prefix), ("", ""))[1] if pd.notna(prefix) else ""
    )
    resolved = audit_df.apply(lambda row: _final_noncardiac_resolution(row, policy=resolution_policy), axis=1)
    audit_df["final_noncardiac_action"] = resolved.map(lambda value: value[0])
    audit_df["final_noncardiac_note"] = resolved.map(lambda value: value[1])
    _validate_annotation_coverage(audit_df)
    return audit_df.reset_index(drop=True), cms_reference


def build_procedure_audit_frame(artifacts: ArtifactManager, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels_path = artifacts.paths.artifact_path("cohort", "labels.csv")
    if not labels_path.exists():
        raise FileNotFoundError(f"Required procedure-audit input was not found: {labels_path}")

    labels_df = pd.read_csv(labels_path, usecols=["op_id"])
    labels_df["op_id_norm"] = normalize_op_ids(labels_df["op_id"])
    labels_df = labels_df.dropna(subset=["op_id_norm"]).drop_duplicates(subset=["op_id_norm"], keep="last")[["op_id_norm"]]

    operations_df = read_csv_optimized(
        artifacts.paths.raw_inspire_dir / "operations.csv",
        config=config,
        usecols=["op_id", "subject_id", "department", "icd10_pcs", "cpbon_time", "cpboff_time"],
        large=True,
    )
    annotated_ops_df, cms_reference = annotate_operations_with_procedure_audit(operations_df, config)

    audit_df = labels_df.merge(annotated_ops_df, on="op_id_norm", how="left")
    if audit_df[["subject_id_norm", "department", "pcs_prefix"]].isna().all(axis=1).any():
        missing_count = int(audit_df[["subject_id_norm", "department", "pcs_prefix"]].isna().all(axis=1).sum())
        raise ValueError(f"Procedure audit could not join {missing_count} final-cohort operations back to operations.csv.")
    return audit_df.reset_index(drop=True), cms_reference


def _qc_sections(audit_df: pd.DataFrame, cms_reference: pd.DataFrame) -> list[TableSection]:
    total_ops = int(len(audit_df))
    nonnull_pcs = int(audit_df["pcs_prefix"].notna().sum())
    unique_prefixes = int(audit_df["pcs_prefix"].dropna().nunique())

    overview_rows = pd.DataFrame(
        [
            {"item": "Final analytic cohort operations", "n_ops": total_ops, "pct_of_final_cohort": _pct(total_ops, total_ops), "notes": "Operation-level op_id rows from cohort/labels.csv."},
            {"item": "Non-null icd10_pcs", "n_ops": nonnull_pcs, "pct_of_final_cohort": _pct(nonnull_pcs, total_ops), "notes": "Observed on the joined final cohort operations."},
            {"item": "Unique observed 5-character prefixes", "n_ops": unique_prefixes, "pct_of_final_cohort": "", "notes": "Unique icd10_pcs prefixes in the final analytic cohort."},
        ]
    )

    length_rows = pd.DataFrame(
        [
            {
                "item": f"Length {int(length)}",
                "n_ops": int(count),
                "pct_of_final_cohort": _pct(int(count), total_ops),
                "notes": "Observed icd10_pcs string length in the final analytic cohort.",
            }
            for length, count in _pcs_length_counts(audit_df).items()
        ]
    )

    top_prefix_counts = (
        audit_df["pcs_prefix"]
        .dropna()
        .value_counts()
        .rename_axis("pcs_prefix")
        .reset_index(name="n_ops")
        .head(25)
        .merge(cms_reference[["pcs_prefix", "canonical_prefix_label"]], left_on="pcs_prefix", right_on="pcs_prefix", how="left")
    )
    top_prefix_rows = top_prefix_counts.rename(columns={"pcs_prefix": "item", "canonical_prefix_label": "notes"}).copy()
    top_prefix_rows["pct_of_final_cohort"] = top_prefix_rows["n_ops"].map(lambda count: _pct(int(count), total_ops))
    top_prefix_rows["notes"] = top_prefix_rows["notes"].fillna("No collapsed CMS title available.")

    return [
        TableSection(title="Overview", display_df=overview_rows, csv_df=overview_rows),
        TableSection(title="Observed ICD-10-PCS Lengths", display_df=length_rows, csv_df=length_rows),
        TableSection(title="Top Observed Prefixes", display_df=top_prefix_rows[["item", "n_ops", "pct_of_final_cohort", "notes"]], csv_df=top_prefix_rows[["item", "n_ops", "pct_of_final_cohort", "notes"]]),
    ]


def _global_summary_frame(audit_df: pd.DataFrame) -> pd.DataFrame:
    final_total = int(len(audit_df))
    rows: list[dict[str, Any]] = []
    for audit_class in _SUMMARY_CLASS_ORDER:
        class_df = audit_df.loc[audit_df["audit_class"] == audit_class].copy()
        n_ops = int(len(class_df))
        rows.append(
            {
                "audit_class": audit_class,
                "n_ops": n_ops,
                "pct_of_final_cohort": _pct(n_ops, final_total),
                "pct_ct_department": _pct(int(class_df["department"].eq("CTS").sum()), n_ops),
                "pct_with_cpb_flag": _pct(int(class_df["cpb_present"].astype(bool).sum()), n_ops),
                "notes": _SUMMARY_CLASS_NOTES[audit_class],
            }
        )
    return pd.DataFrame(rows)


def _ct_summary_frame(audit_df: pd.DataFrame, *, ct_department_code: str) -> pd.DataFrame:
    ct_df = audit_df.loc[audit_df["department"].astype("string") == ct_department_code].copy()
    ct_total = int(len(ct_df))
    final_total = int(len(audit_df))
    rows: list[dict[str, Any]] = []
    for audit_class in _SUMMARY_CLASS_ORDER:
        if audit_class == "other_operation":
            continue
        class_df = ct_df.loc[ct_df["audit_class"] == audit_class].copy()
        n_ops = int(len(class_df))
        rows.append(
            {
                "audit_class": audit_class,
                "n_ops": n_ops,
                "pct_of_ct_department_ops": _pct(n_ops, ct_total),
                "pct_of_final_cohort": _pct(n_ops, final_total),
                "pct_with_cpb_flag": _pct(int(class_df["cpb_present"].astype(bool).sum()), n_ops),
                "notes": _SUMMARY_CLASS_NOTES[audit_class],
            }
        )
    return pd.DataFrame(rows)


def _ct_manuscript_summary_frame(audit_df: pd.DataFrame, *, ct_department_code: str) -> pd.DataFrame:
    ct_df = audit_df.loc[audit_df["department"].astype("string") == ct_department_code].copy()
    ct_total = int(len(ct_df))
    thoracic_retained_mask = ct_df["audit_class"].isin(
        ["thoracic_keep", "thoracic_or_chest_related_noncardiac"]
    ) & ct_df["final_noncardiac_action"].eq("retain")
    residual_excluded_mask = ct_df["final_noncardiac_action"].eq("exclude") & ct_df["audit_class"].ne("cardiac_exclude")

    rows = [
        {
            "item": f"{_department_label(ct_department_code)}-labeled operations reviewed",
            "n_ops": ct_total,
            "pct_of_ct_department_ops": _pct(ct_total, ct_total),
            "notes": "Administrative CTS subset audited at the operation level; department was not used alone to define operative phenotype.",
        },
        {
            "item": "Definite cardiac procedures excluded",
            "n_ops": int(ct_df["audit_class"].eq("cardiac_exclude").sum()),
            "pct_of_ct_department_ops": _pct(int(ct_df["audit_class"].eq("cardiac_exclude").sum()), ct_total),
            "notes": "Official Heart and Great Vessels families or explicit cardiac title families excluded from the final noncardiac cohort.",
        },
        {
            "item": "Clearly noncardiac thoracic or chest procedures retained",
            "n_ops": int(thoracic_retained_mask.sum()),
            "pct_of_ct_department_ops": _pct(int(thoracic_retained_mask.sum()), ct_total),
            "notes": "Respiratory plus thoracic, mediastinal, foregut, and chest-related noncardiac families retained after operation-level adjudication.",
        },
        {
            "item": "Ambiguous CPB-supported or unresolved procedures excluded",
            "n_ops": int(residual_excluded_mask.sum()),
            "pct_of_ct_department_ops": _pct(int(residual_excluded_mask.sum()), ct_total),
            "notes": "Residual CPB-supported discordant or unresolved CTS families excluded conservatively under the default strict noncardiac rule.",
        },
        {
            "item": "Final retained noncardiac CTS-labeled operations",
            "n_ops": int(ct_df["final_noncardiac_action"].eq("retain").sum()),
            "pct_of_ct_department_ops": _pct(int(ct_df["final_noncardiac_action"].eq("retain").sum()), ct_total),
            "notes": "CTS-labeled operations retained in the final operation-level noncardiac cohort after adjudication.",
        },
    ]
    return pd.DataFrame(rows)


def _prefix_summary_audit_class(prefix_df: pd.DataFrame) -> str:
    classes = sorted(set(prefix_df["audit_class"].dropna().astype(str).tolist()))
    if not classes:
        return "other_operation"
    if len(classes) == 1:
        return classes[0]
    if "cardiac_exclude" in classes:
        return "cardiac_exclude"
    if "manual_review" in classes:
        return "manual_review"
    return classes[0]


def _cpb_duration_minutes(frame: pd.DataFrame) -> pd.Series:
    return pd.to_numeric(frame["cpboff_time"], errors="coerce") - pd.to_numeric(frame["cpbon_time"], errors="coerce")


def _clinician_review_priority(bucket: Any) -> str:
    bucket_name = str(bucket or "")
    if bucket_name == "cpb_positive_aortic_or_vascular":
        return "high"
    if bucket_name in {"respiratory_plus_cpb", "unresolved_prefix_or_title", "other_cpb_discordant_nonvascular_nonrespiratory"}:
        return "medium"
    if bucket_name == "other_prefix_level_review":
        return "low"
    return ""


def _same4_neighbor_lookup(audit_df: pd.DataFrame) -> dict[str, tuple[str, str]]:
    all_prefixes = (
        audit_df["pcs_prefix"]
        .dropna()
        .astype("string")
        .drop_duplicates()
        .sort_values(kind="stable")
        .tolist()
    )
    candidates = (
        audit_df.loc[audit_df["canonical_prefix_label"].notna(), ["pcs_prefix", "canonical_prefix_label"]]
        .dropna(subset=["pcs_prefix"])
        .astype({"pcs_prefix": "string", "canonical_prefix_label": "string"})
    )
    if candidates.empty:
        return {}
    counts = candidates["pcs_prefix"].value_counts().rename_axis("pcs_prefix").reset_index(name="n_ops")
    counts = counts.merge(
        candidates.drop_duplicates(subset=["pcs_prefix"], keep="first"),
        on="pcs_prefix",
        how="left",
    )
    lookup: dict[str, tuple[str, str]] = {}
    for prefix in [str(value) for value in all_prefixes]:
        same4 = counts.loc[
            counts["pcs_prefix"].astype(str).str.startswith(prefix[:4]) & counts["pcs_prefix"].astype(str).ne(prefix)
        ].copy()
        if same4.empty:
            continue
        same4 = same4.sort_values(["n_ops", "pcs_prefix"], ascending=[False, True], kind="stable")
        best = same4.iloc[0]
        lookup[prefix] = (str(best["pcs_prefix"]), str(best["canonical_prefix_label"]))
    return lookup


def _validate_annotation_coverage(audit_df: pd.DataFrame) -> None:
    if audit_df["audit_class"].isna().any():
        raise ValueError("Procedure audit left some operations without an audit_class.")
    if audit_df["final_noncardiac_action"].isna().any():
        raise ValueError("Procedure audit left some operations without a final_noncardiac_action.")
    observed_actions = set(audit_df["final_noncardiac_action"].dropna().astype(str).tolist())
    unknown_actions = sorted(observed_actions - _FINAL_NONCARDIAC_ACTIONS)
    if unknown_actions:
        raise ValueError(f"Procedure audit emitted unknown final_noncardiac_action values: {unknown_actions}")


def _clinician_review_summary_frame(audit_df: pd.DataFrame) -> pd.DataFrame:
    manual_df = audit_df.loc[audit_df["audit_class"] == "manual_review"].copy()
    if manual_df.empty:
        return pd.DataFrame(
            columns=[
                "clinician_review_bucket",
                "n_ops",
                "pct_of_manual_review",
                "pct_with_cpb_flag",
                "median_cpb_duration_min",
                "n_retained_in_final_cohort",
                "n_excluded_from_final_cohort",
                "recommended_final_action",
                "top_prefixes",
                "notes",
            ]
        )
    manual_total = int(len(manual_df))
    manual_df["cpb_duration_min"] = _cpb_duration_minutes(manual_df)
    rows: list[dict[str, Any]] = []
    for bucket in _CLINICIAN_REVIEW_BUCKET_ORDER:
        bucket_df = manual_df.loc[manual_df["clinician_review_bucket"] == bucket].copy()
        n_ops = int(len(bucket_df))
        top_prefixes = ", ".join(
            f"{prefix} ({count})"
            for prefix, count in bucket_df["pcs_prefix"].fillna("").value_counts().head(5).items()
            if str(prefix).strip()
        )
        median_duration = bucket_df.loc[bucket_df["cpb_present"].astype(bool), "cpb_duration_min"].dropna()
        rows.append(
            {
                "clinician_review_bucket": bucket,
                "n_ops": n_ops,
                "pct_of_manual_review": _pct(n_ops, manual_total),
                "pct_with_cpb_flag": _pct(int(bucket_df["cpb_present"].astype(bool).sum()), n_ops),
                "median_cpb_duration_min": round(float(median_duration.median()), 1) if not median_duration.empty else "",
                "n_retained_in_final_cohort": int(bucket_df["final_noncardiac_action"].eq("retain").sum()),
                "n_excluded_from_final_cohort": int(bucket_df["final_noncardiac_action"].eq("exclude").sum()),
                "recommended_final_action": (
                    bucket_df["final_noncardiac_action"].dropna().astype(str).mode().iat[0]
                    if not bucket_df.empty and bucket_df["final_noncardiac_action"].dropna().any()
                    else ""
                ),
                "top_prefixes": top_prefixes,
                "notes": _CLINICIAN_REVIEW_BUCKET_NOTES[bucket],
            }
        )
    return pd.DataFrame(rows)


def _ct_top_prefixes_frame(audit_df: pd.DataFrame, *, ct_department_code: str) -> pd.DataFrame:
    ct_df = audit_df.loc[audit_df["department"].astype("string") == ct_department_code].copy()
    if ct_df.empty:
        return pd.DataFrame(
            columns=[
                "pcs_prefix",
                "n_ops",
                "pct_of_ct_department_ops",
                "canonical_prefix_label",
                "body_system_desc",
                "root_op_desc",
                "approach_desc",
                "audit_class",
            ]
        )

    rows: list[dict[str, Any]] = []
    ct_total = int(len(ct_df))
    for pcs_prefix, prefix_df in ct_df.groupby("pcs_prefix", dropna=False, sort=False):
        prefix_df = prefix_df.copy()
        rows.append(
            {
                "pcs_prefix": pcs_prefix if pd.notna(pcs_prefix) else "",
                "n_ops": int(len(prefix_df)),
                "pct_of_ct_department_ops": _pct(int(len(prefix_df)), ct_total),
                "canonical_prefix_label": prefix_df["canonical_prefix_label"].dropna().astype(str).iloc[0]
                if prefix_df["canonical_prefix_label"].notna().any()
                else "",
                "body_system_desc": prefix_df["body_system_desc"].dropna().astype(str).iloc[0]
                if prefix_df["body_system_desc"].notna().any()
                else "",
                "root_op_desc": prefix_df["root_op_desc"].dropna().astype(str).iloc[0]
                if prefix_df["root_op_desc"].notna().any()
                else "",
                "approach_desc": prefix_df["approach_desc"].dropna().astype(str).iloc[0]
                if prefix_df["approach_desc"].notna().any()
                else "",
                "audit_class": _prefix_summary_audit_class(prefix_df),
            }
        )
    prefix_df = pd.DataFrame(rows)
    return prefix_df.sort_values(["n_ops", "pcs_prefix"], ascending=[False, True], kind="stable").reset_index(drop=True)


def _flagged_cardiac_frame(audit_df: pd.DataFrame) -> pd.DataFrame:
    flagged_df = audit_df.loc[audit_df["audit_class"] == "cardiac_exclude"].copy()
    columns = [
        "op_id",
        "subject_id",
        "department_display",
        "pcs_prefix",
        "canonical_prefix_label",
        "body_system_desc",
        "root_op_desc",
        "approach_desc",
        "cpb_present",
        "cpbon_time",
        "cpboff_time",
        "exclusion_reason",
        "audit_note",
    ]
    available_columns = [column for column in columns if column in flagged_df.columns]
    return flagged_df[available_columns].rename(columns={"department_display": "department", "audit_note": "notes"}).reset_index(drop=True)


def _manual_review_frame(audit_df: pd.DataFrame, *, ct_department_code: str) -> pd.DataFrame:
    manual_df = audit_df.loc[
        (audit_df["department"].astype("string") == ct_department_code) & (audit_df["audit_class"] == "manual_review")
    ].copy()
    manual_df["cpb_duration_min"] = _cpb_duration_minutes(manual_df)
    manual_df["clinician_review_priority"] = manual_df["clinician_review_bucket"].map(_clinician_review_priority)
    manual_df["review_disposition"] = ""
    manual_df["review_comment"] = ""
    columns = [
        "op_id",
        "subject_id",
        "department_display",
        "pcs_prefix",
        "canonical_prefix_label",
        "body_system_desc",
        "root_op_desc",
        "approach_desc",
        "cpb_present",
        "cpbon_time",
        "cpboff_time",
        "cpb_duration_min",
        "audit_reason_code",
        "audit_note",
        "clinician_review_bucket",
        "clinician_review_priority",
        "same4_neighbor_prefix",
        "same4_neighbor_label",
        "final_noncardiac_action",
        "final_noncardiac_note",
        "review_disposition",
        "review_comment",
    ]
    available_columns = [column for column in columns if column in manual_df.columns]
    return manual_df[available_columns].rename(columns={"department_display": "department", "audit_note": "notes"}).reset_index(drop=True)


def _simple_table_spec(
    *,
    file_stem: str,
    title: str,
    caption: str,
    columns: list[ColumnSpec],
    frame: pd.DataFrame,
    empty_message: str,
) -> TableSpec:
    display_columns = [column.key for column in columns if column.key in frame.columns]
    return TableSpec(
        file_stem=file_stem,
        title=title,
        caption=caption,
        columns=columns,
        sections=[TableSection(title=None, display_df=frame[display_columns].copy(), csv_df=frame.copy())],
        include_section_column_in_csv=False,
        empty_message=empty_message,
    )


def generate_procedure_audit_outputs(artifacts: ArtifactManager, config: dict | None = None) -> list[Path]:
    resolved_config = artifacts.config if config is None else config
    audit_df, cms_reference = build_procedure_audit_frame(artifacts, resolved_config)
    ct_department_code = str(_procedure_audit_config(resolved_config).get("ct_department_code", "CTS")).upper()

    outputs: list[Path] = []
    qc_spec = TableSpec(
        file_stem="procedure_audit_qc_summary",
        title="Procedure Audit QC Summary",
        caption="Observed ICD-10-PCS contract and top 5-character prefixes in the final analytic cohort.",
        columns=[
            ColumnSpec("item", "Item", align="left"),
            ColumnSpec("n_ops", "Operations"),
            ColumnSpec("pct_of_final_cohort", "% of Final Cohort"),
            ColumnSpec("notes", "Notes", align="left"),
        ],
        sections=_qc_sections(audit_df, cms_reference),
        include_section_column_in_csv=False,
    )
    outputs.extend(write_table_outputs(artifacts, qc_spec, resolved_config))

    global_summary = _global_summary_frame(audit_df)
    outputs.extend(
        write_table_outputs(
            artifacts,
            _simple_table_spec(
                file_stem="procedure_audit_global_summary",
                title="Global Cardiac Leak Check",
                caption="Operation-level audit summary across the final analytic cohort.",
                columns=[
                    ColumnSpec("audit_class", "Audit Class", align="left"),
                    ColumnSpec("n_ops", "Operations"),
                    ColumnSpec("pct_of_final_cohort", "% of Final Cohort"),
                    ColumnSpec("pct_ct_department", "% in CTS"),
                    ColumnSpec("pct_with_cpb_flag", "% with CPB Flag"),
                    ColumnSpec("notes", "Notes", align="left"),
                ],
                frame=global_summary,
                empty_message="No global procedure-audit rows were available.",
            ),
            resolved_config,
        )
    )

    ct_summary = _ct_summary_frame(audit_df, ct_department_code=ct_department_code)
    outputs.extend(
        write_table_outputs(
            artifacts,
            _simple_table_spec(
                file_stem="procedure_audit_ct_department_summary",
                title=f"{_department_label(ct_department_code)} Procedure Audit Summary",
                caption="Audit classes within the reviewer-facing cardiothoracic department subset.",
                columns=[
                    ColumnSpec("audit_class", "Audit Class", align="left"),
                    ColumnSpec("n_ops", "Operations"),
                    ColumnSpec("pct_of_ct_department_ops", "% of CTS"),
                    ColumnSpec("pct_of_final_cohort", "% of Final Cohort"),
                    ColumnSpec("pct_with_cpb_flag", "% with CPB Flag"),
                    ColumnSpec("notes", "Notes", align="left"),
                ],
                frame=ct_summary,
                empty_message="No CTS procedure-audit rows were available.",
            ),
            resolved_config,
        )
    )

    ct_top_prefixes = _ct_top_prefixes_frame(audit_df, ct_department_code=ct_department_code)
    outputs.extend(
        write_table_outputs(
            artifacts,
            _simple_table_spec(
                file_stem="procedure_audit_ct_top_prefixes",
                title=f"{_department_label(ct_department_code)} Top ICD-10-PCS Prefixes",
                caption="Top 5-character ICD-10-PCS prefixes within the reviewer-facing cardiothoracic subset.",
                columns=[
                    ColumnSpec("pcs_prefix", "PCS Prefix", align="left"),
                    ColumnSpec("n_ops", "Operations"),
                    ColumnSpec("pct_of_ct_department_ops", "% of CTS"),
                    ColumnSpec("canonical_prefix_label", "Canonical Prefix Label", align="left"),
                    ColumnSpec("body_system_desc", "Body System", align="left"),
                    ColumnSpec("root_op_desc", "Root Operation", align="left"),
                    ColumnSpec("approach_desc", "Approach", align="left"),
                    ColumnSpec("audit_class", "Audit Class", align="left"),
                ],
                frame=ct_top_prefixes,
                empty_message="No CTS ICD-10-PCS prefixes were available.",
            ),
            resolved_config,
        )
    )

    ct_manuscript_summary = _ct_manuscript_summary_frame(audit_df, ct_department_code=ct_department_code)
    outputs.extend(
        write_table_outputs(
            artifacts,
            _simple_table_spec(
                file_stem="procedure_audit_ct_manuscript_summary",
                title=f"{_department_label(ct_department_code)} Audit Manuscript Summary",
                caption="Compact cardiothoracic count summary for manuscript and reviewer-response use under the canonical strict noncardiac cohort definition.",
                columns=[
                    ColumnSpec("item", "Item", align="left"),
                    ColumnSpec("n_ops", "Operations"),
                    ColumnSpec("pct_of_ct_department_ops", "% of CTS"),
                    ColumnSpec("notes", "Notes", align="left"),
                ],
                frame=ct_manuscript_summary,
                empty_message="No CTS manuscript-summary rows were available.",
            ),
            resolved_config,
        )
    )

    clinician_review_summary = _clinician_review_summary_frame(audit_df)
    outputs.extend(
        write_table_outputs(
            artifacts,
            _simple_table_spec(
                file_stem="procedure_audit_clinician_review_summary",
                title="Clinician Review Buckets",
                caption="Direct-review summary for the residual clinician-review buckets, including the final keep versus exclude recommendation for a fully defined noncardiac cohort.",
                columns=[
                    ColumnSpec("clinician_review_bucket", "Clinician Review Bucket", align="left"),
                    ColumnSpec("n_ops", "Operations"),
                    ColumnSpec("pct_of_manual_review", "% of Manual Review"),
                    ColumnSpec("pct_with_cpb_flag", "% with CPB Flag"),
                    ColumnSpec("median_cpb_duration_min", "Median CPB Duration (min)"),
                    ColumnSpec("n_retained_in_final_cohort", "Retained in Final Cohort"),
                    ColumnSpec("n_excluded_from_final_cohort", "Excluded from Final Cohort"),
                    ColumnSpec("recommended_final_action", "Recommended Final Action", align="left"),
                    ColumnSpec("top_prefixes", "Top Prefixes", align="left"),
                    ColumnSpec("notes", "Notes", align="left"),
                ],
                frame=clinician_review_summary,
                empty_message="No clinician-review buckets were available.",
            ),
            resolved_config,
        )
    )

    flagged_cardiac = _flagged_cardiac_frame(audit_df)
    outputs.extend(
        write_table_outputs(
            artifacts,
            _simple_table_spec(
                file_stem="procedure_audit_flagged_cardiac_cases",
                title="Flagged Definite Cardiac Cases",
                caption="Machine-readable exclusion manifest for definite cardiac procedures that remain in the final analytic cohort.",
                columns=[
                    ColumnSpec("op_id", "Op ID", align="left"),
                    ColumnSpec("subject_id", "Subject ID", align="left"),
                    ColumnSpec("department", "Department", align="left"),
                    ColumnSpec("pcs_prefix", "PCS Prefix", align="left"),
                    ColumnSpec("canonical_prefix_label", "Canonical Prefix Label", align="left"),
                    ColumnSpec("body_system_desc", "Body System", align="left"),
                    ColumnSpec("root_op_desc", "Root Operation", align="left"),
                    ColumnSpec("approach_desc", "Approach", align="left"),
                    ColumnSpec("cpb_present", "CPB Flag", align="left"),
                    ColumnSpec("cpbon_time", "CPB On", align="left"),
                    ColumnSpec("cpboff_time", "CPB Off", align="left"),
                    ColumnSpec("exclusion_reason", "Exclusion Reason", align="left"),
                    ColumnSpec("notes", "Notes", align="left"),
                ],
                frame=flagged_cardiac,
                empty_message="No definite cardiac procedures were flagged in the final analytic cohort.",
            ),
            resolved_config,
        )
    )

    manual_review = _manual_review_frame(audit_df, ct_department_code=ct_department_code)
    outputs.extend(
        write_table_outputs(
            artifacts,
            _simple_table_spec(
                file_stem="procedure_audit_manual_review",
                title="Cardiothoracic Procedure Gray-Zone Ledger",
                caption="Row-level ledger for CTS cases that remain prefix-level gray-zone after conservative ICD-10-PCS and CPB screening, with the final noncardiac-cohort recommendation recorded explicitly.",
                columns=[
                    ColumnSpec("op_id", "Op ID", align="left"),
                    ColumnSpec("subject_id", "Subject ID", align="left"),
                    ColumnSpec("department", "Department", align="left"),
                    ColumnSpec("pcs_prefix", "PCS Prefix", align="left"),
                    ColumnSpec("canonical_prefix_label", "Canonical Prefix Label", align="left"),
                    ColumnSpec("body_system_desc", "Body System", align="left"),
                    ColumnSpec("root_op_desc", "Root Operation", align="left"),
                    ColumnSpec("approach_desc", "Approach", align="left"),
                    ColumnSpec("cpb_present", "CPB Flag", align="left"),
                    ColumnSpec("cpbon_time", "CPB On", align="left"),
                    ColumnSpec("cpboff_time", "CPB Off", align="left"),
                    ColumnSpec("cpb_duration_min", "CPB Duration (min)", align="left"),
                    ColumnSpec("audit_reason_code", "Audit Reason Code", align="left"),
                    ColumnSpec("clinician_review_bucket", "Clinician Review Bucket", align="left"),
                    ColumnSpec("clinician_review_priority", "Priority", align="left"),
                    ColumnSpec("same4_neighbor_prefix", "Same-4 Neighbor", align="left"),
                    ColumnSpec("same4_neighbor_label", "Neighbor Label", align="left"),
                    ColumnSpec("notes", "Audit Note", align="left"),
                    ColumnSpec("final_noncardiac_action", "Final Cohort Action", align="left"),
                    ColumnSpec("final_noncardiac_note", "Final Cohort Note", align="left"),
                    ColumnSpec("review_disposition", "Review Disposition", align="left"),
                    ColumnSpec("review_comment", "Review Comment", align="left"),
                ],
                frame=manual_review,
                empty_message="No CTS cases were routed to the residual gray-zone ledger.",
            ),
            resolved_config,
        )
    )

    return outputs
