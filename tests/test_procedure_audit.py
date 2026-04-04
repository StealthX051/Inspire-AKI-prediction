from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
import pytest

from inspire_aki.config import load_config
from inspire_aki.cohort.preop import build_preop_features
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.pipelines.report import run_procedure_audit
from inspire_aki.reporting.procedure_audit import (
    annotate_operations_with_procedure_audit,
    build_cms_prefix_reference,
    build_procedure_audit_frame,
    load_cms_order_entries,
)


def _cms_line(seq: int, code: str, valid: int, title: str, long_title: str | None = None) -> str:
    return f"{seq:05d} {code:<7} {valid} {title}  {long_title or title}"


def _write_synthetic_cms_order_zip(base_dir: Path) -> Path:
    zip_path = base_dir / "synthetic_icd10pcs_order.zip"
    lines = [
        _cms_line(3200, "021", 0, "Heart and Great Vessels, Bypass"),
        _cms_line(
            3275,
            "0211083",
            1,
            "Bypass Coronary Artery, Two Arteries from Aorta with Autologous Venous Tissue, Open Approach",
        ),
        _cms_line(3700, "02R", 0, "Heart and Great Vessels, Replacement"),
        _cms_line(
            3750,
            "02RF07Z",
            1,
            "Replacement of Aortic Valve with Synthetic Substitute, Open Approach",
        ),
        _cms_line(27100, "0BT", 0, "Respiratory System, Resection"),
        _cms_line(
            27125,
            "0BTC4ZZ",
            1,
            "Resection of Right Upper Lung Lobe, Percutaneous Endoscopic Approach",
        ),
        _cms_line(3900, "04R", 0, "Lower Arteries, Replacement"),
        _cms_line(
            3925,
            "04R00JZ",
            1,
            "Replacement of Abdominal Aorta with Synthetic Substitute, Open Approach",
        ),
        _cms_line(45100, "0WJ", 0, "General Anatomical Regions, Inspection"),
        _cms_line(
            45125,
            "0WJ84ZZ",
            1,
            "Inspection of Chest Wall, Percutaneous Endoscopic Approach",
        ),
        _cms_line(4700, "07B", 0, "Lymphatic and Hemic Systems, Excision"),
        _cms_line(
            4725,
            "07BM0ZZ",
            1,
            "Excision of Thymus, Open Approach",
        ),
        _cms_line(4800, "0HD", 0, "Skin and Breast, Extraction"),
        _cms_line(
            4825,
            "0HD0XZZ",
            1,
            "Extraction of Scalp Skin, External Approach",
        ),
        _cms_line(4850, "0HB", 0, "Skin and Breast, Excision"),
        _cms_line(
            4875,
            "0HB0XZZ",
            1,
            "Excision of Scalp Skin, External Approach",
        ),
    ]
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("icd10pcs_order_2026.txt", "\n".join(lines) + "\n")
    return zip_path


def _write_procedure_audit_inputs(config_path: Path, *, with_mixed_lengths: bool = False) -> dict:
    config = load_config(config_path)
    cms_zip_path = _write_synthetic_cms_order_zip(config_path.parent)
    config["reports"]["procedure_audit"]["cms_order_zip_path"] = str(cms_zip_path)
    artifacts = ArtifactManager(config)

    operations = pd.DataFrame(
        [
            {"op_id": 1, "subject_id": 1001, "department": "CTS", "icd10_pcs": "02110", "cpbon_time": None, "cpboff_time": None},
            {"op_id": 2, "subject_id": 1002, "department": "CTS", "icd10_pcs": "0BTC4", "cpbon_time": None, "cpboff_time": None},
            {"op_id": 3, "subject_id": 1003, "department": "CTS", "icd10_pcs": "0BTC4", "cpbon_time": "111", "cpboff_time": None},
            {"op_id": 4, "subject_id": 1004, "department": "CTS", "icd10_pcs": "0WJ84", "cpbon_time": None, "cpboff_time": None},
            {"op_id": 5, "subject_id": 1005, "department": "CTS", "icd10_pcs": "0HB0X", "cpbon_time": None, "cpboff_time": None},
            {"op_id": 6, "subject_id": 1006, "department": "CTS", "icd10_pcs": "04R00", "cpbon_time": None, "cpboff_time": None},
            {"op_id": 7, "subject_id": 1007, "department": "CTS", "icd10_pcs": "04R00", "cpbon_time": "222", "cpboff_time": "333"},
            {"op_id": 8, "subject_id": 1008, "department": "GS", "icd10_pcs": "02RF0", "cpbon_time": None, "cpboff_time": None},
            {"op_id": 9, "subject_id": 1009, "department": "GS", "icd10_pcs": "0DTP0", "cpbon_time": None, "cpboff_time": None},
            {"op_id": 10, "subject_id": 1010, "department": "CTS", "icd10_pcs": "0HB00", "cpbon_time": None, "cpboff_time": None},
            {"op_id": 11, "subject_id": 1011, "department": "CTS", "icd10_pcs": "07BM0", "cpbon_time": None, "cpboff_time": None},
            {"op_id": 12, "subject_id": 1012, "department": "CTS", "icd10_pcs": "07BM0", "cpbon_time": "444", "cpboff_time": "555"},
            {"op_id": 13, "subject_id": 1013, "department": "CTS", "icd10_pcs": "0A152", "cpbon_time": None, "cpboff_time": None},
        ]
    )
    if with_mixed_lengths:
        operations.loc[operations["op_id"] == 10, "icd10_pcs"] = "0HB0"

    operations.to_csv(config_path.parent / "raw" / "operations.csv", index=False)
    artifacts.write_dataframe(
        pd.DataFrame({"op_id": operations["op_id"], "aki_boolean": [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]}),
        "cohort",
        "labels.csv",
    )
    return config


def test_procedure_audit_reference_parser_builds_expected_prefix_metadata(tmp_path: Path) -> None:
    cms_zip_path = _write_synthetic_cms_order_zip(tmp_path)

    entries = load_cms_order_entries(cms_zip_path)
    reference = build_cms_prefix_reference(entries).set_index("pcs_prefix")

    assert reference.loc["02110", "body_system_desc"] == "Heart and Great Vessels"
    assert reference.loc["02110", "root_op_desc"] == "Bypass"
    assert "Bypass Coronary Artery" in reference.loc["02110", "canonical_prefix_label"]

    assert reference.loc["02RF0", "body_system_desc"] == "Heart and Great Vessels"
    assert reference.loc["02RF0", "root_op_desc"] == "Replacement"
    assert reference.loc["02RF0", "canonical_prefix_label"] == "Replacement of Aortic Valve with Synthetic Substitute"

    assert reference.loc["0BTC4", "body_system_desc"] == "Respiratory System"
    assert reference.loc["0BTC4", "root_op_desc"] == "Resection"
    assert reference.loc["0BTC4", "canonical_prefix_label"] == "Resection of Right Upper Lung Lobe"
    assert reference.loc["0BTC4", "approach_desc"] == "Percutaneous Endoscopic Approach"

    assert reference.loc["0HB0X", "body_system_desc"] == "Skin and Breast"
    assert reference.loc["0HB0X", "root_op_desc"] == "Excision"
    assert reference.loc["0HB0X", "canonical_prefix_label"] == "Excision of Scalp Skin"


def test_build_procedure_audit_frame_classifies_rows(synthetic_config: Path) -> None:
    config = _write_procedure_audit_inputs(synthetic_config)
    artifacts = ArtifactManager(config)

    audit_df, _ = build_procedure_audit_frame(artifacts, config)
    classes = dict(zip(audit_df["op_id"].astype(str), audit_df["audit_class"], strict=True))
    clinician_buckets = dict(zip(audit_df["op_id"].astype(str), audit_df["clinician_review_bucket"].astype("string"), strict=True))

    assert classes == {
        "1": "cardiac_exclude",
        "2": "thoracic_keep",
        "3": "manual_review",
        "4": "thoracic_or_chest_related_noncardiac",
        "5": "ct_service_noncardiac_keep",
        "6": "vascular_noncardiac_describe",
        "7": "manual_review",
        "8": "cardiac_exclude",
        "9": "other_operation",
        "10": "manual_review",
        "11": "thoracic_or_chest_related_noncardiac",
        "12": "manual_review",
        "13": "manual_review",
    }
    assert clinician_buckets["3"] == "respiratory_plus_cpb"
    assert clinician_buckets["7"] == "cpb_positive_aortic_or_vascular"
    assert clinician_buckets["10"] == "unresolved_prefix_or_title"
    assert clinician_buckets["12"] == "other_cpb_discordant_nonvascular_nonrespiratory"
    assert clinician_buckets["13"] == "unresolved_prefix_or_title"


def test_procedure_audit_annotations_compute_final_noncardiac_actions(synthetic_config: Path) -> None:
    config = _write_procedure_audit_inputs(synthetic_config)
    artifacts = ArtifactManager(config)
    operations_df = pd.read_csv(artifacts.paths.raw_inspire_dir / "operations.csv")

    audit_df, _ = annotate_operations_with_procedure_audit(operations_df, config)
    actions = dict(zip(audit_df["op_id"].astype(str), audit_df["final_noncardiac_action"], strict=True))

    assert audit_df["audit_class"].notna().all()
    assert audit_df["final_noncardiac_action"].notna().all()
    assert set(audit_df["final_noncardiac_action"]) == {"retain", "exclude"}
    assert actions == {
        "1": "exclude",
        "2": "retain",
        "3": "exclude",
        "4": "retain",
        "5": "retain",
        "6": "retain",
        "7": "exclude",
        "8": "exclude",
        "9": "retain",
        "10": "retain",
        "11": "retain",
        "12": "exclude",
        "13": "exclude",
    }

    op10 = audit_df.loc[audit_df["op_id"].astype(str) == "10"].iloc[0]
    op13 = audit_df.loc[audit_df["op_id"].astype(str) == "13"].iloc[0]
    assert op10["same4_neighbor_prefix"] == "0HB0X"
    assert "benign noncardiac family" in op10["final_noncardiac_note"]
    assert op13["same4_neighbor_prefix"] == ""
    assert "benign same-4 neighbor support was absent" in op13["final_noncardiac_note"]


def test_build_procedure_audit_frame_rejects_mixed_length_icd10_pcs(synthetic_config: Path) -> None:
    config = _write_procedure_audit_inputs(synthetic_config, with_mixed_lengths=True)
    artifacts = ArtifactManager(config)

    with pytest.raises(ValueError, match="5-character icd10_pcs contract"):
        build_procedure_audit_frame(artifacts, config)


def test_run_procedure_audit_writes_outputs_and_manifest(synthetic_config: Path) -> None:
    config = _write_procedure_audit_inputs(synthetic_config)
    artifacts = ArtifactManager(config)

    payload = run_procedure_audit(config)

    expected_stems = [
        "procedure_audit_qc_summary",
        "procedure_audit_global_summary",
        "procedure_audit_ct_department_summary",
        "procedure_audit_ct_top_prefixes",
        "procedure_audit_ct_manuscript_summary",
        "procedure_audit_clinician_review_summary",
        "procedure_audit_flagged_cardiac_cases",
        "procedure_audit_manual_review",
    ]
    assert len(payload["outputs"]) == len(expected_stems) * 3
    for stem in expected_stems:
        for suffix in ("csv", "html", "md"):
            assert artifacts.paths.artifact_path("reports", "tables", f"{stem}.{suffix}").exists()

    flagged_df = pd.read_csv(artifacts.paths.artifact_path("reports", "tables", "procedure_audit_flagged_cardiac_cases.csv"))
    manual_df = pd.read_csv(artifacts.paths.artifact_path("reports", "tables", "procedure_audit_manual_review.csv"))
    clinician_df = pd.read_csv(artifacts.paths.artifact_path("reports", "tables", "procedure_audit_clinician_review_summary.csv"))
    manuscript_df = pd.read_csv(
        artifacts.paths.artifact_path("reports", "tables", "procedure_audit_ct_manuscript_summary.csv")
    )
    manifest = artifacts.read_json("manifests", "report_procedure_audit.json")

    assert set(flagged_df["op_id"].astype(str)) == {"1", "8"}
    assert set(manual_df["op_id"].astype(str)) == {"3", "7", "10", "12", "13"}
    assert set(clinician_df["clinician_review_bucket"]) == {
        "cpb_positive_aortic_or_vascular",
        "respiratory_plus_cpb",
        "unresolved_prefix_or_title",
        "other_cpb_discordant_nonvascular_nonrespiratory",
        "other_prefix_level_review",
    }
    assert {
        "audit_reason_code",
        "clinician_review_bucket",
        "clinician_review_priority",
        "same4_neighbor_prefix",
        "same4_neighbor_label",
        "final_noncardiac_action",
        "final_noncardiac_note",
    } <= set(manual_df.columns)
    unresolved_row = manual_df.loc[manual_df["op_id"].astype(str) == "10"].iloc[0]
    assert unresolved_row["final_noncardiac_action"] == "retain"
    assert manuscript_df.to_dict("records") == [
        {
            "item": "Cardiothoracic Surgery-labeled operations reviewed",
            "n_ops": 11,
            "pct_of_ct_department_ops": "100.00%",
            "notes": "Administrative CTS subset audited at the operation level; department was not used alone to define operative phenotype.",
        },
        {
            "item": "Definite cardiac procedures excluded",
            "n_ops": 1,
            "pct_of_ct_department_ops": "9.09%",
            "notes": "Official Heart and Great Vessels families or explicit cardiac title families excluded from the final noncardiac cohort.",
        },
        {
            "item": "Clearly noncardiac thoracic or chest procedures retained",
            "n_ops": 3,
            "pct_of_ct_department_ops": "27.27%",
            "notes": "Respiratory plus thoracic, mediastinal, foregut, and chest-related noncardiac families retained after operation-level adjudication.",
        },
        {
            "item": "Ambiguous CPB-supported or unresolved procedures excluded",
            "n_ops": 4,
            "pct_of_ct_department_ops": "36.36%",
            "notes": "Residual CPB-supported discordant or unresolved CTS families excluded conservatively under the default strict noncardiac rule.",
        },
        {
            "item": "Final retained noncardiac CTS-labeled operations",
            "n_ops": 6,
            "pct_of_ct_department_ops": "54.55%",
            "notes": "CTS-labeled operations retained in the final operation-level noncardiac cohort after adjudication.",
        },
    ]
    assert clinician_df.loc[
        clinician_df["clinician_review_bucket"] == "unresolved_prefix_or_title",
        "recommended_final_action",
    ].iat[0] == "exclude"
    assert manifest["stage"] == "report_procedure_audit"
    assert len(manifest["outputs"]) == len(expected_stems) * 3


def test_build_preop_features_applies_procedure_audit_resolution_on_default_cohort(synthetic_config: Path) -> None:
    config = load_config(synthetic_config)
    assert config["study"]["cohort_key"] == "strict_noncardiac_adult_procedure_audit"
    assert config["cohort"]["procedure_audit_resolution"]["enabled"] is True
    config["reports"]["procedure_audit"]["cms_order_zip_path"] = str(_write_synthetic_cms_order_zip(synthetic_config.parent))

    operations_path = synthetic_config.parent / "raw" / "operations.csv"
    operations_df = pd.read_csv(operations_path)
    operations_df["cpbon_time"] = operations_df["cpbon_time"].astype("object")
    operations_df["cpboff_time"] = operations_df["cpboff_time"].astype("object")
    operations_df.loc[operations_df["op_id"] == 1, ["department", "icd10_pcs"]] = ["CTS", "02110"]
    operations_df.loc[operations_df["op_id"] == 2, ["department", "icd10_pcs"]] = ["CTS", "0BTC4"]
    operations_df.loc[operations_df["op_id"] == 3, ["department", "icd10_pcs", "cpbon_time", "cpboff_time"]] = ["CTS", "04R00", "100", "250"]
    operations_df.loc[operations_df["op_id"] == 4, ["department", "icd10_pcs"]] = ["CTS", "0HB00"]
    operations_df.loc[operations_df["op_id"] == 5, ["department", "icd10_pcs"]] = ["CTS", "0A152"]
    operations_df.loc[operations_df["op_id"] == 6, ["department", "icd10_pcs"]] = ["CTS", "0HB0X"]
    operations_df.to_csv(operations_path, index=False)

    preop_df, audit_df = build_preop_features(config, Path(config["paths"]["raw_inspire_dir"]))

    assert 1 not in preop_df["op_id"].tolist()
    assert 3 not in preop_df["op_id"].tolist()
    assert 5 not in preop_df["op_id"].tolist()
    assert 2 in preop_df["op_id"].tolist()
    assert 4 in preop_df["op_id"].tolist()
    assert 6 in preop_df["op_id"].tolist()
    assert "after_procedure_audit_resolution" in audit_df["step"].tolist()
