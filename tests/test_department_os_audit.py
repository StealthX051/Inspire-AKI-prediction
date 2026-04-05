from __future__ import annotations

from pathlib import Path

import pandas as pd

from inspire_aki.reporting.department_os_audit import (
    _load_context,
    build_patient_level_department_counts,
    compare_department_indicator,
    current_department_label_frame,
    default_reviewer_output_dir,
    load_raw_department_dictionary,
)


def test_current_department_labels_match_raw_dictionary(tmp_path: Path) -> None:
    department_path = tmp_path / "department.csv"
    department_path.write_text(
        "Abbreviations,Full name\n"
        "AN,Anesthesiology\n"
        "CTS,Cardio-Thoracic Surgery\n"
        "DM,Dermatology\n"
        "EM,Emergency Medicine\n"
        "GS,General Surgery\n"
        "IM,Internal Medicine\n"
        "NS,Neurosurgery\n"
        "OG,Obstetrics & Gynecology\n"
        "OL,Oto-laryngology\n"
        "OS,Orthopedic Surgery\n"
        "OT,Ophthalmology\n"
        "PED,Pediatrics\n"
        "PS,Plastic Surgery\n"
        "RAD,Radiology\n"
        "RO,Radiation Oncology\n"
        "UR,Urology\n",
        encoding="utf-8",
    )

    raw_dictionary = load_raw_department_dictionary(department_path)
    labels = current_department_label_frame(raw_dictionary["department_code"].tolist())
    merged = raw_dictionary.merge(labels, on="department_code", how="left")

    assert merged["raw_dictionary_label"].tolist() == merged["current_report_label"].tolist()
    assert merged["raw_dictionary_label"].tolist() == merged["current_procedure_audit_label"].tolist()


def test_compare_department_indicator_detects_exact_match_and_mismatch() -> None:
    audit_df = pd.DataFrame(
        {
            "op_id": [1, 2, 3],
            "raw_department": ["OS", "OT", "OS"],
            "department_OS": [1, 0, 0],
            "department_OT": [0, 1, 0],
        }
    )

    os_check = compare_department_indicator(audit_df, "OS")
    ot_check = compare_department_indicator(audit_df, "OT")

    assert os_check["indicator_positive_n"] == 1
    assert os_check["raw_positive_n"] == 2
    assert os_check["mismatch_n"] == 1
    assert os_check["positive_mismatch_n"] == 0
    assert os_check["matches_exactly"] is False
    assert ot_check["mismatch_n"] == 0
    assert ot_check["matches_exactly"] is True


def test_build_patient_level_department_counts_uses_last_operation_per_subject() -> None:
    audit_df = pd.DataFrame(
        {
            "op_id": [10, 20, 30],
            "subject_id": [1001, 1001, 1002],
            "department_GS": [1, 0, 0],
            "department_OS": [0, 1, 0],
            "department_OT": [0, 0, 1],
        }
    )

    patient_counts = build_patient_level_department_counts(audit_df).set_index("department_code")

    assert int(patient_counts.loc["GS", "n_patients"]) == 0
    assert int(patient_counts.loc["OS", "n_patients"]) == 1
    assert int(patient_counts.loc["OT", "n_patients"]) == 1
    assert float(patient_counts.loc["OS", "pct_patients"]) == 50.0
    assert float(patient_counts.loc["OT", "pct_patients"]) == 50.0


def test_department_audit_default_output_dir_uses_artifact_root(synthetic_config: Path) -> None:
    context = _load_context(config_path=synthetic_config, raw_dir=None, artifacts_dir=None, out_dir=None)

    assert context.out_dir == default_reviewer_output_dir(context.paths)
