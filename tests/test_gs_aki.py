from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest

from inspire_aki.clinical_baselines.gs_aki import (
    build_gs_aki_features,
    derive_gs_aki_diagnosis_features,
    load_intraperitoneal_proxy_map,
    score_gs_aki_counts,
)
from inspire_aki.clinical_baselines.intraperitoneal_map_builder import (
    build_intraperitoneal_proxy_outputs,
    collapse_cdc_map_to_code5,
)
from inspire_aki.config import validate_config
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.pipelines.evaluate import run_calibration
from inspire_aki.pipelines.evaluate_generate import run_evaluate_generate
from inspire_aki.pipelines.preprocess import run_intraop, run_labels, run_preop, run_tabular
from inspire_aki.pipelines.report import run_tables
from inspire_aki.pipelines.train import _tabular_model_dataset, run_train_tabular
from inspire_aki.reporting.tables import _performance_table_spec


def _enabled_gs_aki_config(config: dict) -> dict:
    updated = copy.deepcopy(config)
    updated["models"]["tabular_enabled"] = [
        *[model_key for model_key in updated["models"]["tabular_enabled"] if model_key != "gs_aki_rule"],
        "gs_aki_rule",
    ]
    updated["models"]["tabular_hpo_enabled"] = [
        model_key for model_key in updated["models"]["tabular_hpo_enabled"] if model_key != "gs_aki_rule"
    ]
    return updated


def _minimal_gs_aki_config(base_config: dict, *, raw_dir: Path, map_path: Path) -> dict:
    config = copy.deepcopy(base_config)
    config["paths"]["raw_inspire_dir"] = str(raw_dir)
    config["clinical_baselines"]["gs_aki"]["intraperitoneal_map_path"] = str(map_path)
    return config


def _write_minimal_raw_inputs(raw_dir: Path, operations_df: pd.DataFrame, diagnosis_df: pd.DataFrame) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    operations_df.to_csv(raw_dir / "operations.csv", index=False)
    diagnosis_df.to_csv(raw_dir / "diagnosis.csv", index=False)


def _performance_row(dataset_regime: str, model_key: str, model_name: str) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset_regime": dataset_regime,
        "population_id": dataset_regime,
        "model_key": model_key,
        "model_name": model_name,
    }
    for metric in ("auroc", "auprc", "sensitivity", "specificity", "precision", "f_score", "balanced_accuracy"):
        row[metric] = 0.5
        row[f"{metric}_display"] = "0.500"
        row[f"{metric}_ci_display"] = "(0.400, 0.600)"
    return row


def test_score_gs_aki_counts_maps_published_classes(loaded_synthetic_config) -> None:
    counts = pd.Series([0, 2, 3, 4, 5, 6, 9], name="gs_aki_count")

    labels = score_gs_aki_counts(counts, loaded_synthetic_config)

    assert labels.tolist() == ["I", "I", "II", "III", "IV", "V", "V"]


def test_derive_gs_aki_diagnosis_features_strictly_filters_preop_and_repeated_subject(loaded_synthetic_config) -> None:
    base_df = pd.DataFrame(
        [
            {"op_id": 1, "subject_id": 101, "opstart_time": 100},
            {"op_id": 2, "subject_id": 101, "opstart_time": 200},
            {"op_id": 3, "subject_id": 202, "opstart_time": 300},
        ]
    )
    diagnosis_df = pd.DataFrame(
        [
            {"subject_id": 101, "chart_time": 50, "icd10_cm": "E11.9"},
            {"subject_id": 101, "chart_time": 60, "icd10_cm": "I10"},
            {"subject_id": 101, "chart_time": 150, "icd10_cm": "I50.1"},
            {"subject_id": 101, "chart_time": 180, "icd10_cm": "R18.8"},
            {"subject_id": 101, "chart_time": 200, "icd10_cm": "E11.9"},
            {"subject_id": 202, "chart_time": 290, "icd10_cm": "I50.1"},
            {"subject_id": 202, "chart_time": 301, "icd10_cm": "I10"},
        ]
    )

    features = derive_gs_aki_diagnosis_features(
        base_df=base_df,
        diagnosis_df=diagnosis_df,
        config=loaded_synthetic_config,
    ).sort_values("op_id", kind="stable")

    rows = {int(row.op_id): row for row in features.itertuples(index=False)}
    assert rows[1].gs_aki_diabetes == 1
    assert rows[1].gs_aki_hypertension == 1
    assert rows[1].gs_aki_chf_30d == 0
    assert rows[1].gs_aki_ascites_30d == 0
    assert rows[2].gs_aki_diabetes == 1
    assert rows[2].gs_aki_hypertension == 1
    assert rows[2].gs_aki_chf_30d == 1
    assert rows[2].gs_aki_ascites_30d == 1
    assert rows[3].gs_aki_diabetes == 0
    assert rows[3].gs_aki_hypertension == 0
    assert rows[3].gs_aki_chf_30d == 1
    assert rows[3].gs_aki_ascites_30d == 0


def test_build_gs_aki_features_applies_renal_boundaries(loaded_synthetic_config, tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    operations_df = pd.DataFrame(
        [
            {"op_id": 1, "subject_id": 1, "icd10_pcs": "0WJF4"},
            {"op_id": 2, "subject_id": 2, "icd10_pcs": "0WJF4"},
            {"op_id": 3, "subject_id": 3, "icd10_pcs": "0WJF4"},
            {"op_id": 4, "subject_id": 4, "icd10_pcs": "0WJF4"},
            {"op_id": 5, "subject_id": 5, "icd10_pcs": "0WJF4"},
        ]
    )
    diagnosis_df = pd.DataFrame(columns=["subject_id", "chart_time", "icd10_cm"])
    _write_minimal_raw_inputs(raw_dir, operations_df, diagnosis_df)

    map_path = tmp_path / "intraperitoneal.csv"
    pd.DataFrame(
        [
            {
                "icd10_pcs_5char": "0WJF4",
                "approach": "4",
                "nhsn_category": "HER",
                "intraperitoneal_proxy": 0,
                "source": "unit_test",
                "rationale": "negative_control",
            }
        ]
    ).to_csv(map_path, index=False)
    config = _minimal_gs_aki_config(loaded_synthetic_config, raw_dir=raw_dir, map_path=map_path)
    preop_df = pd.DataFrame(
        [
            {"op_id": 1, "subject_id": 1, "age": 40, "sex": "F", "emop": 0, "opstart_time": 1000, "preop_creatinine": 1.19},
            {"op_id": 2, "subject_id": 2, "age": 40, "sex": "F", "emop": 0, "opstart_time": 1000, "preop_creatinine": 1.20},
            {"op_id": 3, "subject_id": 3, "age": 40, "sex": "F", "emop": 0, "opstart_time": 1000, "preop_creatinine": 1.99},
            {"op_id": 4, "subject_id": 4, "age": 40, "sex": "F", "emop": 0, "opstart_time": 1000, "preop_creatinine": 2.00},
            {"op_id": 5, "subject_id": 5, "age": 40, "sex": "F", "emop": 0, "opstart_time": 1000, "preop_creatinine": 4.50},
        ]
    )

    features_df, _audit_df = build_gs_aki_features(config, raw_dir, preop_df)

    assert features_df["gs_aki_renal_mild"].tolist() == [0, 1, 1, 0, 0]
    assert features_df["gs_aki_renal_moderate"].tolist() == [0, 0, 0, 1, 1]
    assert features_df["gs_aki_renal_insufficiency"].tolist() == [0, 1, 1, 1, 1]


def test_load_intraperitoneal_proxy_map_validates_required_columns_and_binary_values(
    loaded_synthetic_config,
    tmp_path: Path,
) -> None:
    missing_cols_path = tmp_path / "missing_cols.csv"
    pd.DataFrame([{"icd10_pcs_5char": "0DTP0", "intraperitoneal_proxy": 1}]).to_csv(missing_cols_path, index=False)
    missing_cfg = copy.deepcopy(loaded_synthetic_config)
    missing_cfg["clinical_baselines"]["gs_aki"]["intraperitoneal_map_path"] = str(missing_cols_path)
    with pytest.raises(ValueError, match="missing required columns"):
        load_intraperitoneal_proxy_map(missing_cfg)

    invalid_value_path = tmp_path / "invalid_value.csv"
    pd.DataFrame(
        [
            {
                "icd10_pcs_5char": "0DTP0",
                "approach": "0",
                "nhsn_category": "REC",
                "intraperitoneal_proxy": 2,
                "source": "unit_test",
                "rationale": "invalid",
            }
        ]
    ).to_csv(invalid_value_path, index=False)
    invalid_cfg = copy.deepcopy(loaded_synthetic_config)
    invalid_cfg["clinical_baselines"]["gs_aki"]["intraperitoneal_map_path"] = str(invalid_value_path)
    with pytest.raises(ValueError, match="invalid intraperitoneal_proxy values"):
        load_intraperitoneal_proxy_map(invalid_cfg)


def test_build_gs_aki_features_fails_when_mapping_does_not_cover_retained_codes(
    loaded_synthetic_config,
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    operations_df = pd.DataFrame([{"op_id": 1, "subject_id": 1, "icd10_pcs": "0DTP0"}])
    diagnosis_df = pd.DataFrame(columns=["subject_id", "chart_time", "icd10_cm"])
    _write_minimal_raw_inputs(raw_dir, operations_df, diagnosis_df)

    map_path = tmp_path / "intraperitoneal.csv"
    pd.DataFrame(
        [
            {
                "icd10_pcs_5char": "0WJF4",
                "approach": "4",
                "nhsn_category": "HER",
                "intraperitoneal_proxy": 0,
                "source": "unit_test",
                "rationale": "missing_code",
            }
        ]
    ).to_csv(map_path, index=False)
    config = _minimal_gs_aki_config(loaded_synthetic_config, raw_dir=raw_dir, map_path=map_path)
    preop_df = pd.DataFrame(
        [{"op_id": 1, "subject_id": 1, "age": 40, "sex": "F", "emop": 0, "opstart_time": 1000, "preop_creatinine": 1.0}]
    )

    with pytest.raises(ValueError, match="did not cover every retained operation code"):
        build_gs_aki_features(config, raw_dir, preop_df)


def test_build_gs_aki_features_excludes_missing_required_inputs_before_label_alignment(
    loaded_synthetic_config,
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "raw"
    operations_df = pd.DataFrame(
        [
            {"op_id": 1, "subject_id": 1, "icd10_pcs": "0DTP0"},
            {"op_id": 2, "subject_id": 2, "icd10_pcs": "0DTP0"},
        ]
    )
    diagnosis_df = pd.DataFrame(columns=["subject_id", "chart_time", "icd10_cm"])
    _write_minimal_raw_inputs(raw_dir, operations_df, diagnosis_df)

    map_path = tmp_path / "intraperitoneal.csv"
    pd.DataFrame(
        [
            {
                "icd10_pcs_5char": "0DTP0",
                "approach": "0",
                "nhsn_category": "REC",
                "intraperitoneal_proxy": 1,
                "source": "unit_test",
                "rationale": "positive_control",
            }
        ]
    ).to_csv(map_path, index=False)
    config = _minimal_gs_aki_config(loaded_synthetic_config, raw_dir=raw_dir, map_path=map_path)
    preop_df = pd.DataFrame(
        [
            {"op_id": 1, "subject_id": 1, "age": 40, "sex": "F", "emop": 0, "opstart_time": 1000, "preop_creatinine": 1.0},
            {"op_id": 2, "subject_id": 2, "age": 50, "sex": "M", "emop": 1, "opstart_time": 2000, "preop_creatinine": pd.NA},
        ]
    )

    features_df, audit_df = build_gs_aki_features(config, raw_dir, preop_df)

    assert features_df["op_id"].tolist() == [1]
    assert "gs_aki_missing_required_inputs_excluded" in audit_df["step"].tolist()


def test_cdc_collapse_fails_on_conflicting_code5_labels() -> None:
    cdc_df = pd.DataFrame(
        [
            {"nhsn_category": "HYST", "pcs7": "0AB9000"},
            {"nhsn_category": "AAA", "pcs7": "0AB9001"},
        ]
    )

    with pytest.raises(ValueError, match="Conflicting CDC/NHSN-derived intraperitoneal labels"):
        collapse_cdc_map_to_code5(cdc_df)


def test_intraperitoneal_builder_fails_when_unmatched_positive_keywords_exceed_threshold() -> None:
    observed_counts_df = pd.DataFrame([{"icd10_pcs_5char": "0ZZZ0", "n_ops": 5}])
    cdc_collapsed_df = pd.DataFrame(
        columns=["icd10_pcs_5char", "approach", "nhsn_category", "intraperitoneal_proxy", "source", "rationale"]
    )
    cms_titles_df = pd.DataFrame(
        [
            {
                "pcs7": "0ZZZ000",
                "code5": "0ZZZ0",
                "short_title": "stomach excision",
                "long_title": "Excision of stomach, open approach",
            }
        ]
    )

    with pytest.raises(ValueError, match="positive intraperitoneal keywords exceeded"):
        build_intraperitoneal_proxy_outputs(
            observed_counts_df=observed_counts_df,
            cdc_collapsed_df=cdc_collapsed_df,
            cms_titles_df=cms_titles_df,
        )


def test_validate_config_rejects_gs_aki_rule_for_hpo_and_non_aki_outcomes(loaded_synthetic_config) -> None:
    hpo_cfg = _enabled_gs_aki_config(loaded_synthetic_config)
    hpo_cfg["models"]["tabular_hpo_enabled"] = ["gs_aki_rule"]
    with pytest.raises(ValueError, match="cannot be added to models.tabular_hpo_enabled"):
        validate_config(hpo_cfg)

    macce_cfg = _enabled_gs_aki_config(loaded_synthetic_config)
    macce_cfg["study"]["outcome_key"] = "macce"
    macce_cfg["outcome"] = copy.deepcopy(macce_cfg["outcomes"]["catalog"]["macce"])
    macce_cfg["models"]["target"] = macce_cfg["outcome"]["target_column"]
    with pytest.raises(ValueError, match="only supported for the AKI outcome"):
        validate_config(macce_cfg)


def test_performance_table_orders_preop_asa_then_gs_aki_and_suppresses_rules_elsewhere() -> None:
    summary_df = pd.DataFrame(
        [
            _performance_row("preop", "log_reg", "Logistic Regression"),
            _performance_row("preop", "gs_aki_rule", "Adapted GS-AKI"),
            _performance_row("preop", "asa_rule", "ASA Rule"),
            _performance_row("intraop", "log_reg", "Logistic Regression"),
            _performance_row("intraop", "gs_aki_rule", "Adapted GS-AKI"),
            _performance_row("intraop", "asa_rule", "ASA Rule"),
        ]
    )

    spec = _performance_table_spec(
        summary_df,
        file_stem="performance_table",
        title="Performance Metrics",
        caption="unit-test ordering",
    )
    sections = {
        str(section.csv_df["dataset_regime"].iloc[0]): (
            section.csv_df.reset_index(drop=True),
            section.display_df.reset_index(drop=True),
        )
        for section in spec.sections
    }

    preop_csv, preop_display = sections["preop"]
    intraop_csv, _intraop_display = sections["intraop"]

    assert preop_csv["model_key"].tolist() == ["asa_rule", "gs_aki_rule", "log_reg"]
    assert intraop_csv["model_key"].tolist() == ["log_reg"]

    asa_row = preop_display.loc[preop_csv["model_key"] == "asa_rule"].iloc[0]
    gs_aki_row = preop_display.loc[preop_csv["model_key"] == "gs_aki_rule"].iloc[0]
    assert asa_row["sensitivity"] == "0.500"
    assert gs_aki_row["sensitivity"] == "—"
    assert gs_aki_row["specificity"] == "—"
    assert gs_aki_row["precision"] == "—"
    assert gs_aki_row["f_score"] == "—"
    assert gs_aki_row["balanced_accuracy"] == "—"


def test_gs_aki_rule_uses_dedicated_dataset_and_writes_incidence_outputs(synthetic_config: Path) -> None:
    from inspire_aki.config import load_config

    config = _enabled_gs_aki_config(load_config(synthetic_config))
    artifacts = ArtifactManager(config)

    run_preop(config)
    run_intraop(config)
    run_tabular(config)
    run_labels(config)

    preop_labeled = pd.read_csv(artifacts.paths.artifact_path("datasets", "tabular", "tabular_preop_labeled.csv"))
    gs_aki_labeled_path = artifacts.paths.artifact_path("datasets", "tabular", "tabular_gs_aki_labeled.csv")
    gs_aki_labeled = pd.read_csv(gs_aki_labeled_path)
    dedicated_df, dedicated_path = _tabular_model_dataset(
        artifacts=artifacts,
        dataset_regime="preop",
        model_key="gs_aki_rule",
        default_dataset_df=preop_labeled,
    )

    assert "gs_aki_count" not in preop_labeled.columns
    assert "gs_aki_count" in gs_aki_labeled.columns
    assert dedicated_path == gs_aki_labeled_path
    assert "gs_aki_count" in dedicated_df.columns

    run_evaluate_generate(config)
    run_train_tabular(config)
    run_calibration(config)
    run_tables(config)

    predictions = pd.read_parquet(artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet"))
    gs_aki_predictions = predictions.loc[predictions["model_key"].astype(str) == "gs_aki_rule"].copy()
    incidence_path = artifacts.paths.artifact_path("reports", "tables", "gs_aki_incidence_table.csv")

    assert not gs_aki_predictions.empty
    assert set(gs_aki_predictions["dataset_regime"].astype(str)) == {"preop"}
    assert set(gs_aki_predictions["calibration_method"].astype(str)) == {"identity_prespecified_class_iii_plus"}
    assert gs_aki_predictions["threshold"].astype(float).nunique() == 1
    assert float(gs_aki_predictions["threshold"].iloc[0]) == pytest.approx(4.0 / 9.0)
    assert incidence_path.exists()
    incidence_df = pd.read_csv(incidence_path)
    assert set(incidence_df["score_type"].astype(str)) == {"count", "class"}
