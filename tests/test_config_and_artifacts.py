from __future__ import annotations

import pandas as pd
import pytest

from inspire_aki.config import config_hash, load_config, validate_config
from inspire_aki.io.artifacts import ArtifactManager


def test_load_config_merges_override_and_preserves_defaults(synthetic_config) -> None:
    config = load_config(synthetic_config)
    assert config["paths"]["artifacts_dir"].endswith("artifacts")
    assert config["features"]["preop_lab_items"] == ["creatinine", "sodium"]
    assert config["study"]["cohort_key"] == "strict_noncardiac_adult_procedure_audit"
    assert config["study"]["outcome_key"] == "aki"
    assert config["cohort"]["min_age"] == 18
    assert "02" in config["cohort"]["exclude_icd10_prefixes"]
    assert config["cohort"]["procedure_audit_resolution"]["enabled"] is True
    assert config["outcome"]["target_column"] == "aki_boolean"
    assert config["models"]["target"] == "aki_boolean"


def test_config_hash_changes_with_override(synthetic_config) -> None:
    config_a = load_config(synthetic_config)
    config_b = load_config(synthetic_config)
    config_b["splits"]["random_state"] = 999
    assert config_hash(config_a) == config_hash(load_config(synthetic_config))
    assert config_hash(config_a) != config_hash(config_b)


def test_load_config_normalizes_legacy_shap_key_and_removes_dead_compat(synthetic_config) -> None:
    config = load_config(synthetic_config)
    assert config["evaluation_mode"] == "grouped_nested_cv"
    assert "batch_shap_jobs" not in config["reports"]
    assert config["reports"]["shap_jobs"] == []
    assert config["reports"]["procedure_audit"]["ct_department_code"] == "CTS"
    assert config["reports"]["procedure_audit"]["definite_cardiac_prefixes"] == ["02"]
    assert config["reports"]["manuscript_sections"] == ["consort", "tables", "curves", "statistics", "reclassification", "shap"]
    assert config["reports"]["table_formats"] == ["html", "md", "csv"]
    assert config["reports"]["figure_formats"] == ["png", "svg"]
    assert config["reports"]["figure_png_dpi"] == 600
    assert "compat" not in config or "export_legacy_aliases" not in config.get("compat", {})


def test_default_config_validates() -> None:
    config = load_config()
    assert config["paths"]["artifacts_dir"] == "/media/volume/ncs_inspire_data/ncs_aki/artifacts/default"
    assert config["study"]["cohort_key"] == "strict_noncardiac_adult_procedure_audit"
    assert "02" in config["cohort"]["exclude_icd10_prefixes"]
    assert config["cohort"]["procedure_audit_resolution"]["enabled"] is True
    assert "default_noncardiac_adult" in config["cohorts"]["profiles"]
    assert config["cohorts"]["profiles"]["default_noncardiac_adult"]["procedure_audit_resolution"]["enabled"] is False
    assert config["evaluation_mode"] == "grouped_nested_cv"
    assert config["reports"]["shap_jobs"]
    assert all(job["plots"] == ["beeswarm", "scatter"] for job in config["reports"]["shap_jobs"])
    assert config["reports"]["featured_shap_scatter_features"] == [
        "BSA",
        "max_hr",
        "max_inibg_mbp",
        "op_len",
        "preop_hb",
        "ward_rr",
    ]
    assert config["reports"]["procedure_audit"]["cms_order_zip_path"] == "external/cms_icd10pcs/april-1-2026-icd10pcs-order.zip"
    assert config["reports"]["manuscript_sections"] == ["consort", "tables", "curves", "statistics", "reclassification", "shap"]
    assert config["reports"]["figure_png_dpi"] == 600
    assert config["runtime"]["profile"] == "throughput"
    assert config["runtime"]["orchestration"]["mode"] == "overlap"
    assert config["runtime"]["progress_interval_seconds"] == 60
    assert config["models"]["hpo"]["sequence_batch_size"] == 4096
    assert config["models"]["autogluon"]["num_cpus"] == 32
    assert config["models"]["autogluon"]["num_gpus"] == "auto"


def test_smoke_config_validates_and_is_lightweight() -> None:
    config = load_config("configs/aki/smoke.yaml")
    assert config["paths"]["artifacts_dir"] == "/media/volume/ncs_inspire_data/ncs_aki/artifacts/smoke"
    assert config["evaluation_mode"] == "grouped_holdout"
    assert config["runtime"]["profile"] == "balanced"
    assert config["runtime"]["orchestration"]["mode"] == "serial"
    assert config["splits"]["use_bootstrapping"] is False
    assert config["models"]["tabular_enabled"] == ["log_reg"]
    assert config["models"]["sequence_enabled"] == ["lstm_only"]
    assert config["models"]["tabular_hpo_enabled"] == []
    assert config["models"]["sequence_hpo_enabled"] == []
    assert config["reports"]["shap_jobs"] == [
        {
            "run_name": "LogReg_Combined_Smoke",
            "model_key": "log_reg",
            "dataset_regime": "combined",
            "plots": ["beeswarm"],
            "scatter_features": [],
            "dependence_pairs": [],
        }
    ]


def test_smoke_hpo_config_validates_and_limits_trials() -> None:
    config = load_config("configs/aki/smoke_hpo.yaml")
    assert config["paths"]["artifacts_dir"] == "/media/volume/ncs_inspire_data/ncs_aki/artifacts/smoke_hpo"
    assert config["evaluation_mode"] == "grouped_holdout"
    assert config["runtime"]["profile"] == "balanced"
    assert config["runtime"]["orchestration"]["mode"] == "serial"
    assert config["models"]["hpo"]["n_trials"] == 1
    assert config["models"]["hpo"]["tabular_mlp_epochs"] == 10
    assert config["models"]["hpo"]["sequence_epochs"] == 3
    assert config["models"]["hpo"]["sequence_patience"] == 5
    assert config["models"]["hpo"]["sequence_batch_size"] == 1024
    assert config["models"]["tabular_hpo_enabled"] == ["log_reg", "xgb", "rf", "svm", "mlp", "knn"]
    assert config["models"]["sequence_hpo_enabled"] == ["lstm_only", "hybrid"]


def test_strict_noncardiac_review_config_remains_a_strict_alias_with_separate_artifact_root() -> None:
    default_config = load_config("configs/aki/default.yaml")
    strict_config = load_config("configs/aki/strict_noncardiac_review.yaml")

    assert default_config["study"]["cohort_key"] == "strict_noncardiac_adult_procedure_audit"
    assert "02" in default_config["cohort"]["exclude_icd10_prefixes"]
    assert default_config["cohort"]["procedure_audit_resolution"]["enabled"] is True

    assert strict_config["study"]["cohort_key"] == "strict_noncardiac_adult_procedure_audit"
    assert strict_config["paths"]["artifacts_dir"] == "/media/volume/ncs_inspire_data/ncs_aki/artifacts/strict_noncardiac_review"
    assert "02" in strict_config["cohort"]["exclude_icd10_prefixes"]
    assert strict_config["reports"]["procedure_audit"]["definite_cardiac_prefixes"] == ["02"]
    assert strict_config["cohort"]["procedure_audit_resolution"]["enabled"] is True
    assert strict_config["cohort"]["procedure_audit_resolution"]["exclude_manual_review_buckets"] == [
        "cpb_positive_aortic_or_vascular",
        "respiratory_plus_cpb",
        "other_cpb_discordant_nonvascular_nonrespiratory",
        "other_prefix_level_review",
    ]


def test_reviewer_combined_xgb_baseline_config_is_narrow_and_valid() -> None:
    config = load_config("configs/aki/reviewer_combined_xgb_baseline.yaml")

    assert config["paths"]["artifacts_dir"] == "/media/volume/ncs_inspire_data/ncs_aki/artifacts/reviewer_combined_xgb_baseline"
    assert config["evaluation_mode"] == "grouped_nested_cv"
    assert config["models"]["tabular_enabled"] == ["xgb"]
    assert config["models"]["sequence_enabled"] == []
    assert config["models"]["tabular_hpo_enabled"] == []
    assert config["models"]["sequence_hpo_enabled"] == []
    assert config["reports"]["shap_jobs"] == [
        {
            "run_name": "XGBoost_Combined_Reviewer_Baseline",
            "model_key": "xgb",
            "dataset_regime": "combined",
            "plots": ["beeswarm"],
            "scatter_features": [],
            "dependence_pairs": [],
        }
    ]


def test_macce_default_config_validates_and_switches_to_grouped_holdout() -> None:
    config = load_config("configs/macce/default.yaml")

    assert config["paths"]["artifacts_dir"] == "/media/volume/ncs_inspire_data/ncs_aki/artifacts/macce_default"
    assert config["study"]["cohort_key"] == "strict_noncardiac_adult_procedure_audit"
    assert config["study"]["outcome_key"] == "macce"
    assert config["evaluation_mode"] == "grouped_holdout"
    assert config["splits"]["use_bootstrapping"] is False
    assert config["outcome"]["display_name"] == "30-day MACCE"
    assert config["models"]["target"] == "macce"


def test_macce_five_fold_config_validates_and_switches_to_grouped_nested_cv() -> None:
    config = load_config("configs/macce/five_fold.yaml")

    assert config["paths"]["artifacts_dir"] == "/media/volume/ncs_inspire_data/ncs_aki/artifacts/macce_5fold_cv"
    assert config["study"]["cohort_key"] == "strict_noncardiac_adult_procedure_audit"
    assert config["study"]["outcome_key"] == "macce"
    assert config["evaluation_mode"] == "grouped_nested_cv"
    assert config["splits"]["n_cv_folds"] == 5
    assert config["models"]["target"] == "macce"


def test_macce_smoke_configs_validate_and_limit_trials() -> None:
    smoke = load_config("configs/macce/smoke.yaml")
    smoke_hpo = load_config("configs/macce/smoke_hpo.yaml")

    assert smoke["paths"]["artifacts_dir"] == "/media/volume/ncs_inspire_data/ncs_aki/artifacts/macce_smoke"
    assert smoke["study"]["cohort_key"] == "strict_noncardiac_adult_procedure_audit"
    assert smoke["study"]["outcome_key"] == "macce"
    assert smoke["evaluation_mode"] == "grouped_holdout"
    assert smoke["models"]["target"] == "macce"
    assert smoke["models"]["tabular_enabled"] == ["log_reg"]
    assert smoke["reports"]["shap_jobs"] == [
        {
            "run_name": "LogReg_Combined_Macce_Smoke",
            "model_key": "log_reg",
            "dataset_regime": "combined",
            "plots": ["beeswarm"],
            "scatter_features": [],
            "dependence_pairs": [],
        }
    ]

    assert smoke_hpo["paths"]["artifacts_dir"] == "/media/volume/ncs_inspire_data/ncs_aki/artifacts/macce_smoke_hpo"
    assert smoke_hpo["study"]["cohort_key"] == "strict_noncardiac_adult_procedure_audit"
    assert smoke_hpo["study"]["outcome_key"] == "macce"
    assert smoke_hpo["evaluation_mode"] == "grouped_holdout"
    assert smoke_hpo["models"]["target"] == "macce"
    assert smoke_hpo["models"]["hpo"]["n_trials"] == 1
    assert smoke_hpo["models"]["tabular_hpo_enabled"] == ["log_reg", "xgb", "rf", "svm", "mlp", "knn"]
    assert smoke_hpo["models"]["sequence_hpo_enabled"] == ["lstm_only", "hybrid"]


def test_legacy_noncardiac_profile_remains_selectable_explicitly() -> None:
    config = load_config()
    config["study"]["cohort_key"] = "default_noncardiac_adult"
    config["cohort"] = config["cohorts"]["profiles"]["default_noncardiac_adult"].copy()

    validate_config(config)

    assert config["cohort"]["procedure_audit_resolution"]["enabled"] is False
    assert "02" not in config["cohort"]["exclude_icd10_prefixes"]


def test_validate_config_rejects_unsupported_shap_jobs(loaded_synthetic_config) -> None:
    loaded_synthetic_config["reports"]["shap_jobs"] = [{"dataset_regime": "combined", "model_key": "svm"}]
    with pytest.raises(ValueError, match="Unsupported SHAP model_key"):
        validate_config(loaded_synthetic_config)


def test_validate_config_rejects_unsupported_shap_plot_family(loaded_synthetic_config) -> None:
    loaded_synthetic_config["reports"]["shap_jobs"] = [
        {"dataset_regime": "combined", "model_key": "log_reg", "plots": ["beeswarm", "unknown"]}
    ]
    with pytest.raises(ValueError, match="Unsupported SHAP plot families"):
        validate_config(loaded_synthetic_config)


def test_validate_config_rejects_malformed_shap_dependence_pairs(loaded_synthetic_config) -> None:
    loaded_synthetic_config["reports"]["shap_jobs"] = [
        {
            "dataset_regime": "combined",
            "model_key": "log_reg",
            "plots": ["dependence"],
            "dependence_pairs": [{"main_feature": "creatinine"}],
        }
    ]
    with pytest.raises(ValueError, match="main_feature and interaction_feature"):
        validate_config(loaded_synthetic_config)


def test_validate_config_rejects_nonlist_featured_shap_scatter_features(loaded_synthetic_config) -> None:
    loaded_synthetic_config["reports"]["featured_shap_scatter_features"] = "BSA"
    with pytest.raises(ValueError, match="featured_shap_scatter_features"):
        validate_config(loaded_synthetic_config)


def test_validate_config_rejects_invalid_bootstrap_ratio(loaded_synthetic_config) -> None:
    loaded_synthetic_config["splits"]["n_bootstrap_iterations"] = 5
    loaded_synthetic_config["splits"]["n_cv_folds"] = 2
    with pytest.raises(ValueError, match="n_bootstrap_iterations"):
        validate_config(loaded_synthetic_config)


def test_validate_config_rejects_invalid_evaluation_mode(loaded_synthetic_config) -> None:
    loaded_synthetic_config["evaluation_mode"] = "invalid"
    with pytest.raises(ValueError, match="evaluation_mode"):
        validate_config(loaded_synthetic_config)


def test_validate_config_rejects_unknown_outcome_key(loaded_synthetic_config) -> None:
    loaded_synthetic_config["study"]["outcome_key"] = "unknown"
    with pytest.raises(ValueError, match="Unknown study.outcome_key"):
        validate_config(loaded_synthetic_config)


def test_validate_config_rejects_invalid_hpo_trial_count(loaded_synthetic_config) -> None:
    loaded_synthetic_config["models"]["hpo"] = {"n_trials": 0}
    with pytest.raises(ValueError, match="n_trials"):
        validate_config(loaded_synthetic_config)


def test_validate_config_rejects_invalid_runtime_orchestration_mode(loaded_synthetic_config) -> None:
    loaded_synthetic_config["runtime"]["orchestration"]["mode"] = "invalid"
    with pytest.raises(ValueError, match="runtime.orchestration.mode"):
        validate_config(loaded_synthetic_config)


def test_validate_config_rejects_invalid_runtime_progress_interval(loaded_synthetic_config) -> None:
    loaded_synthetic_config["runtime"]["progress_interval_seconds"] = 0
    with pytest.raises(ValueError, match="runtime.progress_interval_seconds"):
        validate_config(loaded_synthetic_config)


def test_validate_config_rejects_invalid_sequence_batch_size(loaded_synthetic_config) -> None:
    loaded_synthetic_config["models"]["hpo"] = {
        "n_trials": 1,
        "tabular_mlp_epochs": 1,
        "sequence_epochs": 1,
        "sequence_patience": 1,
        "sequence_batch_size": 0,
    }
    with pytest.raises(ValueError, match="sequence_batch_size"):
        validate_config(loaded_synthetic_config)


def test_validate_config_rejects_invalid_autogluon_num_gpus(loaded_synthetic_config) -> None:
    loaded_synthetic_config["models"]["autogluon"]["num_gpus"] = -1
    with pytest.raises(ValueError, match="models.autogluon.num_gpus"):
        validate_config(loaded_synthetic_config)


def test_artifact_manager_round_trips_and_writes_manifest(loaded_synthetic_config) -> None:
    artifacts = ArtifactManager(loaded_synthetic_config)
    frame = pd.DataFrame({"op_id": [1, 2], "value": [0.1, 0.2]})
    csv_path = artifacts.write_dataframe(frame, "unit", "frame.csv")
    parquet_path = artifacts.write_dataframe(frame, "unit", "frame.parquet")
    json_path = artifacts.write_json({"stage": "unit"}, "unit", "payload.json")
    pickle_path = artifacts.write_pickle({"numbers": [1, 2, 3]}, "unit", "payload.pkl")
    manifest_path = artifacts.write_manifest(
        "unit_test",
        ["unit", "manifest.json"],
        inputs=["tests/input.csv"],
        outputs=[artifacts.relative(csv_path)],
        metadata={"rows": len(frame)},
    )

    pd.testing.assert_frame_equal(artifacts.read_dataframe("unit", "frame.csv"), frame)
    pd.testing.assert_frame_equal(artifacts.read_dataframe("unit", "frame.parquet"), frame)
    assert artifacts.read_json("unit", "payload.json") == {"stage": "unit"}
    assert artifacts.read_pickle("unit", "payload.pkl") == {"numbers": [1, 2, 3]}
    manifest = artifacts.read_json("unit", "manifest.json")
    assert manifest["stage"] == "unit_test"
    assert manifest["metadata"]["rows"] == 2
    assert csv_path.exists() and parquet_path.exists() and json_path.exists() and pickle_path.exists() and manifest_path.exists()
