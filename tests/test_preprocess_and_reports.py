from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from inspire_aki.config import load_config
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.compat import export_legacy_datasets
from inspire_aki.pipelines.evaluate import run_calibration, run_dca, run_delong, run_metrics
from inspire_aki.pipelines.evaluate_generate import run_evaluate_generate
from inspire_aki.pipelines.preprocess import run_intraop, run_labels, run_preop, run_tabular
from inspire_aki.pipelines.report import run_consort, run_tables
from inspire_aki.pipelines.train import run_train_tabular
from inspire_aki.pipelines.tune import run_tune_tabular
from inspire_aki.reporting.consort import generate_consort_outputs
from inspire_aki.reporting.curves import generate_curve_outputs
from inspire_aki.reporting.tables import _performance_summary_frame, _performance_table_spec, generate_table_outputs
from inspire_aki.reporting.rendering import write_table_outputs


def _activate_outcome_config(config: dict, outcome_key: str) -> dict:
    updated = copy.deepcopy(config)
    updated["study"]["outcome_key"] = outcome_key
    updated["outcome"] = copy.deepcopy(updated["outcomes"]["catalog"][outcome_key])
    updated["models"]["target"] = updated["outcome"]["target_column"]
    return updated


def _inject_postop_macce_events(config_path) -> None:
    raw_dir = config_path.parent / "raw"
    diagnosis_path = raw_dir / "diagnosis.csv"
    operations_df = pd.read_csv(raw_dir / "operations.csv", usecols=["subject_id", "opend_time"])
    diagnosis_df = pd.read_csv(diagnosis_path)
    macce_codes = ["I21.9", "I63.9", "I20.0", "I50.1", "I46.0", "I21.4"]
    macce_rows = [
        {
            "subject_id": int(row.subject_id),
            "chart_time": int(row.opend_time) + 60,
            "icd10_cm": macce_codes[idx],
        }
        for idx, row in enumerate(operations_df.tail(len(macce_codes)).itertuples(index=False))
    ]
    pd.concat([diagnosis_df, pd.DataFrame(macce_rows)], ignore_index=True).to_csv(diagnosis_path, index=False)


def _macce_smoke_config(config_path) -> dict:
    config = _activate_outcome_config(load_config(config_path), "macce")
    config["paths"]["artifacts_dir"] = str(config_path.parent / "artifacts_macce")
    config["evaluation_mode"] = "grouped_holdout"
    config["splits"]["use_bootstrapping"] = False
    config["models"]["tabular_enabled"] = ["log_reg"]
    config["models"]["sequence_enabled"] = []
    config["models"]["tabular_hpo_enabled"] = ["log_reg"]
    config["models"]["sequence_hpo_enabled"] = []
    config["models"]["hpo"]["n_trials"] = 1
    return config


def test_preprocess_and_evaluation_artifacts_exist(completed_pipeline) -> None:
    artifacts = ArtifactManager(completed_pipeline)
    expected_paths = [
        artifacts.paths.artifact_path("features", "preop", "preop_features.csv"),
        artifacts.paths.artifact_path("datasets", "tabular", "tabular_combined_labeled.csv"),
        artifacts.paths.artifact_path("features", "timeseries", "time_series_cleaned.csv"),
        artifacts.paths.artifact_path("datasets", "sequence", "lstm_trainable.pkl"),
        artifacts.paths.artifact_path("predictions", "raw", "tabular.parquet"),
        artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet"),
        artifacts.paths.artifact_path("evaluation", "metrics_summary.csv"),
        artifacts.paths.artifact_path("evaluation", "thresholds.csv"),
        artifacts.paths.artifact_path("evaluation", "delong_matrix.csv"),
        artifacts.paths.artifact_path("evaluation", "dca_curves.csv"),
        artifacts.paths.artifact_path("reports", "tables", "performance_table.html"),
        artifacts.paths.artifact_path("reports", "tables", "performance_table.csv"),
        artifacts.paths.artifact_path("reports", "tables", "performance_table.md"),
        artifacts.paths.artifact_path("reports", "tables", "performance_table_calibrated.csv"),
        artifacts.paths.artifact_path("reports", "tables", "consort_audit.md"),
        artifacts.paths.artifact_path("reports", "tables", "consort_audit.html"),
        artifacts.paths.artifact_path("reports", "figures", "roc_curves_preop.png"),
        artifacts.paths.artifact_path("reports", "figures", "roc_curves_preop.svg"),
        artifacts.paths.artifact_path("reports", "figures", "calibration_curves_combined.png"),
        artifacts.paths.artifact_path("reports", "figures", "consort.png"),
        artifacts.paths.artifact_path("reports", "figures", "consort.svg"),
    ]
    for path in expected_paths:
        assert path.exists(), str(path)


def test_sequence_artifact_and_legacy_export(completed_pipeline) -> None:
    artifacts = ArtifactManager(completed_pipeline)
    sequence_df = artifacts.read_pickle("datasets", "sequence", "lstm_trainable.pkl")
    assert not sequence_df.empty
    first_tensor = sequence_df["time_tensors"].iloc[0]
    assert isinstance(first_tensor, np.ndarray)
    assert first_tensor.shape[0] == completed_pipeline["sequence"]["pad_length"]
    assert int(sequence_df["seq_len"].max()) < completed_pipeline["sequence"]["pad_length"]

    exported = export_legacy_datasets(artifacts)
    assert exported
    assert (artifacts.paths.compat_aki_dir / "preop_data.csv").exists()
    assert (artifacts.paths.compat_base_dir / "tabular_combined.csv").exists()
    assert (artifacts.paths.compat_results_dir / "performance_table.md").exists()


def test_export_legacy_rejects_non_aki_outcome(loaded_synthetic_config) -> None:
    artifacts = ArtifactManager(_activate_outcome_config(loaded_synthetic_config, "macce"))

    with pytest.raises(ValueError, match="only supported for the AKI outcome"):
        export_legacy_datasets(artifacts)


def test_report_curves_separates_dca_figures_by_population(loaded_synthetic_config) -> None:
    artifacts = ArtifactManager(loaded_synthetic_config)
    prediction_rows = []
    dca_rows = []

    for population_id, model_key, treat_all in [
        ("intraop", "log_reg", 0.08),
        ("sequence_common", "lstm_only", 0.03),
    ]:
        for idx, (y_true, y_prob) in enumerate([(0, 0.1), (1, 0.8), (0, 0.2), (1, 0.9)], start=1):
            prediction_rows.append(
                {
                    "op_id": idx if population_id == "intraop" else idx + 10,
                    "dataset_regime": "intraop",
                    "population_id": population_id,
                    "repeat_id": 0,
                    "fold_id": 0,
                    "split_name": "test",
                    "model_key": model_key,
                    "target": "aki_boolean",
                    "y_true": y_true,
                    "y_prob_raw": y_prob,
                    "y_prob_calibrated": y_prob,
                    "y_pred": int(y_prob >= 0.5),
                    "threshold": 0.5,
                    "calibration_method": "identity",
                    "run_id": f"intraop:{population_id}:{model_key}",
                }
            )
        for threshold, net_benefit_model in [(0.1, 0.05), (0.2, 0.04)]:
            dca_rows.append(
                {
                    "dataset_regime": "intraop",
                    "population_id": population_id,
                    "model_key": model_key,
                    "threshold_prob": threshold,
                    "net_benefit_model": net_benefit_model,
                    "net_benefit_treat_all": treat_all,
                    "net_benefit_treat_none": 0.0,
                }
            )

    artifacts.write_dataframe(pd.DataFrame(prediction_rows), "predictions", "calibrated_predictions.parquet")
    artifacts.write_dataframe(pd.DataFrame(dca_rows), "evaluation", "dca_curves.csv")

    outputs = generate_curve_outputs(artifacts, loaded_synthetic_config)

    intraop_path = artifacts.paths.artifact_path("reports", "figures", "dca_curve_intraop_Logistic_Regression.png")
    sequence_path = artifacts.paths.artifact_path("reports", "figures", "dca_curve_intraop_LSTM_sequence_common.png")
    legacy_path = artifacts.paths.artifact_path("reports", "figures", "dca_curves_intraop.png")

    assert intraop_path in outputs
    assert sequence_path in outputs
    assert intraop_path.exists()
    assert sequence_path.exists()
    assert not legacy_path.exists()


def test_consort_outputs_branch_to_final_active_outcome_split(loaded_synthetic_config) -> None:
    config = _activate_outcome_config(loaded_synthetic_config, "macce")
    artifacts = ArtifactManager(config)
    preop_audit = pd.DataFrame(
        [
            {"step": "raw_operations", "count": 120, "note": "raw"},
            {"step": "adult_only", "count": 118, "note": "adult filter"},
            {"step": "has_opend_time", "count": 110, "note": "op end"},
            {"step": "after_prefix_exclusions", "count": 100, "note": "prefix"},
        ]
    )
    labels_audit = pd.DataFrame(
        [
            {"step": "tabular_ops_before_labels", "count": 92, "note": "tabular"},
            {"step": "has_preop_creatinine", "count": 88, "note": "preop creatinine"},
            {"step": "has_postop_creatinine_or_dialysis", "count": 80, "note": "postop creatinine"},
            {"step": "final_labeled_ops", "count": 80, "note": "final"},
        ]
    )
    artifacts.write_dataframe(preop_audit, "cohort", "preop_audit.csv")
    artifacts.write_dataframe(labels_audit, "cohort", "labels_audit.csv")
    artifacts.write_dataframe(pd.DataFrame({"op_id": range(100)}), "features", "preop", "preop_features.csv")
    artifacts.write_dataframe(pd.DataFrame({"op_id": range(92)}), "features", "intraop", "feature_engineered.csv")
    artifacts.write_dataframe(pd.DataFrame({"op_id": range(80), "macce": [0] * 64 + [1] * 16}), "cohort", "labels.csv")

    outputs = generate_consort_outputs(artifacts)

    dot_path = artifacts.paths.artifact_path("reports", "figures", "consort.dot")
    dot_text = dot_path.read_text(encoding="utf-8")

    assert dot_path in outputs
    assert 'rankdir="TB"' in dot_text
    assert 'label="Study Cohort Flow and Final 30-day MACCE Split"' in dot_text
    assert "splines=ortho" in dot_text
    assert 'ordering="out"' in dot_text
    assert "final_labeled:s -> final_negative:n [minlen=1];" in dot_text
    assert "final_labeled:s -> final_positive:n [minlen=1];" in dot_text
    assert 'analytic_preop:e -> excluded_preop:w' in dot_text
    assert "Excluded after preoperative cohort" in dot_text
    assert "30-day MACCE" in dot_text


def test_macce_grouped_holdout_smoke_pipeline(synthetic_config) -> None:
    _inject_postop_macce_events(synthetic_config)
    config = _macce_smoke_config(synthetic_config)
    artifacts = ArtifactManager(config)

    run_preop(config)
    run_intraop(config)
    run_tabular(config)
    run_labels(config)
    run_evaluate_generate(config)
    run_tune_tabular(config)
    run_train_tabular(config)
    run_calibration(config)
    run_metrics(config)
    run_delong(config)
    run_dca(config)
    run_consort(config)
    run_tables(config)

    labels_df = pd.read_csv(artifacts.paths.artifact_path("cohort", "labels.csv"))
    manifest_df = pd.read_parquet(artifacts.paths.artifact_path("datasets", "splits", "grouped_holdout_combined.parquet"))
    predictions_df = pd.read_parquet(artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet"))
    cohort_table = pd.read_csv(artifacts.paths.artifact_path("reports", "tables", "cohort_characteristics.csv"))

    assert "macce" in labels_df.columns
    assert int(labels_df["macce"].astype(int).sum()) >= 4
    assert set(manifest_df["evaluation_mode"]) == {"grouped_holdout"}
    assert set(predictions_df["target"]) == {"macce"}
    assert cohort_table["characteristic"].astype(str).str.contains("30-day MACCE", regex=False).any()


def test_cohort_characteristics_use_legacy_sex_encoding_and_deduped_departments(loaded_synthetic_config) -> None:
    artifacts = ArtifactManager(loaded_synthetic_config)
    combined_df = pd.DataFrame(
        [
            {
                "op_id": 1,
                "age": 50.0,
                "sex": False,
                "height": 160.0,
                "weight": 60.0,
                "asa": 2.0,
                "BSA": 1.6,
                "BMI": 23.4,
                "booking_case_length": 120.0,
                "num_card_events": 0.0,
                "department_GS": 1,
                "department_UR": 0,
            },
            {
                "op_id": 2,
                "age": 60.0,
                "sex": True,
                "height": 170.0,
                "weight": 70.0,
                "asa": 3.0,
                "BSA": 1.8,
                "BMI": 24.2,
                "booking_case_length": 150.0,
                "num_card_events": 1.0,
                "department_GS": 0,
                "department_UR": 1,
            },
            {
                "op_id": 3,
                "age": 55.0,
                "sex": False,
                "height": 165.0,
                "weight": 65.0,
                "asa": 1.0,
                "BSA": 1.7,
                "BMI": 23.9,
                "booking_case_length": 90.0,
                "num_card_events": 0.0,
                "department_GS": 1,
                "department_UR": 0,
            },
        ]
    )
    preop_df = combined_df[["op_id", "sex", "department_GS", "department_UR"]].copy()
    labels_df = pd.DataFrame({"op_id": [1, 2, 3], "aki_boolean": [1, 0, 0]})

    artifacts.write_dataframe(combined_df, "datasets", "tabular", "tabular_combined_unnormalized.csv")
    artifacts.write_dataframe(preop_df, "features", "preop", "preop_features.csv")
    artifacts.write_dataframe(labels_df, "cohort", "labels.csv")

    generate_table_outputs(artifacts)

    cohort_table = pd.read_csv(artifacts.paths.artifact_path("reports", "tables", "cohort_characteristics.csv"))
    cohort_markdown = artifacts.paths.artifact_path("reports", "tables", "cohort_characteristics.md").read_text(encoding="utf-8")
    total_patients_row = cohort_table.loc[cohort_table["characteristic"] == "Total patients, n"].iloc[0]
    total_operations_row = cohort_table.loc[cohort_table["characteristic"] == "Total operations, n"].iloc[0]
    female_row = cohort_table.loc[cohort_table["characteristic"] == "Female sex, n (%)"].iloc[0]
    department_rows = cohort_table[cohort_table["characteristic"].isin(["General Surgery", "Urology"])]

    assert total_patients_row["finding"] == "3"
    assert total_operations_row["finding"] == "3"
    assert female_row["finding"] == "2 (66.67%)"
    assert len(department_rows) == 2
    assert not cohort_table["characteristic"].astype(str).str.contains("preop", case=False).any()
    assert "GS" not in cohort_table["characteristic"].tolist()
    assert "UR" not in cohort_table["characteristic"].tolist()
    assert "Department, n (%)" in cohort_markdown
    assert "Department Surgery type, n (%)" not in cohort_markdown


def test_performance_tables_filter_asa_rule_and_render_column_gradients(loaded_synthetic_config) -> None:
    artifacts = ArtifactManager(loaded_synthetic_config)
    summary_df = pd.DataFrame(
        [
            {
                "dataset_regime": "preop",
                "population_id": "preop",
                "model_key": "asa_rule",
                "model_name": "ASA Rule",
                "auroc": 0.75,
                "auroc_display": "0.750",
                "auroc_ci_display": "(0.700, 0.800)",
                "auprc": 0.19,
                "auprc_display": "0.190",
                "auprc_ci_display": "(0.150, 0.230)",
                "sensitivity": 0.10,
                "sensitivity_display": "0.100",
                "sensitivity_ci_display": "(0.080, 0.120)",
                "specificity": 0.99,
                "specificity_display": "0.990",
                "specificity_ci_display": "(0.980, 1.000)",
                "precision": 0.30,
                "precision_display": "0.300",
                "precision_ci_display": "(0.250, 0.350)",
                "f_score": 0.15,
                "f_score_display": "0.150",
                "f_score_ci_display": "(0.120, 0.180)",
                "balanced_accuracy": 0.55,
                "balanced_accuracy_display": "0.550",
                "balanced_accuracy_ci_display": "(0.500, 0.600)",
            },
            {
                "dataset_regime": "preop",
                "population_id": "preop",
                "model_key": "autogluon",
                "model_name": "AutoGluon",
                "auroc": 0.93,
                "auroc_display": "0.930",
                "auroc_ci_display": "(0.920, 0.940)",
                "auprc": 0.62,
                "auprc_display": "0.620",
                "auprc_ci_display": "(0.600, 0.640)",
                "sensitivity": 0.82,
                "sensitivity_display": "0.820",
                "sensitivity_ci_display": "(0.790, 0.850)",
                "specificity": 0.89,
                "specificity_display": "0.890",
                "specificity_ci_display": "(0.870, 0.910)",
                "precision": 0.26,
                "precision_display": "0.260",
                "precision_ci_display": "(0.240, 0.280)",
                "f_score": 0.39,
                "f_score_display": "0.390",
                "f_score_ci_display": "(0.370, 0.410)",
                "balanced_accuracy": 0.85,
                "balanced_accuracy_display": "0.850",
                "balanced_accuracy_ci_display": "(0.830, 0.870)",
            },
            {
                "dataset_regime": "preop",
                "population_id": "preop",
                "model_key": "log_reg",
                "model_name": "Logistic Regression",
                "auroc": 0.91,
                "auroc_display": "0.910",
                "auroc_ci_display": "(0.900, 0.920)",
                "auprc": 0.54,
                "auprc_display": "0.540",
                "auprc_ci_display": "(0.520, 0.560)",
                "sensitivity": 0.80,
                "sensitivity_display": "0.800",
                "sensitivity_ci_display": "(0.780, 0.820)",
                "specificity": 0.88,
                "specificity_display": "0.880",
                "specificity_ci_display": "(0.860, 0.900)",
                "precision": 0.24,
                "precision_display": "0.240",
                "precision_ci_display": "(0.220, 0.260)",
                "f_score": 0.36,
                "f_score_display": "0.360",
                "f_score_ci_display": "(0.340, 0.380)",
                "balanced_accuracy": 0.84,
                "balanced_accuracy_display": "0.840",
                "balanced_accuracy_ci_display": "(0.820, 0.860)",
            },
            {
                "dataset_regime": "combined",
                "population_id": "combined",
                "model_key": "asa_rule",
                "model_name": "ASA Rule",
                "auroc": 0.75,
                "auroc_display": "0.750",
                "auroc_ci_display": "(0.700, 0.800)",
                "auprc": 0.19,
                "auprc_display": "0.190",
                "auprc_ci_display": "(0.150, 0.230)",
                "sensitivity": 0.10,
                "sensitivity_display": "0.100",
                "sensitivity_ci_display": "(0.080, 0.120)",
                "specificity": 0.99,
                "specificity_display": "0.990",
                "specificity_ci_display": "(0.980, 1.000)",
                "precision": 0.30,
                "precision_display": "0.300",
                "precision_ci_display": "(0.250, 0.350)",
                "f_score": 0.15,
                "f_score_display": "0.150",
                "f_score_ci_display": "(0.120, 0.180)",
                "balanced_accuracy": 0.55,
                "balanced_accuracy_display": "0.550",
                "balanced_accuracy_ci_display": "(0.500, 0.600)",
            },
            {
                "dataset_regime": "combined",
                "population_id": "combined",
                "model_key": "autogluon",
                "model_name": "AutoGluon",
                "auroc": 0.94,
                "auroc_display": "0.940",
                "auroc_ci_display": "(0.930, 0.950)",
                "auprc": 0.64,
                "auprc_display": "0.640",
                "auprc_ci_display": "(0.620, 0.660)",
                "sensitivity": 0.83,
                "sensitivity_display": "0.830",
                "sensitivity_ci_display": "(0.800, 0.860)",
                "specificity": 0.91,
                "specificity_display": "0.910",
                "specificity_ci_display": "(0.890, 0.930)",
                "precision": 0.28,
                "precision_display": "0.280",
                "precision_ci_display": "(0.260, 0.300)",
                "f_score": 0.42,
                "f_score_display": "0.420",
                "f_score_ci_display": "(0.400, 0.440)",
                "balanced_accuracy": 0.86,
                "balanced_accuracy_display": "0.860",
                "balanced_accuracy_ci_display": "(0.840, 0.880)",
            },
            {
                "dataset_regime": "combined",
                "population_id": "combined",
                "model_key": "hybrid",
                "model_name": "Hybrid (MLP + LSTM)",
                "auroc": 0.82,
                "auroc_display": "0.820",
                "auroc_ci_display": "(0.790, 0.850)",
                "auprc": 0.27,
                "auprc_display": "0.270",
                "auprc_ci_display": "(0.240, 0.300)",
                "sensitivity": 0.71,
                "sensitivity_display": "0.710",
                "sensitivity_ci_display": "(0.650, 0.770)",
                "specificity": 0.79,
                "specificity_display": "0.790",
                "specificity_ci_display": "(0.760, 0.820)",
                "precision": 0.18,
                "precision_display": "0.180",
                "precision_ci_display": "(0.160, 0.200)",
                "f_score": 0.27,
                "f_score_display": "0.270",
                "f_score_ci_display": "(0.240, 0.300)",
                "balanced_accuracy": 0.75,
                "balanced_accuracy_display": "0.750",
                "balanced_accuracy_ci_display": "(0.720, 0.780)",
            },
        ]
    )

    spec = _performance_table_spec(
        summary_df,
        file_stem="performance_table",
        title="Performance Metrics",
        caption="Test table",
    )
    outputs = write_table_outputs(artifacts, spec, loaded_synthetic_config)
    html_path = artifacts.paths.artifact_path("reports", "tables", "performance_table.html")

    preop_section = next(section for section in spec.sections if section.title == "Preop Data")
    combined_section = next(section for section in spec.sections if section.title == "Combined Data")

    assert preop_section.csv_df["model_key"].tolist() == ["asa_rule", "autogluon", "log_reg"]
    assert "asa_rule" not in combined_section.csv_df["model_key"].tolist()
    assert html_path in outputs
    assert 'style="background: linear-gradient(180deg, rgb(' in html_path.read_text(encoding="utf-8")


def test_grouped_holdout_performance_summary_uses_bootstrap_ci(loaded_synthetic_config) -> None:
    config = copy.deepcopy(loaded_synthetic_config)
    config["evaluation_mode"] = "grouped_holdout"
    config["evaluation"]["bootstrap_reps"] = 25

    rows = []
    for idx in range(60):
        y_true = 1 if idx % 6 == 0 else 0
        y_prob = 0.82 if y_true else 0.12 + (idx % 5) * 0.03
        rows.append(
            {
                "op_id": idx + 1,
                "dataset_regime": "combined",
                "population_id": "combined",
                "repeat_id": 0,
                "fold_id": 0,
                "split_name": "test",
                "model_key": "log_reg",
                "target": "aki_boolean",
                "y_true": y_true,
                "y_prob_raw": y_prob,
                "y_prob_calibrated": y_prob,
                "y_pred": int(y_prob >= 0.5),
                "threshold": 0.5,
                "calibration_method": "identity",
                "run_id": "combined:log_reg:0",
            }
        )

    summary_df = _performance_summary_frame(
        pd.DataFrame(rows),
        prob_col="y_prob_calibrated",
        config=config,
        use_existing_threshold=True,
    )

    row = summary_df.iloc[0]
    assert row["auroc_ci_display"] != "N/A"
    assert row["auprc_ci_display"] != "N/A"
    assert row["balanced_accuracy_ci_display"] != "N/A"
