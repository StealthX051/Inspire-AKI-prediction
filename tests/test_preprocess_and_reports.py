from __future__ import annotations

import numpy as np
import pandas as pd

from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.compat import export_legacy_datasets
from inspire_aki.reporting.consort import generate_consort_outputs
from inspire_aki.reporting.curves import generate_curve_outputs
from inspire_aki.reporting.tables import _performance_table_spec
from inspire_aki.reporting.rendering import write_table_outputs


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


def test_consort_outputs_branch_to_final_aki_split(loaded_synthetic_config) -> None:
    artifacts = ArtifactManager(loaded_synthetic_config)
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
    artifacts.write_dataframe(pd.DataFrame({"op_id": range(80), "aki_boolean": [0] * 64 + [1] * 16}), "cohort", "aki_labels.csv")

    outputs = generate_consort_outputs(artifacts)

    dot_path = artifacts.paths.artifact_path("reports", "figures", "consort.dot")
    dot_text = dot_path.read_text(encoding="utf-8")

    assert dot_path in outputs
    assert 'rankdir="TB"' in dot_text
    assert 'label="Study Cohort Flow and Final Postoperative AKI Split"' in dot_text
    assert "splines=ortho" in dot_text
    assert 'ordering="out"' in dot_text
    assert "final_labeled:s -> aki_negative:n [minlen=1];" in dot_text
    assert "final_labeled:s -> aki_positive:n [minlen=1];" in dot_text
    assert 'analytic_preop:e -> excluded_preop:w' in dot_text
    assert "Excluded after preoperative cohort" in dot_text


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
