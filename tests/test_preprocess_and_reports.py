from __future__ import annotations

import numpy as np
import pandas as pd

from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.compat import export_legacy_datasets
from inspire_aki.reporting.curves import generate_curve_outputs


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
        artifacts.paths.artifact_path("reports", "tables", "performance_table.csv"),
        artifacts.paths.artifact_path("reports", "tables", "consort_audit.md"),
        artifacts.paths.artifact_path("reports", "figures", "roc_curves_preop.png"),
        artifacts.paths.artifact_path("reports", "figures", "calibration_curves_combined.png"),
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

    intraop_path = artifacts.paths.artifact_path("reports", "figures", "dca_curves_intraop_intraop.png")
    sequence_path = artifacts.paths.artifact_path("reports", "figures", "dca_curves_intraop_sequence_common.png")
    legacy_path = artifacts.paths.artifact_path("reports", "figures", "dca_curves_intraop.png")

    assert intraop_path in outputs
    assert sequence_path in outputs
    assert intraop_path.exists()
    assert sequence_path.exists()
    assert not legacy_path.exists()
