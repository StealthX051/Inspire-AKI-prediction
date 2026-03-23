from __future__ import annotations

import numpy as np

from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.compat import export_legacy_datasets


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
