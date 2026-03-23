from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

import inspire_aki.cli as cli_module
from inspire_aki.cli import app
from inspire_aki.config import load_config
from inspire_aki.datasets.tabular import build_tabular_datasets
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.predictions import PREDICTION_PRIMARY_KEY
from inspire_aki.pipelines.preprocess import run_intraop, run_labels, run_preop, run_sequence, run_tabular, run_timeseries
from inspire_aki.pipelines.report import run_manuscript
from inspire_aki.pipelines.train import run_train_sequence, run_train_tabular
from inspire_aki.pipelines.tune import run_tune_sequence, run_tune_tabular
from inspire_aki.reporting.manuscript import generate_manuscript_outputs


def _prepare_training_inputs(config_path: Path, *, include_sequence: bool = False) -> dict:
    config = load_config(config_path)
    run_preop(config)
    run_intraop(config)
    run_tabular(config)
    run_labels(config)
    if include_sequence:
        run_timeseries(config)
        run_sequence(config)
    return config


def test_train_tabular_is_idempotent(synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config)
    artifacts = ArtifactManager(config)

    run_train_tabular(config)
    first_partition = pd.read_parquet(artifacts.paths.artifact_path("predictions", "raw", "tabular.parquet"))
    first_combined = pd.read_parquet(artifacts.paths.artifact_path("predictions", "raw_predictions.parquet"))

    run_train_tabular(config)
    second_partition = pd.read_parquet(artifacts.paths.artifact_path("predictions", "raw", "tabular.parquet"))
    second_combined = pd.read_parquet(artifacts.paths.artifact_path("predictions", "raw_predictions.parquet"))

    pd.testing.assert_frame_equal(first_partition, second_partition)
    pd.testing.assert_frame_equal(first_combined, second_combined)
    assert not second_combined.duplicated(PREDICTION_PRIMARY_KEY).any()


def test_train_sequence_is_idempotent_and_preserves_tabular(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config, include_sequence=True)
    config["models"]["sequence_enabled"] = ["lstm_only"]
    artifacts = ArtifactManager(config)

    def fake_fit_sequence_model(**_kwargs):
        return object()

    def fake_predict_sequence_bundle(_bundle, test_df):
        y_prob = np.linspace(0.2, 0.8, len(test_df), dtype=float)
        return (y_prob >= 0.5).astype(int), y_prob

    monkeypatch.setattr("inspire_aki.pipelines.train.fit_sequence_model", fake_fit_sequence_model)
    monkeypatch.setattr("inspire_aki.pipelines.train.predict_sequence_bundle", fake_predict_sequence_bundle)

    run_train_tabular(config)
    tabular_partition = pd.read_parquet(artifacts.paths.artifact_path("predictions", "raw", "tabular.parquet"))

    run_train_sequence(config)
    first_combined = pd.read_parquet(artifacts.paths.artifact_path("predictions", "raw_predictions.parquet"))

    pd.testing.assert_frame_equal(
        tabular_partition,
        pd.read_parquet(artifacts.paths.artifact_path("predictions", "raw", "tabular.parquet")),
    )
    assert not first_combined.duplicated(PREDICTION_PRIMARY_KEY).any()

    run_train_sequence(config)
    second_combined = pd.read_parquet(artifacts.paths.artifact_path("predictions", "raw_predictions.parquet"))

    pd.testing.assert_frame_equal(first_combined, second_combined)
    assert not second_combined.duplicated(PREDICTION_PRIMARY_KEY).any()


def test_train_tabular_preserves_existing_sequence_partition(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config, include_sequence=True)
    config["models"]["sequence_enabled"] = ["lstm_only"]
    artifacts = ArtifactManager(config)

    def fake_fit_sequence_model(**_kwargs):
        return object()

    def fake_predict_sequence_bundle(_bundle, test_df):
        y_prob = np.linspace(0.1, 0.9, len(test_df), dtype=float)
        return (y_prob >= 0.5).astype(int), y_prob

    monkeypatch.setattr("inspire_aki.pipelines.train.fit_sequence_model", fake_fit_sequence_model)
    monkeypatch.setattr("inspire_aki.pipelines.train.predict_sequence_bundle", fake_predict_sequence_bundle)

    run_train_sequence(config)
    sequence_partition = pd.read_parquet(artifacts.paths.artifact_path("predictions", "raw", "sequence.parquet"))

    run_train_tabular(config)
    combined = pd.read_parquet(artifacts.paths.artifact_path("predictions", "raw_predictions.parquet"))

    pd.testing.assert_frame_equal(
        sequence_partition,
        pd.read_parquet(artifacts.paths.artifact_path("predictions", "raw", "sequence.parquet")),
    )
    assert len(combined[combined["model_key"] == "lstm_only"]) == len(sequence_partition)
    assert not combined.duplicated(PREDICTION_PRIMARY_KEY).any()


def test_generate_manuscript_outputs_runs_sections_in_order(monkeypatch, loaded_synthetic_config) -> None:
    artifacts = ArtifactManager(loaded_synthetic_config)
    loaded_synthetic_config["reports"]["manuscript_sections"] = ["consort", "tables", "curves", "shap"]
    calls: list[str] = []

    def _section(name: str):
        def _impl(*_args, **_kwargs):
            calls.append(name)
            return [artifacts.resolve("reports", "unit", f"{name}.txt")]

        return _impl

    monkeypatch.setattr("inspire_aki.reporting.manuscript.generate_consort_outputs", _section("consort"))
    monkeypatch.setattr("inspire_aki.reporting.manuscript.generate_table_outputs", _section("tables"))
    monkeypatch.setattr("inspire_aki.reporting.manuscript.generate_curve_outputs", _section("curves"))
    monkeypatch.setattr("inspire_aki.reporting.manuscript.generate_shap_outputs", _section("shap"))

    outputs = generate_manuscript_outputs(artifacts, loaded_synthetic_config)

    assert calls == ["consort", "tables", "curves", "shap"]
    assert [path.name for path in outputs] == ["consort.txt", "tables.txt", "curves.txt", "shap.txt"]


def test_report_manuscript_fails_when_required_shap_bundle_is_missing(loaded_synthetic_config) -> None:
    loaded_synthetic_config["reports"]["manuscript_sections"] = ["shap"]
    loaded_synthetic_config["reports"]["shap_jobs"] = [{"dataset_regime": "combined", "model_key": "xgb"}]

    with pytest.raises(FileNotFoundError, match="bundle.joblib"):
        run_manuscript(loaded_synthetic_config)


def test_run_all_relies_on_report_manuscript_instead_of_run_shap(monkeypatch, synthetic_config: Path) -> None:
    runner = CliRunner()
    calls: list[str] = []

    def _stub(name: str):
        def _impl(_cfg):
            calls.append(name)
            return {}

        return _impl

    monkeypatch.setattr(cli_module, "_cfg", lambda _path: {})
    monkeypatch.setattr(cli_module, "run_preop", _stub("run_preop"))
    monkeypatch.setattr(cli_module, "run_intraop", _stub("run_intraop"))
    monkeypatch.setattr(cli_module, "run_tabular", _stub("run_tabular"))
    monkeypatch.setattr(cli_module, "run_labels", _stub("run_labels"))
    monkeypatch.setattr(cli_module, "run_timeseries", _stub("run_timeseries"))
    monkeypatch.setattr(cli_module, "run_sequence", _stub("run_sequence"))
    monkeypatch.setattr(cli_module, "run_tune_tabular", _stub("run_tune_tabular"))
    monkeypatch.setattr(cli_module, "run_tune_sequence", _stub("run_tune_sequence"))
    monkeypatch.setattr(cli_module, "run_train_tabular", _stub("run_train_tabular"))
    monkeypatch.setattr(cli_module, "run_train_sequence", _stub("run_train_sequence"))
    monkeypatch.setattr(cli_module, "run_calibration", _stub("run_calibration"))
    monkeypatch.setattr(cli_module, "run_metrics", _stub("run_metrics"))
    monkeypatch.setattr(cli_module, "run_delong", _stub("run_delong"))
    monkeypatch.setattr(cli_module, "run_dca", _stub("run_dca"))
    monkeypatch.setattr(cli_module, "run_manuscript", _stub("run_manuscript"))
    monkeypatch.setattr(cli_module, "run_shap", lambda _cfg: (_ for _ in ()).throw(AssertionError("run_shap should not be called by run all")))

    result = runner.invoke(app, ["run", "all", "--config", str(synthetic_config)])

    assert result.exit_code == 0, result.stdout
    assert "run_manuscript" in calls


def test_tune_tabular_uses_pipeline_written_hpo_manifests(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config)
    artifacts = ArtifactManager(config)
    captured: dict[str, pd.DataFrame] = {}

    def fake_tune_tabular_dataset(dataset_df, dataset_regime, manifest, _config):
        captured[dataset_regime] = manifest.copy()
        return {}, pd.DataFrame()

    monkeypatch.setattr("inspire_aki.pipelines.tune.tune_tabular_dataset", fake_tune_tabular_dataset)
    run_tune_tabular(config)

    assert set(captured) == {"preop", "intraop", "combined"}
    for dataset_regime, manifest in captured.items():
        assert set(manifest["split_name"]) == {"train", "val", "holdout"}
        assert artifacts.paths.artifact_path("datasets", "splits", f"hpo_{dataset_regime}.parquet").exists()


def test_tune_sequence_uses_pipeline_written_hpo_manifest(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config, include_sequence=True)
    artifacts = ArtifactManager(config)
    captured: dict[str, pd.DataFrame] = {}

    def fake_tune_sequence_dataset(sequence_df, manifest, _config):
        captured["sequence"] = manifest.copy()
        return {}, pd.DataFrame()

    monkeypatch.setattr("inspire_aki.pipelines.tune.tune_sequence_dataset", fake_tune_sequence_dataset)
    run_tune_sequence(config)

    assert set(captured["sequence"]["split_name"]) == {"train", "val", "holdout"}
    assert artifacts.paths.artifact_path("datasets", "splits", "hpo_sequence.parquet").exists()


def test_build_tabular_datasets_requires_op_id(loaded_synthetic_config) -> None:
    preop_missing = pd.DataFrame({"subject_id": [1], "age": [42]})
    intraop_ok = pd.DataFrame({"op_id": [1], "feature_a": [0.1]})
    with pytest.raises(ValueError, match="preop_df"):
        build_tabular_datasets(preop_missing, intraop_ok, loaded_synthetic_config)

    preop_ok = pd.DataFrame({"op_id": [1], "age": [42]})
    intraop_missing = pd.DataFrame({"subject_id": [1], "feature_a": [0.1]})
    with pytest.raises(ValueError, match="intraop_df"):
        build_tabular_datasets(preop_ok, intraop_missing, loaded_synthetic_config)
