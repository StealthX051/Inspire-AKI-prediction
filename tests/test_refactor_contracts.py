from __future__ import annotations

from pathlib import Path
import sys
import types

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
from inspire_aki.datasets.splits import build_hpo_split_manifest
from inspire_aki.models.hpo import _has_completed_trials, tune_sequence_dataset, tune_tabular_dataset


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
        return {"log_reg": {"C": 1.0}}, pd.DataFrame(
            [
                {
                    "dataset_regime": dataset_regime,
                    "model_key": "log_reg",
                    "trial_number": 0,
                    "value": 0.9,
                    "params": {"C": 1.0},
                    "state": "COMPLETE",
                }
            ]
        )

    monkeypatch.setattr("inspire_aki.pipelines.tune.tune_tabular_dataset", fake_tune_tabular_dataset)
    outputs = run_tune_tabular(config)

    assert set(captured) == {"preop", "intraop", "combined"}
    assert set(outputs) == {"best_params", "trials"}
    best_params_output = artifacts.relative(artifacts.paths.artifact_path("tuning", "tabular_best_params.json"))
    trials_output = artifacts.relative(artifacts.paths.artifact_path("tuning", "tabular_trials.parquet"))
    for dataset_regime, manifest in captured.items():
        assert set(manifest["split_name"]) == {"train", "val", "holdout"}
        split_output = artifacts.paths.artifact_path("datasets", "splits", f"hpo_{dataset_regime}.parquet")
        assert split_output.exists()
        manifest_payload = artifacts.read_json("manifests", f"tune_tabular_{dataset_regime}.json")
        assert artifacts.relative(split_output) in manifest_payload["outputs"]
        assert best_params_output in manifest_payload["outputs"]
        assert trials_output in manifest_payload["outputs"]

    aggregate_manifest = artifacts.read_json("manifests", "tune_tabular.json")
    assert best_params_output in aggregate_manifest["outputs"]
    assert trials_output in aggregate_manifest["outputs"]
    assert len([output for output in aggregate_manifest["outputs"] if output.endswith(".parquet")]) == 4


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


def _fake_optuna_module(captured_trials: list[int], *, trial_state: object = "1") -> types.SimpleNamespace:
    class _FakeLogging:
        WARNING = "WARNING"

        @staticmethod
        def set_verbosity(_value) -> None:
            return None

    class _FakeStudy:
        def __init__(self) -> None:
            self.best_params: dict[str, object] = {}
            self.trials = [types.SimpleNamespace(number=0, value=0.5, params={}, state=trial_state)]

        def optimize(self, _objective, *, n_trials: int, show_progress_bar: bool) -> None:
            captured_trials.append(n_trials)
            assert show_progress_bar is False

    return types.SimpleNamespace(
        logging=_FakeLogging(),
        create_study=lambda **_kwargs: _FakeStudy(),
    )


def test_has_completed_trials_accepts_optuna4_numeric_state() -> None:
    study = types.SimpleNamespace(
        trials=[
            types.SimpleNamespace(state="0"),
            types.SimpleNamespace(state="1"),
        ]
    )

    assert _has_completed_trials(study) is True


def test_has_completed_trials_accepts_enum_like_state_name() -> None:
    complete_state = types.SimpleNamespace(name="COMPLETE")
    study = types.SimpleNamespace(trials=[types.SimpleNamespace(state=complete_state)])

    assert _has_completed_trials(study) is True


def test_tabular_hpo_uses_configured_trial_count(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config)
    config["models"]["tabular_hpo_enabled"] = ["log_reg", "xgb", "rf", "svm", "mlp", "knn"]
    config["models"]["hpo"] = {"n_trials": 3, "tabular_mlp_epochs": 5, "sequence_epochs": 2, "sequence_patience": 5}

    artifacts = ArtifactManager(config)
    dataset_df = pd.read_csv(artifacts.paths.artifact_path("datasets", "tabular", "tabular_preop_labeled.csv"))
    manifest = build_hpo_split_manifest(
        dataset_df,
        target=config["models"]["target"],
        dataset_regime="preop",
        population_id="preop",
        random_state=config["splits"]["random_state"],
        holdout_fraction=config["splits"]["holdout_fraction"],
        validation_fraction_within_train=config["splits"]["hpo_validation_fraction_within_train"],
    )

    captured_trials: list[int] = []
    monkeypatch.setitem(sys.modules, "optuna", _fake_optuna_module(captured_trials))
    monkeypatch.setitem(sys.modules, "xgboost", types.SimpleNamespace(XGBClassifier=object))

    results, trials_df = tune_tabular_dataset(dataset_df, "preop", manifest, config)

    assert set(results) == {"log_reg", "xgb", "rf", "svm", "mlp", "knn"}
    assert trials_df["model_key"].tolist() == ["log_reg", "xgb", "rf", "svm", "mlp", "knn"]
    assert set(trials_df["state"]) == {"COMPLETE"}
    assert captured_trials == [3, 3, 3, 3, 3, 3]


def test_sequence_hpo_uses_configured_trial_count(monkeypatch, synthetic_config: Path) -> None:
    pytest.importorskip("torch")

    config = _prepare_training_inputs(synthetic_config, include_sequence=True)
    config["models"]["sequence_hpo_enabled"] = ["lstm_only", "hybrid"]
    config["models"]["hpo"] = {"n_trials": 2, "tabular_mlp_epochs": 5, "sequence_epochs": 2, "sequence_patience": 5}

    artifacts = ArtifactManager(config)
    sequence_df = artifacts.read_pickle("datasets", "sequence", "lstm_trainable.pkl")
    manifest = build_hpo_split_manifest(
        sequence_df,
        target=config["models"]["target"],
        dataset_regime="sequence",
        population_id="sequence_common",
        random_state=config["splits"]["random_state"],
        holdout_fraction=config["splits"]["holdout_fraction"],
        validation_fraction_within_train=config["splits"]["hpo_validation_fraction_within_train"],
    )

    captured_trials: list[int] = []
    monkeypatch.setitem(sys.modules, "optuna", _fake_optuna_module(captured_trials))

    results, trials_df = tune_sequence_dataset(sequence_df, manifest, config)

    assert set(results) == {"lstm_only", "hybrid"}
    assert trials_df["model_key"].tolist() == ["lstm_only", "hybrid"]
    assert set(trials_df["state"]) == {"COMPLETE"}
    assert captured_trials == [2, 2]


def test_build_tabular_datasets_requires_op_id(loaded_synthetic_config) -> None:
    preop_missing = pd.DataFrame({"subject_id": [1], "age": [42]})
    intraop_ok = pd.DataFrame({"op_id": [1], "feature_a": [0.1]})
    with pytest.raises(ValueError, match="preop_df"):
        build_tabular_datasets(preop_missing, intraop_ok, loaded_synthetic_config)

    preop_ok = pd.DataFrame({"op_id": [1], "age": [42]})
    intraop_missing = pd.DataFrame({"subject_id": [1], "feature_a": [0.1]})
    with pytest.raises(ValueError, match="intraop_df"):
        build_tabular_datasets(preop_ok, intraop_missing, loaded_synthetic_config)
