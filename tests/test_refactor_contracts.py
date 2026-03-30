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
from inspire_aki.models.hpo import _has_completed_trials, _safe_study_best_value, tune_sequence_dataset, tune_tabular_dataset
from inspire_aki.models.tabular import tabular_execution_policy
from inspire_aki.models.weighting import safe_balanced_accuracy


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


def test_train_tabular_requires_grouped_manifest_when_configured(synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config)
    config["evaluation_mode"] = "grouped_nested_cv"

    with pytest.raises(FileNotFoundError, match="evaluate generate"):
        run_train_tabular(config)


def test_train_tabular_uses_precomputed_grouped_manifest_when_configured(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config)
    artifacts = ArtifactManager(config)
    config["evaluation_mode"] = "grouped_nested_cv"
    target = config["models"]["target"]

    for dataset_regime in ["preop", "intraop", "combined"]:
        dataset_df = pd.read_csv(artifacts.paths.artifact_path("datasets", "tabular", f"tabular_{dataset_regime}_labeled.csv"))
        train_ids = dataset_df["op_id"].iloc[::2].tolist()
        test_ids = dataset_df["op_id"].iloc[1::2].tolist()
        manifest = pd.DataFrame(
            [
                {"op_id": op_id, "repeat_id": 0, "fold_id": 0, "split_name": "train"}
                for op_id in train_ids
            ]
            + [
                {"op_id": op_id, "repeat_id": 0, "fold_id": 0, "split_name": "test"}
                for op_id in test_ids
            ]
        )
        artifacts.write_dataframe(manifest, "datasets", "splits", f"grouped_nested_cv_{dataset_regime}.parquet")

    monkeypatch.setattr(
        "inspire_aki.pipelines.train.build_bootstrap_split_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("bootstrap manifest builder should not run")),
    )

    run_train_tabular(config)

    assert artifacts.paths.artifact_path("predictions", "raw", "tabular.parquet").exists()
    assert not artifacts.paths.artifact_path("datasets", "splits", "bootstrap_preop.parquet").exists()
    manifest_payload = artifacts.read_json("manifests", "train_tabular.json")
    assert artifacts.relative(artifacts.paths.artifact_path("datasets", "splits", "grouped_nested_cv_preop.parquet")) in manifest_payload["inputs"]


def test_train_tabular_uses_repeat_executor_for_svm(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config)
    config["models"]["tabular_enabled"] = ["svm"]
    submissions: list[int] = []
    captured_max_workers: list[int] = []

    class FakeFuture:
        def __init__(self, payload):
            self._payload = payload

        def result(self):
            return self._payload

    class FakeExecutor:
        def __init__(self, max_workers: int):
            captured_max_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, task, cfg):
            submissions.append(task.repeat_id)
            return FakeFuture(fn(task, cfg))

    monkeypatch.setattr("inspire_aki.pipelines.train.ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr("inspire_aki.pipelines.train.as_completed", lambda futures: list(futures))

    run_train_tabular(config)

    assert captured_max_workers == [2, 2, 2]
    assert submissions == [0, 1, 0, 1, 0, 1]


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
    monkeypatch.setattr(cli_module, "run_reclassification", _stub("run_reclassification"))
    monkeypatch.setattr(cli_module, "run_manuscript", _stub("run_manuscript"))
    monkeypatch.setattr(cli_module, "run_shap", lambda _cfg: (_ for _ in ()).throw(AssertionError("run_shap should not be called by run all")))

    result = runner.invoke(app, ["run", "all", "--config", str(synthetic_config)])

    assert result.exit_code == 0, result.stdout
    assert "run_manuscript" in calls


def test_tune_tabular_uses_pipeline_written_hpo_manifests(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config)
    config["models"]["tabular_hpo_enabled"] = ["log_reg"]
    artifacts = ArtifactManager(config)
    captured: dict[str, pd.DataFrame] = {}

    def fake_tune_tabular_dataset(dataset_df, dataset_regime, manifest, _config, **_kwargs):
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


def test_tune_tabular_resumes_completed_per_study_outputs(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config)
    config["models"]["tabular_hpo_enabled"] = ["log_reg"]
    artifacts = ArtifactManager(config)

    def fake_tune_tabular_dataset(dataset_df, dataset_regime, manifest, _config, **_kwargs):
        return {"log_reg": {"C": 1.0}}, pd.DataFrame(
            [
                {
                    "dataset_regime": dataset_regime,
                    "model_key": "log_reg",
                    "trial_number": 0,
                    "value": 0.8,
                    "params": {"C": 1.0},
                    "state": "COMPLETE",
                }
            ]
        )

    monkeypatch.setattr("inspire_aki.pipelines.tune.tune_tabular_dataset", fake_tune_tabular_dataset)
    run_tune_tabular(config)

    study_dir = artifacts.paths.artifact_path("tuning", "tabular_studies")
    assert (study_dir / "preop__log_reg_best_params.json").exists()
    assert (study_dir / "intraop__log_reg_best_params.json").exists()
    assert (study_dir / "combined__log_reg_best_params.json").exists()

    monkeypatch.setattr(
        "inspire_aki.pipelines.tune.tune_tabular_dataset",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume should skip completed studies")),
    )

    outputs = run_tune_tabular(config)

    assert set(outputs) == {"best_params", "trials"}


def test_tabular_execution_policy_targets_low_cpu_models(monkeypatch) -> None:
    monkeypatch.setenv("INSPIRE_AKI_EXECUTION_POLICY", "optimized_low_cpu")

    assert tabular_execution_policy("svm").hpo_parallel_by_regime is True
    assert tabular_execution_policy("svm").train_parallel_by_repeat is True
    assert tabular_execution_policy("svm").train_tol == pytest.approx(0.01)
    assert tabular_execution_policy("log_reg").hpo_thread_cap == 4
    assert tabular_execution_policy("log_reg").train_thread_cap == 4


def test_tune_tabular_uses_regime_executor_for_svm(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config)
    config["models"]["tabular_hpo_enabled"] = ["svm"]
    submissions: list[str] = []
    captured_max_workers: list[int] = []

    class FakeFuture:
        def __init__(self, payload):
            self._payload = payload

        def result(self):
            return self._payload

    class FakeExecutor:
        def __init__(self, max_workers: int):
            captured_max_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, _fn, spec, _config):
            submissions.append(spec.dataset_regime)
            return FakeFuture(
                {
                    "dataset_regime": spec.dataset_regime,
                    "model_key": spec.model_key,
                    "best_params": {"C": 0.5},
                    "trials_df": pd.DataFrame(
                        [
                            {
                                "dataset_regime": spec.dataset_regime,
                                "model_key": spec.model_key,
                                "trial_number": 0,
                                "value": 0.75,
                                "params": {"C": 0.5},
                                "state": "COMPLETE",
                            }
                        ]
                    ),
                    "best_params_path": str(Path(spec.dataset_path).with_name(f"{spec.dataset_regime}_{spec.model_key}.json")),
                    "trials_path": str(Path(spec.dataset_path).with_name(f"{spec.dataset_regime}_{spec.model_key}.parquet")),
                    "manifest_path": str(Path(spec.manifest_path)),
                    "wall_time_seconds": 0.1,
                }
            )

    monkeypatch.setattr("inspire_aki.pipelines.tune.ProcessPoolExecutor", FakeExecutor)
    monkeypatch.setattr("inspire_aki.pipelines.tune.as_completed", lambda futures: list(futures))

    run_tune_tabular(config)

    assert captured_max_workers == [3]
    assert submissions == ["preop", "intraop", "combined"]


def test_tune_sequence_uses_pipeline_written_hpo_manifest(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config, include_sequence=True)
    artifacts = ArtifactManager(config)
    captured: dict[str, pd.DataFrame] = {}

    def fake_tune_sequence_dataset(sequence_df, manifest, _config, **_kwargs):
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

        def optimize(self, _objective, *, n_trials: int, show_progress_bar: bool, callbacks=None) -> None:
            captured_trials.append(n_trials)
            assert show_progress_bar is False
            assert callbacks is None or len(callbacks) == 1

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


def test_safe_study_best_value_returns_none_without_completed_trials() -> None:
    class FakeStudy:
        def __init__(self) -> None:
            self.trials = [types.SimpleNamespace(state=types.SimpleNamespace(name="PRUNED"))]

        @property
        def best_value(self):
            raise ValueError("No trials are completed yet.")

    assert _safe_study_best_value(FakeStudy()) is None


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


def test_tabular_hpo_progress_callback_handles_pruned_first_trial(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config)
    config["models"]["tabular_hpo_enabled"] = ["svm"]
    config["models"]["hpo"] = {"n_trials": 2, "tabular_mlp_epochs": 1, "sequence_epochs": 1, "sequence_patience": 1}

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
    progress_events: list[dict[str, object]] = []

    class _FakeLogging:
        WARNING = "WARNING"

        @staticmethod
        def set_verbosity(_value) -> None:
            return None

    class FakeStudy:
        def __init__(self) -> None:
            self.best_params = {"C": 1.0}
            self._best_value: float | None = None
            self.trials: list[types.SimpleNamespace] = []

        @property
        def best_value(self):
            if self._best_value is None:
                raise ValueError("No trials are completed yet.")
            return self._best_value

        def optimize(self, _objective, *, n_trials: int, show_progress_bar: bool, callbacks=None) -> None:
            assert n_trials == 2
            assert show_progress_bar is False
            pruned = types.SimpleNamespace(number=0, value=None, params={"C": 0.5}, state=types.SimpleNamespace(name="PRUNED"))
            self.trials.append(pruned)
            for callback in callbacks or []:
                callback(self, pruned)
            complete = types.SimpleNamespace(number=1, value=0.8, params={"C": 1.0}, state=types.SimpleNamespace(name="COMPLETE"))
            self._best_value = 0.8
            self.trials.append(complete)
            for callback in callbacks or []:
                callback(self, complete)

    monkeypatch.setitem(sys.modules, "optuna", types.SimpleNamespace(logging=_FakeLogging(), create_study=lambda **_kwargs: FakeStudy()))

    results, trials_df = tune_tabular_dataset(
        dataset_df,
        "preop",
        manifest,
        config,
        progress_callback=lambda **payload: progress_events.append(payload),
    )

    assert results == {"svm": {"C": 1.0}}
    assert progress_events[0]["state"] == "PRUNED"
    assert progress_events[0]["best_value"] is None
    assert progress_events[1]["state"] == "COMPLETE"
    assert progress_events[1]["best_value"] == pytest.approx(0.8)
    assert set(trials_df["state"]) == {"PRUNED", "COMPLETE"}


def test_tabular_hpo_log_reg_uses_balanced_accuracy_and_balance_weights(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_training_inputs(synthetic_config)
    config["models"]["tabular_hpo_enabled"] = ["log_reg"]
    config["models"]["hpo"] = {"n_trials": 1, "tabular_mlp_epochs": 5, "sequence_epochs": 2, "sequence_patience": 5}

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
    val_op_ids = manifest.loc[manifest["split_name"] == "val", "op_id"]
    val_df = dataset_df[dataset_df["op_id"].isin(val_op_ids)]
    expected_value = safe_balanced_accuracy(val_df[config["models"]["target"]].to_numpy(), np.zeros(len(val_df), dtype=int))
    captured: dict[str, object] = {}

    class FakeTrial:
        def __init__(self) -> None:
            self.number = 0
            self.params: dict[str, float] = {}
            self.value = None
            self.state = types.SimpleNamespace(name="COMPLETE")

        def suggest_float(self, name, _low, _high, log=False):  # noqa: ARG002
            self.params[name] = 1.0
            return 1.0

    class FakeStudy:
        def __init__(self) -> None:
            self.best_params: dict[str, float] = {}
            self.best_value: float | None = None
            self.trials: list[FakeTrial] = []

        def optimize(self, objective, *, n_trials: int, show_progress_bar: bool, callbacks=None) -> None:
            assert n_trials == 1
            assert show_progress_bar is False
            trial = FakeTrial()
            trial.value = objective(trial)
            self.best_params = dict(trial.params)
            self.best_value = trial.value
            self.trials = [trial]
            if callbacks:
                for callback in callbacks:
                    callback(self, trial)

    class _FakeLogging:
        WARNING = "WARNING"

        @staticmethod
        def set_verbosity(_value) -> None:
            return None

    class FakeLogisticRegression:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def fit(self, x, y, sample_weight=None):
            captured["fit_rows"] = len(x)
            captured["sample_weight"] = np.asarray(sample_weight)
            captured["y"] = np.asarray(y)
            return self

        def predict(self, x):
            return np.zeros(len(x), dtype=int)

    monkeypatch.setitem(sys.modules, "optuna", types.SimpleNamespace(logging=_FakeLogging(), create_study=lambda **_kwargs: FakeStudy()))
    monkeypatch.setattr("sklearn.linear_model.LogisticRegression", FakeLogisticRegression)

    results, trials_df = tune_tabular_dataset(dataset_df, "preop", manifest, config)

    assert results == {"log_reg": {"C": 1.0}}
    assert trials_df["value"].tolist() == pytest.approx([expected_value])
    assert captured["sample_weight"].shape[0] == captured["fit_rows"]
    assert captured["sample_weight"].max() > captured["sample_weight"].min()
    assert "class_weight" not in captured["init_kwargs"]


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
