from __future__ import annotations

import sys
import types

import pandas as pd
import pytest

import inspire_aki.models.sequence as sequence_module
from inspire_aki.config import load_config
from inspire_aki.datasets.splits import build_hpo_split_manifest
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.models.hpo import tune_sequence_dataset
from inspire_aki.models.sequence import _enable_sequence_cuda_benchmark, _sequence_loader_kwargs
from inspire_aki.pipelines.preprocess import run_intraop, run_labels, run_preop, run_sequence, run_tabular, run_timeseries
from inspire_aki.runtime import SystemResources, build_stage_runtime_plan


def _prepare_sequence_inputs(config_path) -> dict:
    config = load_config(config_path)
    run_preop(config)
    run_intraop(config)
    run_tabular(config)
    run_labels(config)
    run_timeseries(config)
    run_sequence(config)
    return config


def test_sequence_runtime_plan_defaults_to_zero_loader_workers(monkeypatch, loaded_synthetic_config) -> None:
    resources = SystemResources(
        cpu_count=32,
        total_ram_gb=115,
        available_ram_gb=99,
        gpu_available=True,
        gpu_name="NVIDIA A100-SXM4-40GB",
        gpu_total_memory_gb=40,
        gpu_free_memory_gb=38,
    )
    monkeypatch.setattr("inspire_aki.runtime.detect_system_resources", lambda: resources)

    tune_plan = build_stage_runtime_plan(loaded_synthetic_config, "tune_sequence")
    train_plan = build_stage_runtime_plan(loaded_synthetic_config, "train_sequence")

    assert tune_plan.dataloader_workers == 0
    assert train_plan.dataloader_workers == 0


def test_sequence_loader_kwargs_enable_persistent_workers_only_when_needed() -> None:
    zero_worker = _sequence_loader_kwargs(loader_workers=0, use_gpu=True, shuffle=False)
    assert zero_worker["num_workers"] == 0
    assert zero_worker["pin_memory"] is True
    assert "persistent_workers" not in zero_worker

    multi_worker = _sequence_loader_kwargs(loader_workers=4, use_gpu=False, shuffle=True)
    assert multi_worker["num_workers"] == 4
    assert multi_worker["persistent_workers"] is True
    assert multi_worker["prefetch_factor"] == 2


def test_sequence_hpo_uses_configured_batch_size(monkeypatch, synthetic_config) -> None:
    torch = pytest.importorskip("torch")

    config = _prepare_sequence_inputs(synthetic_config)
    config["models"]["sequence_hpo_enabled"] = ["lstm_only"]
    config["models"]["hpo"] = {
        "n_trials": 1,
        "tabular_mlp_epochs": 1,
        "sequence_epochs": 1,
        "sequence_patience": 1,
        "sequence_batch_size": 123,
    }
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

    captured_batch_sizes: list[int] = []
    original_loader = torch.utils.data.DataLoader

    def capture_loader(dataset, batch_size, **kwargs):
        captured_batch_sizes.append(int(batch_size))
        return types.SimpleNamespace(dataset=dataset, batch_size=batch_size, kwargs=kwargs)

    class _FakeLogging:
        WARNING = "WARNING"

        @staticmethod
        def set_verbosity(_value) -> None:
            return None

    class _FakeStudy:
        def __init__(self) -> None:
            self.best_params = {"lr": 0.001, "dropout_rate": 0.2, "lstm_hidden_size": 8, "lstm_num_layers": 1}
            self.best_value = 0.5
            self.trials = [types.SimpleNamespace(number=0, value=0.5, params={}, state="1")]

        def optimize(self, _objective, *, n_trials: int, show_progress_bar: bool, callbacks=None) -> None:
            assert n_trials == 1
            assert show_progress_bar is False
            assert callbacks is None or len(callbacks) == 1

    monkeypatch.setattr(torch.utils.data, "DataLoader", capture_loader)
    monkeypatch.setitem(
        sys.modules,
        "optuna",
        types.SimpleNamespace(logging=_FakeLogging(), create_study=lambda **_kwargs: _FakeStudy()),
    )

    results, trials_df = tune_sequence_dataset(sequence_df, manifest, config)

    monkeypatch.setattr(torch.utils.data, "DataLoader", original_loader)
    assert results["lstm_only"]["lr"] == 0.001
    assert not trials_df.empty
    assert captured_batch_sizes[:2] == [123, 123]


def test_sequence_hpo_progress_callback_handles_pruned_first_trial(monkeypatch, synthetic_config) -> None:
    config = _prepare_sequence_inputs(synthetic_config)
    config["models"]["sequence_hpo_enabled"] = ["lstm_only"]
    config["models"]["hpo"] = {"n_trials": 2, "tabular_mlp_epochs": 1, "sequence_epochs": 1, "sequence_patience": 1}

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
    progress_events: list[dict[str, object]] = []

    class _FakeLogging:
        WARNING = "WARNING"

        @staticmethod
        def set_verbosity(_value) -> None:
            return None

    class FakeStudy:
        def __init__(self) -> None:
            self.best_params = {"lr": 0.001, "dropout_rate": 0.2, "lstm_hidden_size": 8, "lstm_num_layers": 1}
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
            pruned = types.SimpleNamespace(number=0, value=None, params={}, state=types.SimpleNamespace(name="PRUNED"))
            self.trials.append(pruned)
            for callback in callbacks or []:
                callback(self, pruned)
            complete = types.SimpleNamespace(number=1, value=0.5, params={}, state=types.SimpleNamespace(name="COMPLETE"))
            self._best_value = 0.5
            self.trials.append(complete)
            for callback in callbacks or []:
                callback(self, complete)

    monkeypatch.setitem(
        sys.modules,
        "optuna",
        types.SimpleNamespace(logging=_FakeLogging(), create_study=lambda **_kwargs: FakeStudy()),
    )

    results, trials_df = tune_sequence_dataset(
        sequence_df,
        manifest,
        config,
        progress_callback=lambda **payload: progress_events.append(payload),
    )

    assert results["lstm_only"]["lr"] == 0.001
    assert progress_events[0]["state"] == "PRUNED"
    assert progress_events[0]["best_value"] is None
    assert progress_events[1]["state"] == "COMPLETE"
    assert progress_events[1]["best_value"] == pytest.approx(0.5)
    assert set(trials_df["state"]) == {"PRUNED", "COMPLETE"}


def test_sequence_hpo_patience_early_stop_completes_trial(monkeypatch, synthetic_config) -> None:
    torch = pytest.importorskip("torch")

    config = _prepare_sequence_inputs(synthetic_config)
    config["models"]["sequence_hpo_enabled"] = ["lstm_only"]
    config["models"]["hpo"] = {
        "n_trials": 1,
        "tabular_mlp_epochs": 1,
        "sequence_epochs": 3,
        "sequence_patience": 1,
        "sequence_batch_size": 16,
    }

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

    progress_events: list[dict[str, object]] = []

    class _FakeLogging:
        WARNING = "WARNING"

        @staticmethod
        def set_verbosity(_value) -> None:
            return None

    class _FakeTrial:
        number = 0
        params: dict[str, object] = {}
        value: float | None = None
        state = types.SimpleNamespace(name="COMPLETE")

        def suggest_float(self, name, low, high, log=False):
            value = low
            self.params[name] = value
            return value

        def suggest_int(self, name, low, high):
            value = low
            self.params[name] = value
            return value

        def report(self, _value, _step) -> None:
            return None

        def should_prune(self) -> bool:
            return False

    class _FakeStudy:
        def __init__(self) -> None:
            self.trials: list[types.SimpleNamespace] = []
            self.best_params: dict[str, object] = {}
            self._best_value: float | None = None

        @property
        def best_value(self):
            if self._best_value is None:
                raise ValueError("No trials are completed yet.")
            return self._best_value

        def optimize(self, objective, *, n_trials: int, show_progress_bar: bool, callbacks=None) -> None:
            assert n_trials == 1
            assert show_progress_bar is False
            trial = _FakeTrial()
            trial.value = float(objective(trial))
            self.best_params = dict(trial.params)
            self._best_value = trial.value
            self.trials.append(trial)
            for callback in callbacks or []:
                callback(self, trial)

    monkeypatch.setattr("inspire_aki.models.hpo.safe_balanced_accuracy", lambda _y_true, _y_pred: 0.5)
    monkeypatch.setitem(
        sys.modules,
        "optuna",
        types.SimpleNamespace(logging=_FakeLogging(), create_study=lambda **_kwargs: _FakeStudy(), exceptions=types.SimpleNamespace(TrialPruned=RuntimeError)),
    )

    results, trials_df = tune_sequence_dataset(
        sequence_df,
        manifest,
        config,
        progress_callback=lambda **payload: progress_events.append(payload),
    )

    assert results["lstm_only"]["lr"] == pytest.approx(config["models"]["sequence_hpo_search_spaces"]["lstm_only"]["lr"][0])
    assert not trials_df.empty
    assert list(trials_df["state"]) == ["COMPLETE"]
    assert progress_events[-1]["state"] == "COMPLETE"
    assert progress_events[-1]["best_value"] == pytest.approx(0.5)


def test_enable_sequence_cuda_benchmark_sets_cudnn_flag(monkeypatch) -> None:
    fake_torch = types.SimpleNamespace(backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False)))
    monkeypatch.setattr(sequence_module, "torch", fake_torch)

    _enable_sequence_cuda_benchmark(types.SimpleNamespace(type="cuda"))

    assert fake_torch.backends.cudnn.benchmark is True
