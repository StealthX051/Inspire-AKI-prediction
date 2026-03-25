from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

import inspire_aki.cli as cli_module
from inspire_aki.cli import app
from inspire_aki.config import load_config
from inspire_aki.pipelines.preprocess import run_intraop, run_labels, run_preop, run_sequence, run_tabular, run_timeseries
from inspire_aki.pipelines.train import run_train_sequence, run_train_tabular
from inspire_aki.pipelines.tune import run_tune_sequence, run_tune_tabular


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stub(name: str, calls: list[str]):
    def _impl(_cfg):
        calls.append(name)
        return {}

    return _impl


def _prepare_tabular_inputs(config_path: Path) -> dict:
    config = load_config(config_path)
    run_preop(config)
    run_intraop(config)
    run_tabular(config)
    run_labels(config)
    return config


def _prepare_sequence_inputs(config_path: Path) -> dict:
    config = _prepare_tabular_inputs(config_path)
    run_timeseries(config)
    run_sequence(config)
    return config


def test_run_all_emits_stage_progress_and_writes_run_events(monkeypatch, synthetic_config: Path) -> None:
    runner = CliRunner()
    config = load_config(synthetic_config)
    calls: list[str] = []

    monkeypatch.setattr(cli_module, "_cfg", lambda _path: config)
    monkeypatch.setattr(cli_module, "run_preop", _stub("run_preop", calls))
    monkeypatch.setattr(cli_module, "run_intraop", _stub("run_intraop", calls))
    monkeypatch.setattr(cli_module, "run_tabular", _stub("run_tabular", calls))
    monkeypatch.setattr(cli_module, "run_labels", _stub("run_labels", calls))
    monkeypatch.setattr(cli_module, "run_timeseries", _stub("run_timeseries", calls))
    monkeypatch.setattr(cli_module, "run_sequence", _stub("run_sequence", calls))
    monkeypatch.setattr(cli_module, "run_tune_tabular", _stub("run_tune_tabular", calls))
    monkeypatch.setattr(cli_module, "run_tune_sequence", _stub("run_tune_sequence", calls))
    monkeypatch.setattr(cli_module, "run_train_tabular", _stub("run_train_tabular", calls))
    monkeypatch.setattr(cli_module, "run_train_sequence", _stub("run_train_sequence", calls))
    monkeypatch.setattr(cli_module, "run_calibration", _stub("run_calibration", calls))
    monkeypatch.setattr(cli_module, "run_metrics", _stub("run_metrics", calls))
    monkeypatch.setattr(cli_module, "run_delong", _stub("run_delong", calls))
    monkeypatch.setattr(cli_module, "run_dca", _stub("run_dca", calls))
    monkeypatch.setattr(cli_module, "run_manuscript", _stub("run_manuscript", calls))

    result = runner.invoke(app, ["run", "all", "--config", str(synthetic_config)])

    assert result.exit_code == 0, result.stdout
    assert "START preprocess_preop" in result.stdout
    assert result.stdout.index("START preprocess_preop") < result.stdout.index('"preprocess_preop"')
    events_path = synthetic_config.parent / "artifacts" / "logs" / "run_all_events.jsonl"
    events = _read_jsonl(events_path)
    assert events[0]["event_type"] == "run_start"
    assert any(event["event_type"] == "stage_start" and event["stage"] == "preprocess_preop" for event in events)
    assert any(event["event_type"] == "stage_end" and event["stage"] == "report_manuscript" for event in events)
    assert events[-1]["event_type"] == "run_end"
    assert calls[-1] == "run_manuscript"


def test_run_all_failure_writes_stage_error_event(monkeypatch, synthetic_config: Path) -> None:
    runner = CliRunner()
    config = load_config(synthetic_config)

    def _boom(_cfg):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli_module, "_cfg", lambda _path: config)
    monkeypatch.setattr(cli_module, "run_preop", _boom)

    result = runner.invoke(app, ["run", "all", "--config", str(synthetic_config)])

    assert result.exit_code != 0
    events_path = synthetic_config.parent / "artifacts" / "logs" / "run_all_events.jsonl"
    events = _read_jsonl(events_path)
    assert any(event["event_type"] == "stage_error" and event["stage"] == "preprocess_preop" for event in events)
    assert events[-1]["status"] == "error"


def test_tune_tabular_writes_trial_progress_log(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_tabular_inputs(synthetic_config)
    config["models"]["tabular_hpo_enabled"] = ["log_reg"]

    def fake_tune_tabular_dataset(dataset_df, dataset_regime, manifest, _config, *, progress_callback=None):
        if progress_callback is not None:
            progress_callback(
                dataset_regime=dataset_regime,
                model_key="log_reg",
                trial_number=0,
                state="COMPLETE",
                value=0.9,
                best_value=0.9,
                elapsed_seconds=0.1,
            )
        return {"log_reg": {"C": 1.0}}, pd.DataFrame()

    monkeypatch.setattr("inspire_aki.pipelines.tune.tune_tabular_dataset", fake_tune_tabular_dataset)

    run_tune_tabular(config)

    progress_path = synthetic_config.parent / "artifacts" / "logs" / "tune_tabular_progress.jsonl"
    events = _read_jsonl(progress_path)
    assert any(event["event_type"] == "optuna_trial" and event["stage"] == "tune_tabular" for event in events)


def test_tune_sequence_writes_trial_progress_log(monkeypatch, synthetic_config: Path) -> None:
    config = _prepare_sequence_inputs(synthetic_config)

    def fake_tune_sequence_dataset(sequence_df, manifest, _config, *, progress_callback=None):
        if progress_callback is not None:
            progress_callback(
                dataset_regime="sequence_common",
                model_key="lstm_only",
                trial_number=0,
                state="COMPLETE",
                value=0.7,
                best_value=0.7,
                elapsed_seconds=0.2,
            )
        return {"lstm_only": {"lr": 0.001}}, pd.DataFrame()

    monkeypatch.setattr("inspire_aki.pipelines.tune.tune_sequence_dataset", fake_tune_sequence_dataset)

    run_tune_sequence(config)

    progress_path = synthetic_config.parent / "artifacts" / "logs" / "tune_sequence_progress.jsonl"
    events = _read_jsonl(progress_path)
    assert any(event["event_type"] == "optuna_trial" and event["stage"] == "tune_sequence" for event in events)


def test_train_tabular_writes_model_progress_log(monkeypatch, synthetic_config: Path) -> None:
    import numpy as np

    config = _prepare_tabular_inputs(synthetic_config)

    def fake_fit_tabular_model(**_kwargs):
        return object()

    def fake_predict_tabular_bundle(_bundle, test_df, _target):
        y_prob = np.linspace(0.2, 0.8, len(test_df), dtype=float)
        return (y_prob >= 0.5).astype(int), y_prob

    monkeypatch.setattr("inspire_aki.pipelines.train.fit_tabular_model", fake_fit_tabular_model)
    monkeypatch.setattr("inspire_aki.pipelines.train.predict_tabular_bundle", fake_predict_tabular_bundle)

    run_train_tabular(config)

    progress_path = synthetic_config.parent / "artifacts" / "logs" / "train_tabular_progress.jsonl"
    events = _read_jsonl(progress_path)
    assert any(event["event_type"] == "model_fit_complete" and event["stage"] == "train_tabular" for event in events)


def test_train_sequence_writes_validation_progress_log(monkeypatch, synthetic_config: Path) -> None:
    import numpy as np

    config = _prepare_sequence_inputs(synthetic_config)
    config["models"]["sequence_enabled"] = ["lstm_only"]

    def fake_fit_sequence_model(**kwargs):
        progress_callback = kwargs.get("progress_callback")
        if progress_callback is not None:
            progress_callback(
                epoch=1,
                val_loss=0.4,
                best_val_loss=0.4,
                patience_counter=0,
                learning_rate=0.001,
                elapsed_seconds=0.3,
            )
        return object()

    def fake_predict_sequence_bundle(_bundle, test_df):
        y_prob = np.linspace(0.2, 0.8, len(test_df), dtype=float)
        return (y_prob >= 0.5).astype(int), y_prob

    monkeypatch.setattr("inspire_aki.pipelines.train.fit_sequence_model", fake_fit_sequence_model)
    monkeypatch.setattr("inspire_aki.pipelines.train.predict_sequence_bundle", fake_predict_sequence_bundle)

    run_train_sequence(config)

    progress_path = synthetic_config.parent / "artifacts" / "logs" / "train_sequence_progress.jsonl"
    events = _read_jsonl(progress_path)
    assert any(event["event_type"] == "validation_checkpoint" and event["stage"] == "train_sequence" for event in events)
