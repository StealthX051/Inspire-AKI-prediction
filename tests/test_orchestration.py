from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import inspire_aki.cli as cli_module
import inspire_aki.orchestration as orchestration_module
from inspire_aki.cli import app
from inspire_aki.config import load_config
from inspire_aki.orchestration import StageSubprocessResult, run_overlap_stages


class _FakeProcess:
    def __init__(self, polls: list[int | None], wait_return: int) -> None:
        self._polls = list(polls)
        self._wait_return = wait_return

    def poll(self) -> int | None:
        if self._polls:
            value = self._polls.pop(0)
            if value is not None:
                self._wait_return = value
            return value
        return self._wait_return

    def wait(self, timeout: float | None = None) -> int:
        return self._wait_return


def test_run_overlap_stages_collects_successful_results(monkeypatch, tmp_path: Path) -> None:
    def fake_launch(stage: str, *, config_path: str | None, log_path: Path):
        process = _FakeProcess([None, 0] if stage == "tune_sequence" else [0], 0)
        return SimpleNamespace(stage=stage, process=process, log_path=log_path, log_handle=None, started_at=0.0)

    def fake_finalize(handle):
        return StageSubprocessResult(
            stage=handle.stage,
            returncode=handle.process.wait(),
            wall_time_seconds=1.0,
            log_path=handle.log_path,
            payload={"stage": handle.stage},
        )

    monkeypatch.setattr(orchestration_module, "launch_stage_subprocess", fake_launch)
    monkeypatch.setattr(orchestration_module, "finalize_stage_subprocess", fake_finalize)

    results = run_overlap_stages(["tune_sequence", "train_tabular"], config_path="config.yaml", log_dir=tmp_path)

    assert list(results) == ["train_tabular", "tune_sequence"] or list(results) == ["tune_sequence", "train_tabular"]
    assert results["tune_sequence"].returncode == 0
    assert results["train_tabular"].payload == {"stage": "train_tabular"}


def test_run_overlap_stages_terminates_remaining_stage_on_failure(monkeypatch, tmp_path: Path) -> None:
    terminated: list[str] = []

    def fake_launch(stage: str, *, config_path: str | None, log_path: Path):
        if stage == "tune_sequence":
            process = _FakeProcess([1], 1)
        else:
            process = _FakeProcess([None, None], 0)
        return SimpleNamespace(stage=stage, process=process, log_path=log_path, log_handle=None, started_at=0.0)

    def fake_finalize(handle):
        return StageSubprocessResult(
            stage=handle.stage,
            returncode=handle.process.wait(),
            wall_time_seconds=1.0,
            log_path=handle.log_path,
            payload={"stage": handle.stage},
        )

    def fake_terminate(handle):
        terminated.append(handle.stage)
        handle.process._wait_return = -15

    monkeypatch.setattr(orchestration_module, "launch_stage_subprocess", fake_launch)
    monkeypatch.setattr(orchestration_module, "finalize_stage_subprocess", fake_finalize)
    monkeypatch.setattr(orchestration_module, "terminate_stage_subprocess", fake_terminate)

    results = run_overlap_stages(["tune_sequence", "train_tabular"], config_path="config.yaml", log_dir=tmp_path)

    assert results["tune_sequence"].returncode == 1
    assert results["train_tabular"].returncode == -15
    assert terminated == ["train_tabular"]


def test_run_all_overlap_executes_parallel_branch_before_tail(monkeypatch, synthetic_config: Path) -> None:
    runner = CliRunner()
    config = load_config(synthetic_config)
    config["runtime"]["orchestration"]["mode"] = "overlap"
    order: list[str] = []

    def fake_run_stage(*, stage_name, runner, config, progress):
        order.append(stage_name)
        return {stage_name: True}

    def fake_overlap_branch(**_kwargs):
        order.append("overlap_branch")
        return {"tune_sequence": {}, "train_tabular": {}}

    monkeypatch.setattr(cli_module, "_cfg", lambda _path: config)
    monkeypatch.setattr(cli_module, "_run_stage", fake_run_stage)
    monkeypatch.setattr(cli_module, "_run_overlap_branch", fake_overlap_branch)

    result = runner.invoke(app, ["run", "all", "--config", str(synthetic_config)])

    assert result.exit_code == 0, result.stdout
    assert order == [
        "preprocess_preop",
        "preprocess_intraop",
        "preprocess_tabular",
        "preprocess_labels",
        "preprocess_timeseries",
        "preprocess_sequence",
        "tune_tabular",
        "overlap_branch",
        "train_sequence",
        "evaluate_calibrate",
        "evaluate_metrics",
        "evaluate_delong",
        "evaluate_dca",
        "report_manuscript",
    ]


def test_run_all_serial_mode_preserves_stage_order(monkeypatch, synthetic_config: Path) -> None:
    runner = CliRunner()
    config = load_config(synthetic_config)
    config["runtime"]["orchestration"]["mode"] = "serial"
    order: list[str] = []

    def fake_run_stage(*, stage_name, runner, config, progress):
        order.append(stage_name)
        return {stage_name: True}

    monkeypatch.setattr(cli_module, "_cfg", lambda _path: config)
    monkeypatch.setattr(cli_module, "_run_stage", fake_run_stage)

    result = runner.invoke(app, ["run", "all", "--config", str(synthetic_config)])

    assert result.exit_code == 0, result.stdout
    assert order == [
        "preprocess_preop",
        "preprocess_intraop",
        "preprocess_tabular",
        "preprocess_labels",
        "preprocess_timeseries",
        "preprocess_sequence",
        "tune_tabular",
        "tune_sequence",
        "train_tabular",
        "train_sequence",
        "evaluate_calibrate",
        "evaluate_metrics",
        "evaluate_delong",
        "evaluate_dca",
        "report_manuscript",
    ]
