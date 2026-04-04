from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner
import yaml

import inspire_aki.cli as cli_module
from inspire_aki.cli import app


def test_run_all_smoke(synthetic_config: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run", "all", "--config", str(synthetic_config)])
    assert result.exit_code == 0, result.stdout

    artifacts_dir = synthetic_config.parent / "artifacts"
    assert (artifacts_dir / "predictions" / "calibrated_predictions.parquet").exists()
    assert (artifacts_dir / "evaluation" / "metrics_summary.csv").exists()
    assert (artifacts_dir / "evaluation" / "reclassification_summary.csv").exists()
    assert (artifacts_dir / "reports" / "tables" / "performance_table.csv").exists()
    assert (artifacts_dir / "reports" / "tables" / "performance_table_calibrated.html").exists()
    assert (artifacts_dir / "reports" / "tables" / "reclassification_report.html").exists()
    assert (artifacts_dir / "reports" / "figures" / "roc_curves_preop.png").exists()
    assert (artifacts_dir / "reports" / "figures" / "roc_curves_preop.svg").exists()


def test_run_all_grouped_smoke(synthetic_config: Path) -> None:
    runner = CliRunner()
    grouped_config = synthetic_config.parent / "grouped_config.yaml"
    config = yaml.safe_load(synthetic_config.read_text(encoding="utf-8"))
    config["evaluation_mode"] = "grouped_nested_cv"
    config["runtime"]["orchestration"]["mode"] = "serial"
    config["models"]["tabular_enabled"] = ["log_reg"]
    config["models"]["tabular_hpo_enabled"] = []
    config["models"]["sequence_enabled"] = []
    config["models"]["sequence_hpo_enabled"] = []
    config["reports"]["shap_jobs"] = []
    grouped_config.write_text(yaml.safe_dump(config), encoding="utf-8")

    result = runner.invoke(app, ["run", "all", "--config", str(grouped_config)])
    assert result.exit_code == 0, result.stdout

    artifacts_dir = synthetic_config.parent / "artifacts"
    assert (artifacts_dir / "datasets" / "splits" / "grouped_nested_cv_preop.parquet").exists()
    assert (artifacts_dir / "datasets" / "splits" / "grouped_nested_cv_sequence.parquet").exists()
    assert (artifacts_dir / "predictions" / "calibrated_predictions.parquet").exists()
    assert (artifacts_dir / "evaluation" / "metrics_summary.csv").exists()
    assert (artifacts_dir / "reports" / "tables" / "performance_table.csv").exists()
    assert (artifacts_dir / "reports" / "tables" / "performance_table_calibrated.html").exists()


@pytest.mark.parametrize(
    ("command", "runner_attr"),
    [
        (["preprocess", "preop"], "run_preop"),
        (["preprocess", "intraop"], "run_intraop"),
        (["preprocess", "tabular"], "run_tabular"),
        (["preprocess", "labels"], "run_labels"),
        (["preprocess", "timeseries"], "run_timeseries"),
        (["preprocess", "sequence"], "run_sequence"),
        (["tune", "tabular"], "run_tune_tabular"),
        (["tune", "sequence"], "run_tune_sequence"),
        (["train", "tabular"], "run_train_tabular"),
        (["train", "sequence"], "run_train_sequence"),
        (["evaluate", "calibrate"], "run_calibration"),
        (["evaluate", "generate"], "run_evaluate_generate"),
        (["evaluate", "metrics"], "run_metrics"),
        (["evaluate", "delong"], "run_delong"),
        (["evaluate", "dca"], "run_dca"),
        (["evaluate", "reclassification"], "run_reclassification"),
        (["explain", "shap"], "run_shap"),
        (["report", "consort"], "run_consort"),
        (["report", "tables"], "run_tables"),
        (["report", "curves"], "run_curves"),
        (["report", "procedure-audit"], "run_procedure_audit"),
        (["report", "manuscript"], "run_manuscript"),
    ],
)
def test_stage_command_dispatches_to_current_runner(
    monkeypatch,
    synthetic_config: Path,
    command: list[str],
    runner_attr: str,
) -> None:
    runner = CliRunner()
    calls: list[dict] = []

    def _fake_runner(cfg):
        calls.append(cfg)
        return {"runner": runner_attr}

    monkeypatch.setattr(cli_module, runner_attr, _fake_runner)

    result = runner.invoke(app, [*command, "--config", str(synthetic_config)])

    assert result.exit_code == 0, result.stdout
    assert len(calls) == 1
    payload = json.loads(result.stdout)
    assert payload == {"runner": runner_attr}


def test_stage_command_keyboard_interrupt_exits_cleanly(monkeypatch, synthetic_config: Path) -> None:
    runner = CliRunner()

    def _interrupt(_cfg):
        raise KeyboardInterrupt()

    monkeypatch.setattr(cli_module, "run_tune_sequence", _interrupt)

    result = runner.invoke(app, ["tune", "sequence", "--config", str(synthetic_config)])

    assert result.exit_code == 130
    assert "Interrupted tune_sequence; exiting cleanly (130)." in result.output


def test_evaluate_generate_command_invokes_backend(monkeypatch, synthetic_config: Path) -> None:
    runner = CliRunner()
    calls: list[str] = []

    def _fake_generate(_cfg):
        calls.append("evaluate_generate")
        return {"ok": True}

    monkeypatch.setattr(cli_module, "run_evaluate_generate", _fake_generate)

    result = runner.invoke(app, ["evaluate", "generate", "--config", str(synthetic_config)])

    assert result.exit_code == 0, result.stdout
    assert calls == ["evaluate_generate"]


def test_compat_export_legacy_command_emits_exported_paths(monkeypatch, synthetic_config: Path) -> None:
    runner = CliRunner()
    exported_paths = [
        synthetic_config.parent / "compat_aki" / "preop_data.csv",
        synthetic_config.parent / "compat_results" / "performance_table.csv",
    ]

    monkeypatch.setattr(cli_module, "export_legacy_datasets", lambda _artifacts: exported_paths)

    result = runner.invoke(app, ["compat", "export-legacy", "--config", str(synthetic_config)])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload == {"outputs": [str(path) for path in exported_paths]}


def test_runtime_benchmark_relative_output_dir_uses_artifact_root(monkeypatch, synthetic_config: Path) -> None:
    runner = CliRunner()
    captured: dict[str, Path] = {}

    def _fake_benchmarks(**kwargs):
        captured["output_dir"] = kwargs["output_dir"]
        return {"ok": True}

    monkeypatch.setattr(cli_module, "run_runtime_benchmarks", _fake_benchmarks)

    result = runner.invoke(
        app,
        ["runtime", "benchmark", "--config", str(synthetic_config), "--output-dir", "bench-rel"],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["output_dir"] == synthetic_config.parent / "artifacts" / "bench-rel"
