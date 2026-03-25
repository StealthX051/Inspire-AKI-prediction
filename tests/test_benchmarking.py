from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from inspire_aki.benchmarking import _run_subprocess_target, benchmark_sequence_loader, run_runtime_benchmarks
from inspire_aki.cli import app
from inspire_aki.config import load_config
from inspire_aki.pipelines.preprocess import run_intraop, run_labels, run_preop, run_sequence, run_tabular, run_timeseries


def _prepare_sequence_inputs(config_path: Path) -> dict:
    config = load_config(config_path)
    run_preop(config)
    run_intraop(config)
    run_tabular(config)
    run_labels(config)
    run_timeseries(config)
    run_sequence(config)
    return config


def test_runtime_benchmark_cli_writes_summary_files(monkeypatch, synthetic_config: Path, tmp_path: Path) -> None:
    runner = CliRunner()

    def fake_run_target(target: str, config_path: Path, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("Maximum resident set size (kbytes): 1234\n", encoding="utf-8")
        return 0, 1.25, "Maximum resident set size (kbytes): 1234\n"

    monkeypatch.setattr("inspire_aki.benchmarking._run_subprocess_target", fake_run_target)
    monkeypatch.setattr("inspire_aki.benchmarking._manifest_wall_time", lambda _config, _target: 0.75)

    result = runner.invoke(
        app,
        [
            "runtime",
            "benchmark",
            "--config",
            str(synthetic_config),
            "--profiles",
            "balanced",
            "--targets",
            "tune_tabular",
            "--repeats",
            "1",
            "--output-dir",
            str(tmp_path / "bench"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["rows"] == 1
    assert (tmp_path / "bench" / "summary.json").exists()
    assert (tmp_path / "bench" / "summary.csv").exists()


def test_run_subprocess_target_handles_missing_usr_bin_time(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("inspire_aki.benchmarking.shutil.which", lambda _name: None)
    monkeypatch.setattr(
        "inspire_aki.benchmarking.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="ok\n", stderr=""),
    )

    returncode, elapsed, raw_output = _run_subprocess_target("run_all", tmp_path / "config.yaml", tmp_path / "raw.log")

    assert returncode == 0
    assert elapsed >= 0
    assert raw_output == "ok\n"
    assert (tmp_path / "raw.log").read_text(encoding="utf-8") == "ok\n"


def test_sequence_loader_benchmark_writes_results(synthetic_config: Path, tmp_path: Path) -> None:
    pytest.importorskip("torch")

    config = _prepare_sequence_inputs(synthetic_config)
    rows = benchmark_sequence_loader(config, output_dir=tmp_path, sample_size=8, epochs=1)

    assert len(rows) == 4
    assert (tmp_path / "sequence_loader_results.json").exists()


def test_runtime_benchmarks_write_summary_files(monkeypatch, synthetic_config: Path, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "inspire_aki.benchmarking._run_subprocess_target",
        lambda target, config_path, log_path: (0, 0.5, ""),
    )
    monkeypatch.setattr("inspire_aki.benchmarking._manifest_wall_time", lambda _config, _target: None)

    summary = run_runtime_benchmarks(
        config_path=str(synthetic_config),
        profiles=["balanced"],
        targets=["train_tabular"],
        repeats=1,
        output_dir=tmp_path,
    )

    assert summary["rows"] == 1
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "summary.csv").exists()


def test_runtime_benchmarks_forward_execution_policy_filters(monkeypatch, synthetic_config: Path, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_run_target(target, config_path, log_path, **kwargs):
        captured["target"] = target
        captured["env_overrides"] = kwargs.get("env_overrides", {})
        log_path.write_text("", encoding="utf-8")
        return 0, 0.25, ""

    monkeypatch.setattr("inspire_aki.benchmarking._run_subprocess_target", fake_run_target)
    monkeypatch.setattr("inspire_aki.benchmarking._manifest_wall_time", lambda _config, _target: None)

    summary = run_runtime_benchmarks(
        config_path=str(synthetic_config),
        profiles=["balanced"],
        targets=["train_tabular"],
        repeats=1,
        output_dir=tmp_path,
        model_keys=["svm"],
        dataset_regimes=["intraop"],
        execution_policy="serial",
    )

    assert summary["rows"] == 1
    assert captured["target"] == "train_tabular"
    assert captured["env_overrides"] == {
        "INSPIRE_AKI_EXECUTION_POLICY": "serial",
        "INSPIRE_AKI_MODEL_KEYS": "svm",
        "INSPIRE_AKI_DATASET_REGIMES": "intraop",
    }


def test_benchmark_wrapper_script_invokes_runtime_benchmark(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    args_file = tmp_path / "args.txt"
    fake_cli = fake_bin / "inspire-aki"
    fake_cli.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$ARGS_FILE\"\n",
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env["ARGS_FILE"] = str(args_file)

    subprocess.run(
        [
            "bash",
            "scripts/benchmark_runtime_profiles.sh",
            "configs/aki/smoke.yaml",
            "balanced",
            "tune_tabular",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        env=env,
    )

    args = args_file.read_text(encoding="utf-8").splitlines()
    assert args[:2] == ["runtime", "benchmark"]
    assert "--profiles" in args
    assert "--targets" in args
    assert "--repeats" in args
