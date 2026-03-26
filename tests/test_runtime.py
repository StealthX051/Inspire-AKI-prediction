from __future__ import annotations

import copy
import json
from pathlib import Path

from typer.testing import CliRunner

from inspire_aki.cli import app
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.pipelines.preprocess import run_preop
from inspire_aki.runtime import StageRuntimePlan, SystemResources, build_stage_runtime_plan, runtime_summary


def test_balanced_runtime_plan_matches_expected_32_cpu_host(monkeypatch, loaded_synthetic_config) -> None:
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

    config = copy.deepcopy(loaded_synthetic_config)
    config["runtime"]["stages"] = {}

    plan = build_stage_runtime_plan(config, "preprocess_timeseries")

    assert isinstance(plan, StageRuntimePlan)
    assert plan.profile == "balanced"
    assert plan.usable_cpus == 28
    assert plan.csv_read_threads == 8
    assert plan.preop_feature_workers == 4
    assert plan.tabular_column_workers == 8
    assert plan.timeseries_workers == 8
    assert plan.timeseries_partitions == 32
    assert plan.sequence_workers == 8
    assert plan.sequence_partitions == 32
    assert plan.evaluation_workers == 8
    assert plan.bootstrap_workers == 8
    assert plan.report_workers == 4
    assert plan.shap_workers == 2
    assert plan.train_model_threads == 16
    assert plan.hpo_model_threads == 8
    assert plan.dataloader_workers == 7
    assert plan.torch_num_threads == 8
    assert plan.sequence_use_gpu is True
    assert plan.xgb_use_gpu is False
    assert plan.max_concurrent_gpu_jobs == 1


def test_runtime_plan_downscales_on_small_cpu_only_host(monkeypatch, loaded_synthetic_config) -> None:
    resources = SystemResources(
        cpu_count=8,
        total_ram_gb=32,
        available_ram_gb=28,
        gpu_available=False,
        gpu_name=None,
        gpu_total_memory_gb=None,
        gpu_free_memory_gb=None,
    )
    monkeypatch.setattr("inspire_aki.runtime.detect_system_resources", lambda: resources)

    config = copy.deepcopy(loaded_synthetic_config)
    config["runtime"]["stages"] = {}

    plan = build_stage_runtime_plan(config, "train_tabular")

    assert plan.usable_cpus == 4
    assert plan.csv_read_threads == 2
    assert plan.preop_feature_workers == 1
    assert plan.sequence_use_gpu is False
    assert plan.gpu_enabled is False
    assert plan.dataloader_workers >= 1
    assert plan.dataloader_workers < plan.train_model_threads


def test_throughput_runtime_plan_scales_up_cpu_bound_stages(monkeypatch, loaded_synthetic_config) -> None:
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

    baseline_config = copy.deepcopy(loaded_synthetic_config)
    baseline_config["runtime"]["stages"] = {}
    throughput_config = copy.deepcopy(baseline_config)
    throughput_config["runtime"]["profile"] = "throughput"
    throughput_config["runtime"]["cpu_reserve_fraction"] = 0.0625
    throughput_config["runtime"]["cpu_reserve_min"] = 2
    throughput_config["runtime"]["ram_reserve_fraction"] = 0.1
    throughput_config["runtime"]["ram_reserve_gb_min"] = 8

    balanced_plan = build_stage_runtime_plan(baseline_config, "train_tabular")
    throughput_plan = build_stage_runtime_plan(throughput_config, "train_tabular")
    throughput_sequence_plan = build_stage_runtime_plan(throughput_config, "tune_sequence")

    assert throughput_plan.profile == "throughput"
    assert throughput_plan.usable_cpus == 30
    assert throughput_plan.train_model_threads > balanced_plan.train_model_threads
    assert throughput_plan.hpo_model_threads > balanced_plan.hpo_model_threads
    assert throughput_plan.tabular_column_workers > balanced_plan.tabular_column_workers
    assert throughput_sequence_plan.dataloader_workers == 0


def test_runtime_stage_caps_are_clipped_to_usable_cpus(monkeypatch, loaded_synthetic_config) -> None:
    resources = SystemResources(
        cpu_count=8,
        total_ram_gb=32,
        available_ram_gb=28,
        gpu_available=False,
        gpu_name=None,
        gpu_total_memory_gb=None,
        gpu_free_memory_gb=None,
    )
    monkeypatch.setattr("inspire_aki.runtime.detect_system_resources", lambda: resources)

    plan = build_stage_runtime_plan(loaded_synthetic_config, "train_tabular")

    assert plan.usable_cpus == 4
    assert plan.csv_read_threads == 4
    assert plan.tabular_column_workers == 4
    assert plan.evaluation_workers == 4
    assert plan.bootstrap_workers == 4
    assert plan.report_workers == 4
    assert plan.shap_workers == 4


def test_runtime_summary_contains_system_and_stage_plans(monkeypatch, loaded_synthetic_config) -> None:
    resources = SystemResources(
        cpu_count=16,
        total_ram_gb=64,
        available_ram_gb=48,
        gpu_available=False,
        gpu_name=None,
        gpu_total_memory_gb=None,
        gpu_free_memory_gb=None,
    )
    monkeypatch.setattr("inspire_aki.runtime.detect_system_resources", lambda: resources)

    summary = runtime_summary(loaded_synthetic_config)

    assert summary["runtime_profile"] == "balanced"
    assert summary["system_resources"]["cpu_count"] == 16
    assert "preprocess_preop" in summary["stage_runtime_plans"]
    assert summary["stage_runtime_plans"]["train_tabular"]["train_model_threads"] >= 2


def test_runtime_inspect_cli_outputs_json_summary(synthetic_config: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["runtime", "inspect", "--config", str(synthetic_config)])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert "system_resources" in payload
    assert "stage_runtime_plans" in payload
    assert payload["runtime_profile"] == "balanced"


def test_stage_manifest_records_runtime_metadata(synthetic_config: Path) -> None:
    from inspire_aki.config import load_config

    config = load_config(synthetic_config)
    artifacts = ArtifactManager(config)

    run_preop(config)
    manifest = artifacts.read_json("manifests", "preprocess_preop.json")

    assert manifest["runtime_profile"] == config["runtime"]["profile"]
    assert manifest["wall_time_seconds"] is not None
    assert manifest["wall_time_seconds"] >= 0
    assert manifest["system_resources"]["cpu_count"] >= 1
    assert manifest["stage_runtime_plan"]["stage"] == "preprocess_preop"
