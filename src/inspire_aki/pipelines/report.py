from __future__ import annotations

from pathlib import Path
from time import perf_counter

from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.reporting.consort import generate_consort_outputs
from inspire_aki.reporting.curves import generate_curve_outputs
from inspire_aki.reporting.manuscript import generate_manuscript_outputs
from inspire_aki.reporting.shap import generate_shap_outputs
from inspire_aki.reporting.tables import generate_table_outputs
from inspire_aki.runtime import build_stage_runtime_plan


def run_consort(config: dict) -> dict[str, list[str]]:
    stage_name = "report_consort"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    outputs = [str(path) for path in generate_consort_outputs(artifacts)]
    artifacts.write_manifest(
        stage_name,
        ["manifests", "report_consort.json"],
        outputs=[artifacts.relative(artifacts.paths.artifact_path("reports", "tables", "consort_audit.csv"))],
        metadata={"n_outputs": len(outputs)},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return {"outputs": outputs}


def run_tables(config: dict) -> dict[str, list[str]]:
    stage_name = "report_tables"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    outputs = [str(path) for path in generate_table_outputs(artifacts)]
    artifacts.write_manifest(
        stage_name,
        ["manifests", "report_tables.json"],
        outputs=[artifacts.relative(path) for path in map(Path, outputs)],
        metadata={"n_outputs": len(outputs)},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return {"outputs": outputs}


def run_curves(config: dict) -> dict[str, list[str]]:
    stage_name = "report_curves"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    outputs = [str(path) for path in generate_curve_outputs(artifacts, config)]
    artifacts.write_manifest(
        stage_name,
        ["manifests", "report_curves.json"],
        outputs=[artifacts.relative(path) for path in map(Path, outputs)],
        metadata={"n_outputs": len(outputs)},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return {"outputs": outputs}


def run_shap(config: dict) -> dict[str, list[str]]:
    stage_name = "report_shap"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    outputs = [str(path) for path in generate_shap_outputs(artifacts, config)]
    artifacts.write_manifest(
        stage_name,
        ["manifests", "report_shap.json"],
        outputs=[artifacts.relative(path) for path in map(Path, outputs)],
        metadata={"n_outputs": len(outputs)},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return {"outputs": outputs}


def run_manuscript(config: dict) -> dict[str, list[str]]:
    stage_name = "report_manuscript"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    outputs = [str(path) for path in generate_manuscript_outputs(artifacts, config)]
    artifacts.write_manifest(
        stage_name,
        ["manifests", "report_manuscript.json"],
        outputs=[artifacts.relative(path) for path in map(Path, outputs)],
        metadata={"n_outputs": len(outputs)},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return {"outputs": outputs}
