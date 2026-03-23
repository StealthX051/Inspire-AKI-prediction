from __future__ import annotations

from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.reporting.consort import generate_consort_outputs
from inspire_aki.reporting.curves import generate_curve_outputs
from inspire_aki.reporting.manuscript import generate_manuscript_outputs
from inspire_aki.reporting.shap import generate_shap_outputs
from inspire_aki.reporting.tables import generate_table_outputs


def run_consort(config: dict) -> dict[str, list[str]]:
    artifacts = ArtifactManager(config)
    return {"outputs": [str(path) for path in generate_consort_outputs(artifacts)]}


def run_tables(config: dict) -> dict[str, list[str]]:
    artifacts = ArtifactManager(config)
    return {"outputs": [str(path) for path in generate_table_outputs(artifacts)]}


def run_curves(config: dict) -> dict[str, list[str]]:
    artifacts = ArtifactManager(config)
    return {"outputs": [str(path) for path in generate_curve_outputs(artifacts)]}


def run_shap(config: dict) -> dict[str, list[str]]:
    artifacts = ArtifactManager(config)
    return {"outputs": [str(path) for path in generate_shap_outputs(artifacts, config)]}


def run_manuscript(config: dict) -> dict[str, list[str]]:
    artifacts = ArtifactManager(config)
    return {"outputs": [str(path) for path in generate_manuscript_outputs(artifacts, config)]}
