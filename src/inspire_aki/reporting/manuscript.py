from __future__ import annotations

from pathlib import Path

from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.registry import MANUSCRIPT_SECTIONS
from inspire_aki.reporting.consort import generate_consort_outputs
from inspire_aki.reporting.curves import generate_curve_outputs
from inspire_aki.reporting.shap import generate_shap_outputs
from inspire_aki.reporting.tables import generate_table_outputs


def generate_manuscript_outputs(artifacts: ArtifactManager, config: dict) -> list[Path]:
    generators = {
        "consort": lambda: generate_consort_outputs(artifacts),
        "tables": lambda: generate_table_outputs(artifacts),
        "curves": lambda: generate_curve_outputs(artifacts),
        "shap": lambda: generate_shap_outputs(artifacts, config),
    }
    outputs: list[Path] = []
    for section in config["reports"]["manuscript_sections"]:
        if section not in MANUSCRIPT_SECTIONS:
            raise ValueError(f"Unknown manuscript section '{section}'.")
        outputs.extend(generators[section]())
    return outputs
