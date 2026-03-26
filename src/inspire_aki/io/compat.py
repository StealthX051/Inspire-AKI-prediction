from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from inspire_aki.io.artifacts import ArtifactManager


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)


def export_legacy_datasets(artifacts: ArtifactManager) -> list[Path]:
    exported: list[Path] = []
    compat_aki = artifacts.paths.compat_aki_dir
    compat_base = artifacts.paths.compat_base_dir
    compat_results = artifacts.paths.compat_results_dir

    dataset_mapping = {
        artifacts.paths.artifact_path("features", "preop", "preop_features.csv"): [
            compat_aki / "preop_data_test.csv",
            compat_aki / "preop_data.csv",
            compat_aki / "preop_cleaned.csv",
        ],
        artifacts.paths.artifact_path("features", "intraop", "feature_engineered.csv"): [
            compat_aki / "feature_engineered.csv",
        ],
        artifacts.paths.artifact_path("datasets", "tabular", "tabular_combined.csv"): [
            compat_base / "tabular_combined.csv",
            compat_aki / "tabular_combined.csv",
        ],
        artifacts.paths.artifact_path("datasets", "tabular", "tabular_preop.csv"): [
            compat_base / "tabular_preop.csv",
            compat_aki / "tabular_preop.csv",
        ],
        artifacts.paths.artifact_path("datasets", "tabular", "tabular_intraop.csv"): [
            compat_base / "tabular_intraop.csv",
            compat_aki / "tabular_intraop.csv",
        ],
        artifacts.paths.artifact_path("datasets", "tabular", "tabular_combined_unnormalized.csv"): [
            compat_aki / "tabular_combined_unnormalized.csv",
        ],
        artifacts.paths.artifact_path("datasets", "tabular", "normalization_stats.csv"): [
            compat_base / "normalization_stats.csv",
        ],
        artifacts.paths.artifact_path("features", "timeseries", "time_series_cleaned.csv"): [
            compat_aki / "time_series_cleaned.csv",
        ],
        artifacts.paths.artifact_path("datasets", "sequence", "lstm_trainable.pkl"): [
            compat_aki / "lstm_trainable.pkl",
        ],
        artifacts.paths.artifact_path("evaluation", "metrics_summary.csv"): [
            compat_results / "metrics_summary.csv",
        ],
        artifacts.paths.artifact_path("evaluation", "metrics_bootstrap_ci.csv"): [
            compat_results / "metrics_ci.csv",
        ],
        artifacts.paths.artifact_path("evaluation", "delong_matrix.csv"): [
            compat_results / "delong_comparison_results.csv",
        ],
        artifacts.paths.artifact_path("evaluation", "delong_fdr_corrected.csv"): [
            compat_results / "delong_fdr_corrected.csv",
        ],
        artifacts.paths.artifact_path("evaluation", "reclassification_summary.csv"): [
            compat_results / "reclassification_summary.csv",
        ],
        artifacts.paths.artifact_path("reports", "tables", "performance_table.md"): [
            compat_results / "performance_table.md",
        ],
        artifacts.paths.artifact_path("reports", "tables", "performance_table_calibrated.md"): [
            compat_results / "performance_table_calibrated.md",
        ],
        artifacts.paths.artifact_path("reports", "tables", "cohort_characteristics.html"): [
            compat_results / "descriptive_table.html",
            compat_results / "cohort_characteristics.html",
        ],
        artifacts.paths.artifact_path("reports", "tables", "fill_rate_table.html"): [
            compat_results / "fill_rate_table.html",
        ],
        artifacts.paths.artifact_path("reports", "tables", "reclassification_report.html"): [
            compat_results / "reclassification_report.html",
        ],
        artifacts.paths.artifact_path("reports", "tables", "metrics_ci.csv"): [
            compat_results / "metrics_ci_report.csv",
        ],
        artifacts.paths.artifact_path("reports", "tables", "delong_raw.html"): [
            compat_results / "delong_raw.html",
        ],
        artifacts.paths.artifact_path("reports", "tables", "delong_fdr_corrected.html"): [
            compat_results / "delong_fdr_corrected.html",
        ],
    }
    for src, destinations in dataset_mapping.items():
        if not src.exists():
            continue
        for dst in destinations:
            _copy(src, dst)
            exported.append(dst)

    for subdir in ["tables", "figures"]:
        src_dir = artifacts.paths.artifact_path("reports", subdir)
        if not src_dir.exists():
            continue
        for src in src_dir.iterdir():
            if not src.is_file():
                continue
            dst = compat_results / src.name
            _copy(src, dst)
            exported.append(dst)
    return exported
