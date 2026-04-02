from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.registry import LEGACY_DELONG_EXCLUSIONS, model_display_name
from inspire_aki.reporting.rendering import CellFormatRule, ColumnSpec, TableSection, TableSpec, write_table_outputs

_DELONG_ABBREVIATIONS = {
    "Logistic Regression": "LR",
    "GBT": "GBT",
    "Random Forest": "RF",
    "SVM (Linear)": "SVM",
    "MLP": "MLP",
    "KNN": "KNN",
    "ASA Rule": "ASA Rule",
    "Adapted GS-AKI": "Adapted GS-AKI",
    "AutoGluon": "AutoGluon",
    "LSTM": "LSTM",
    "Hybrid (MLP + LSTM)": "MLP+LSTM",
}

_DELONG_PREFIX = {"preop": "p", "intraop": "i", "combined": "c"}


def _format_ci_row(row: pd.Series) -> str:
    lower = row.get("ci_lower_95")
    upper = row.get("ci_upper_95")
    if pd.isna(lower) or pd.isna(upper):
        return "N/A"
    return f"({lower:.3f}, {upper:.3f})"


def _metrics_ci_spec(metrics_df: pd.DataFrame) -> TableSpec:
    if metrics_df.empty:
        sections: list[TableSection] = []
    else:
        display_df = metrics_df.copy()
        display_df["model_name"] = display_df["model_key"].map(model_display_name)
        display_df["estimate"] = display_df["mean"].map(lambda value: f"{value:.3f}" if pd.notna(value) else "N/A")
        display_df["ci_display"] = display_df.apply(_format_ci_row, axis=1)
        sections = [
            TableSection(
                title=None,
                display_df=display_df[["dataset_regime", "population_id", "model_name", "metric", "estimate", "ci_display"]],
                csv_df=display_df,
            )
        ]
    return TableSpec(
        file_stem="metrics_ci",
        title="Bootstrap Confidence Intervals",
        caption="Machine-readable bootstrap estimates rendered for manuscript-facing review.",
        columns=[
            ColumnSpec("dataset_regime", "Dataset", align="left"),
            ColumnSpec("population_id", "Population", align="left"),
            ColumnSpec("model_name", "Model", align="left"),
            ColumnSpec("metric", "Metric", align="left"),
            ColumnSpec("estimate", "Estimate"),
            ColumnSpec("ci_display", "95% CI"),
        ],
        sections=sections,
        include_section_column_in_csv=False,
    )


def _delong_display_name(raw_name: str) -> str:
    dataset_regime, model_key = raw_name.split("_", 1)
    prefix = _DELONG_PREFIX.get(dataset_regime, dataset_regime[:1])
    display_name = _DELONG_ABBREVIATIONS.get(model_display_name(model_key), model_display_name(model_key))
    return f"{prefix}_{display_name}"


def _filter_legacy_exclusions(matrix_df: pd.DataFrame) -> pd.DataFrame:
    excluded: set[str] = set()
    for name in matrix_df.index.astype(str):
        dataset_regime, model_key = name.split("_", 1)
        label = f"{dataset_regime}_{model_display_name(model_key)}"
        if label in LEGACY_DELONG_EXCLUSIONS:
            excluded.add(name)
    remaining = [name for name in matrix_df.index.astype(str) if name not in excluded]
    return matrix_df.loc[remaining, remaining]


def _format_p_value(value: object) -> str:
    if pd.isna(value):
        return "---"
    numeric = float(value)
    stars = "***" if numeric < 0.001 else "**" if numeric < 0.01 else "*" if numeric < 0.05 else ""
    if numeric < 0.0001:
        return f"<0.0001{stars}"
    return f"{numeric:.4f}{stars}"


def _delong_rules(columns: list[ColumnSpec]) -> list[CellFormatRule]:
    rules: list[CellFormatRule] = []
    for column in columns[1:]:
        rules.append(
            CellFormatRule(
                column_key=column.key,
                predicate=lambda _display_row, csv_row, key=column.key: bool(
                    csv_row is not None and key in csv_row.index and pd.notna(csv_row[key]) and float(csv_row[key]) < 0.05
                ),
            )
        )
    return rules


def _delong_spec(matrix_df: pd.DataFrame, *, file_stem: str, title: str) -> TableSpec:
    if matrix_df.empty:
        return TableSpec(file_stem=file_stem, title=title, caption=None, columns=[], sections=[])
    filtered = _filter_legacy_exclusions(matrix_df.copy())
    filtered.index = [_delong_display_name(name) for name in filtered.index.astype(str)]
    filtered.columns = [_delong_display_name(name) for name in filtered.columns.astype(str)]
    csv_df = filtered.reset_index().rename(columns={"index": "comparison"})
    display_df = csv_df.copy()
    for column in display_df.columns[1:]:
        display_df[column] = display_df[column].map(_format_p_value)
    columns = [ColumnSpec("comparison", "Comparison", align="left")] + [
        ColumnSpec(column, column) for column in display_df.columns[1:]
    ]
    return TableSpec(
        file_stem=file_stem,
        title=title,
        caption="Pairwise DeLong test p-values with legacy manuscript exclusions and naming.",
        columns=columns,
        sections=[TableSection(title=None, display_df=display_df, csv_df=csv_df)],
        rules=_delong_rules(columns),
        include_section_column_in_csv=False,
    )


def generate_statistics_outputs(artifacts: ArtifactManager) -> list[Path]:
    config = artifacts.config
    outputs: list[Path] = []

    metrics_ci_path = artifacts.paths.artifact_path("evaluation", "metrics_bootstrap_ci.csv")
    if metrics_ci_path.exists():
        outputs.extend(write_table_outputs(artifacts, _metrics_ci_spec(pd.read_csv(metrics_ci_path)), config))

    raw_delong_path = artifacts.paths.artifact_path("evaluation", "delong_matrix.csv")
    if raw_delong_path.exists():
        raw_df = pd.read_csv(raw_delong_path)
        raw_matrix = raw_df.set_index("model_name") if "model_name" in raw_df.columns else raw_df.set_index(raw_df.columns[0])
        outputs.extend(write_table_outputs(artifacts, _delong_spec(raw_matrix, file_stem="delong_raw", title="Pairwise DeLong Test P-Values"), config))

    corrected_path = artifacts.paths.artifact_path("evaluation", "delong_fdr_corrected.csv")
    if corrected_path.exists():
        corrected_df = pd.read_csv(corrected_path)
        corrected_matrix = corrected_df.set_index("model_name") if "model_name" in corrected_df.columns else corrected_df.set_index(corrected_df.columns[0])
        outputs.extend(
            write_table_outputs(
                artifacts,
                _delong_spec(corrected_matrix, file_stem="delong_fdr_corrected", title="Pairwise DeLong Test P-Values (FDR Corrected)"),
                config,
            )
        )

    return outputs
