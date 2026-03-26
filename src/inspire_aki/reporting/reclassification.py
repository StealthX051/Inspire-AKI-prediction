from __future__ import annotations

from pathlib import Path

import pandas as pd

from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.reporting.rendering import CellFormatRule, ColumnSpec, TableSection, TableSpec, write_table_outputs


def _display_rate(row: pd.Series) -> str:
    rate = row.get("correction_rate")
    corrected = row.get("reclassified_positives")
    missed = row.get("missed_positives")
    if pd.isna(rate):
        return "N/A"
    return f"{rate * 100:.1f}% ({corrected:.1f} / {missed:.1f})"


def _reclassification_spec(summary_df: pd.DataFrame) -> TableSpec:
    if summary_df.empty:
        sections: list[TableSection] = []
    else:
        display_rows = []
        csv_rows = []
        for model_name, model_df in summary_df.groupby("model_name", sort=True):
            row = {"model_name": model_name}
            csv_row = {"model_name": model_name}
            for comparison_name, label in [
                ("intraop_to_preop", "intraop_to_preop"),
                ("preop_to_combined", "preop_to_combined"),
            ]:
                comparison_df = model_df[model_df["comparison_name"] == comparison_name]
                if comparison_df.empty:
                    row[label] = "N/A"
                    csv_row[f"{label}_correction_rate"] = pd.NA
                    csv_row[f"{label}_reclassified_positives"] = pd.NA
                    csv_row[f"{label}_missed_positives"] = pd.NA
                    continue
                comparison_row = comparison_df.iloc[0]
                row[label] = _display_rate(comparison_row)
                csv_row[f"{label}_correction_rate"] = comparison_row["correction_rate"]
                csv_row[f"{label}_reclassified_positives"] = comparison_row["reclassified_positives"]
                csv_row[f"{label}_missed_positives"] = comparison_row["missed_positives"]
            display_rows.append(row)
            csv_rows.append(csv_row)
        sections = [TableSection(title=None, display_df=pd.DataFrame(display_rows), csv_df=pd.DataFrame(csv_rows))]
    return TableSpec(
        file_stem="reclassification_report",
        title="Patient Reclassification Analysis",
        caption="Correction rates for initially missed positive cases when moving to richer feature sets.",
        columns=[
            ColumnSpec("model_name", "Model", align="left"),
            ColumnSpec(
                "intraop_to_preop",
                "Correction Rate: Intraop -> Preop",
                csv_key="intraop_to_preop_correction_rate",
            ),
            ColumnSpec(
                "preop_to_combined",
                "Correction Rate: Preop -> Combined",
                csv_key="preop_to_combined_correction_rate",
            ),
        ],
        sections=sections,
        rules=[
            CellFormatRule("intraop_to_preop_correction_rate", mode="max"),
            CellFormatRule("preop_to_combined_correction_rate", mode="max"),
        ],
        include_section_column_in_csv=False,
    )


def generate_reclassification_outputs(artifacts: ArtifactManager) -> list[Path]:
    config = artifacts.config
    path = artifacts.paths.artifact_path("evaluation", "reclassification_summary.csv")
    if not path.exists():
        return []
    summary_df = pd.read_csv(path)
    return write_table_outputs(artifacts, _reclassification_spec(summary_df), config)
