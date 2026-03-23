from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.registry import model_display_name


def _format_numeric(series: pd.Series) -> str:
    return f"{series.mean():.2f} +/- {series.std(ddof=0):.2f}"


def _cohort_characteristics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "aki_boolean" not in df.columns:
        return pd.DataFrame()
    candidate_cols = [
        "age",
        "sex",
        "BMI",
        "BSA",
        "asa",
        "emop",
        "booking_case_length",
        "op_len",
        "preop_creatinine",
        "num_card_events",
        "fluids_agg",
        "equiv_MAC_totals",
    ]
    rows: list[dict[str, object]] = []
    for column in [col for col in candidate_cols if col in df.columns]:
        series = df[column]
        if pd.api.types.is_bool_dtype(series) or set(series.dropna().unique()).issubset({0, 1, True, False}):
            rows.append(
                {
                    "feature": column,
                    "overall": f"{series.mean() * 100:.1f}%",
                    "aki_negative": f"{df.loc[df['aki_boolean'] == 0, column].mean() * 100:.1f}%",
                    "aki_positive": f"{df.loc[df['aki_boolean'] == 1, column].mean() * 100:.1f}%",
                }
            )
        elif pd.api.types.is_numeric_dtype(series):
            rows.append(
                {
                    "feature": column,
                    "overall": _format_numeric(series.dropna()),
                    "aki_negative": _format_numeric(df.loc[df["aki_boolean"] == 0, column].dropna()),
                    "aki_positive": _format_numeric(df.loc[df["aki_boolean"] == 1, column].dropna()),
                }
            )
    return pd.DataFrame(rows)


def generate_table_outputs(artifacts: ArtifactManager) -> list[Path]:
    outputs: list[Path] = []

    performance_path = artifacts.paths.artifact_path("evaluation", "metrics_summary.csv")
    if performance_path.exists():
        metrics_df = pd.read_csv(performance_path)
        metrics_df["model_name"] = metrics_df["model_key"].map(model_display_name)
        metrics_df = metrics_df.sort_values(["dataset_regime", "auroc"], ascending=[True, False])
        metrics_csv = artifacts.write_dataframe(metrics_df, "reports", "tables", "performance_table.csv")
        outputs.append(metrics_csv)
        metrics_md = artifacts.resolve("reports", "tables", "performance_table.md")
        metrics_md.write_text(metrics_df.to_markdown(index=False) + "\n", encoding="utf-8")
        outputs.append(metrics_md)

    fill_path = artifacts.paths.artifact_path("features", "fill_rates.csv")
    if fill_path.exists():
        fill_df = pd.read_csv(fill_path).sort_values("fill_rate", ascending=False)
        fill_html = artifacts.resolve("reports", "tables", "fill_rate_table.html")
        fill_html.write_text(fill_df.to_html(index=False), encoding="utf-8")
        outputs.append(fill_html)

    cohort_path = artifacts.paths.artifact_path("datasets", "tabular", "tabular_combined_labeled.csv")
    if cohort_path.exists():
        cohort_df = pd.read_csv(cohort_path)
        characteristics = _cohort_characteristics(cohort_df)
        if not characteristics.empty:
            char_csv = artifacts.write_dataframe(characteristics, "reports", "tables", "cohort_characteristics.csv")
            outputs.append(char_csv)
            char_html = artifacts.resolve("reports", "tables", "cohort_characteristics.html")
            char_html.write_text(characteristics.to_html(index=False), encoding="utf-8")
            outputs.append(char_html)

    return outputs

