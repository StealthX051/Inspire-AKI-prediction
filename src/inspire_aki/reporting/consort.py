from __future__ import annotations

from pathlib import Path

import pandas as pd

from inspire_aki.io.artifacts import ArtifactManager


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No consort audit rows were found.\n"
    return df.to_markdown(index=False) + "\n"


def _consort_step_and_count(row: pd.Series) -> tuple[str, str]:
    step = str(row.get("step", row.get("stage", "stage")))
    count = row.get("count", row.get("n_rows", "NA"))
    if pd.notna(count):
        try:
            count = str(int(count))
        except (TypeError, ValueError):
            count = str(count)
    else:
        count = "NA"
    return step, count


def generate_consort_outputs(artifacts: ArtifactManager) -> list[Path]:
    outputs: list[Path] = []
    audit_frames = []
    for rel_path in [
        ("cohort", "preop_audit.csv"),
        ("cohort", "labels_audit.csv"),
    ]:
        path = artifacts.paths.artifact_path(*rel_path)
        if path.exists():
            audit_frames.append(pd.read_csv(path).assign(source="/".join(rel_path)))
    consort_df = (
        pd.concat(audit_frames, ignore_index=True)
        if audit_frames
        else pd.DataFrame(columns=["step", "count", "note", "source"])
    )

    csv_path = artifacts.write_dataframe(consort_df, "reports", "tables", "consort_audit.csv")
    outputs.append(csv_path)

    md_text = "# Consort Audit\n\n" + _markdown_table(consort_df)
    md_path = artifacts.resolve("reports", "tables", "consort_audit.md")
    md_path.write_text(md_text, encoding="utf-8")
    outputs.append(md_path)

    dot_lines = ["digraph consort {", '  rankdir="LR";']
    previous_node = None
    for idx, row in consort_df.reset_index(drop=True).iterrows():
        node_name = f"n{idx}"
        step, count = _consort_step_and_count(row)
        label = f"{step}\\nN={count}"
        dot_lines.append(f'  {node_name} [shape=box, label="{label}"];')
        if previous_node is not None:
            dot_lines.append(f"  {previous_node} -> {node_name};")
        previous_node = node_name
    dot_lines.append("}")
    dot_path = artifacts.resolve("reports", "figures", "consort.dot")
    dot_path.write_text("\n".join(dot_lines) + "\n", encoding="utf-8")
    outputs.append(dot_path)
    return outputs
