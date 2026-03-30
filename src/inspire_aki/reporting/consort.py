from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from textwrap import wrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import pandas as pd

from inspire_aki.config import active_outcome_config, active_target_column
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.reporting.rendering import FigureExportSpec, ColumnSpec, TableSection, TableSpec, save_figure_variants, write_table_outputs

_PREOP_SEQUENCE = (
    "raw_operations",
    "asa_lt_6",
    "adult_only",
    "has_opend_time",
    "has_opstart_time",
    "positive_op_len_only",
    "has_height_weight",
    "nonzero_height_weight",
    "after_antype_department_merge",
    "after_prefix_exclusions",
)
_PREOP_EXCLUSION_LABELS = {
    "asa_lt_6": "ASA class 6",
    "adult_only": "Age under 18 years",
    "has_opend_time": "Missing procedure end time",
    "has_opstart_time": "Missing procedure start time",
    "positive_op_len_only": "Nonpositive procedure duration",
    "has_height_weight": "Missing height or weight",
    "nonzero_height_weight": "Zero height or weight",
    "after_antype_department_merge": "Missing anesthesia type or department",
    "after_prefix_exclusions": "Excluded ICD-10 procedure prefixes",
}
_LABEL_SEQUENCE = (
    "tabular_ops_before_labels",
    "has_preop_creatinine",
    "preop_creatinine_lt_threshold",
    "has_postop_creatinine_or_dialysis",
    "final_labeled_ops",
)
_LABEL_EXCLUSION_LABELS = {
    "has_preop_creatinine": "Missing preoperative creatinine",
    "preop_creatinine_lt_threshold": "Baseline creatinine above threshold",
    "has_postop_creatinine_or_dialysis": "No postoperative creatinine or dialysis record",
    "final_labeled_ops": "Other labeling exclusions",
}


def _safe_count(value: object) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_n(count: int | None) -> str:
    return "N/A" if count is None else f"{count:,}"


def _load_consort_audit(artifacts: ArtifactManager) -> pd.DataFrame:
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

    preop_path = artifacts.paths.artifact_path("features", "preop", "preop_features.csv")
    intraop_path = artifacts.paths.artifact_path("features", "intraop", "feature_engineered.csv")
    labels_path = _labels_artifact_path(artifacts)
    if preop_path.exists() and intraop_path.exists():
        preop_rows = len(pd.read_csv(preop_path))
        intraop_rows = len(pd.read_csv(intraop_path))
        consort_df = pd.concat(
            [
                consort_df,
                pd.DataFrame(
                    [
                        {
                            "step": "excluded_missing_intraop_features",
                            "count": max(0, preop_rows - intraop_rows),
                            "note": "Cases removed before tabular merge because intraoperative features were unavailable.",
                            "source": "derived/preop_vs_intraop",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        target_column = active_target_column(artifacts.config)
        outcome_cfg = active_outcome_config(artifacts.config)
        if target_column in labels_df.columns:
            positive = int(labels_df[target_column].astype(int).sum())
            negative = int(len(labels_df) - positive)
            consort_df = pd.concat(
                [
                    consort_df,
                    pd.DataFrame(
                        [
                            {
                                "step": "final_negative",
                                "count": negative,
                                "note": f"Final cohort without {outcome_cfg['display_name']}.",
                                "source": "derived/labels",
                            },
                            {
                                "step": "final_positive",
                                "count": positive,
                                "note": f"Final cohort with {outcome_cfg['display_name']}.",
                                "source": "derived/labels",
                            },
                        ]
                    ),
                ],
                ignore_index=True,
            )
    return consort_df


def _labels_artifact_path(artifacts: ArtifactManager) -> Path:
    candidates = [
        artifacts.paths.artifact_path("cohort", "labels.csv"),
        artifacts.paths.artifact_path("cohort", "aki_labels.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _step_counts(consort_df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in consort_df.itertuples(index=False):
        step = str(getattr(row, "step", getattr(row, "stage", "")))
        count = _safe_count(getattr(row, "count", getattr(row, "n_rows", None)))
        if step and count is not None:
            counts[step] = count
    return counts


def _last_available_count(step_counts: dict[str, int], steps: tuple[str, ...]) -> int | None:
    for step in reversed(steps):
        if step in step_counts:
            return step_counts[step]
    return None


def _sequential_exclusions(step_counts: dict[str, int], sequence: tuple[str, ...], labels: dict[str, str]) -> list[tuple[str, int]]:
    exclusions: list[tuple[str, int]] = []
    previous_count: int | None = None
    for idx, step in enumerate(sequence):
        count = step_counts.get(step)
        if count is None:
            continue
        if previous_count is None:
            previous_count = count
            continue
        removed = max(0, previous_count - count)
        if removed > 0:
            exclusions.append((labels.get(step, step.replace("_", " ")), removed))
        previous_count = count
    return exclusions


def _build_consort_layout(consort_df: pd.DataFrame, config: dict) -> dict[str, object]:
    step_counts = _step_counts(consort_df)
    outcome_cfg = active_outcome_config(config)
    identified_count = step_counts.get("raw_operations") or _last_available_count(step_counts, _PREOP_SEQUENCE)
    preop_count = _last_available_count(step_counts, _PREOP_SEQUENCE)
    final_labeled_count = step_counts.get("final_labeled_ops") or _last_available_count(step_counts, _LABEL_SEQUENCE) or preop_count
    negative_count = step_counts.get("final_negative")
    positive_count = step_counts.get("final_positive")

    preop_exclusions = _sequential_exclusions(step_counts, _PREOP_SEQUENCE, _PREOP_EXCLUSION_LABELS)
    post_preop_exclusions: list[tuple[str, int]] = []
    missing_intraop = step_counts.get("excluded_missing_intraop_features")
    if missing_intraop:
        post_preop_exclusions.append(("Missing intraoperative features", missing_intraop))
    post_preop_exclusions.extend(_sequential_exclusions(step_counts, _LABEL_SEQUENCE, _LABEL_EXCLUSION_LABELS))

    return {
        "identified": {
            "title": "Operations assessed for eligibility",
            "count": identified_count,
        },
        "analytic_preop": {
            "title": "Analytic preoperative cohort",
            "count": preop_count,
        },
        "final_labeled": {
            "title": "Final labeled analytic cohort",
            "count": final_labeled_count,
        },
        "negative": {
            "title": outcome_cfg["negative_label"],
            "count": negative_count,
        },
        "positive": {
            "title": outcome_cfg["positive_label"],
            "count": positive_count,
        },
        "preop_exclusions": {
            "title": "Excluded before analytic preoperative cohort",
            "count": sum(count for _, count in preop_exclusions),
            "items": preop_exclusions,
        },
        "post_preop_exclusions": {
            "title": "Excluded after preoperative cohort",
            "count": sum(count for _, count in post_preop_exclusions),
            "items": post_preop_exclusions,
        },
    }


def _wrap_lines(lines: list[str], width: int) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        if not line:
            wrapped.append("")
            continue
        if line.startswith("- "):
            fragments = wrap(line[2:], width=width, subsequent_indent="  ")
            if not fragments:
                wrapped.append(line)
                continue
            wrapped.append(f"- {fragments[0]}")
            wrapped.extend(fragments[1:])
            continue
        wrapped.extend(wrap(line, width=width) or [line])
    return wrapped


def _consort_box_lines(title: str, count: int | None) -> list[str]:
    return [title, f"n = {_format_n(count)}"]


def _exclusion_box_lines(title: str, count: int | None, items: list[tuple[str, int]]) -> list[str]:
    lines = [f"{title} (n = {_format_n(count)})"]
    if items:
        lines.extend(f"- {label} (n = {removed:,})" for label, removed in items if removed > 0)
    else:
        lines.append("- No recorded exclusions")
    return _wrap_lines(lines, width=34)


def _dot_label(lines: list[str], *, left_aligned: bool) -> str:
    escaped = [line.replace('"', '\\"') for line in lines]
    if left_aligned:
        return "\\l".join(escaped) + "\\l"
    return "\\n".join(escaped)


def _consort_dot(consort_df: pd.DataFrame, config: dict) -> str:
    layout = _build_consort_layout(consort_df, config)
    outcome_cfg = active_outcome_config(config)
    title = f"Study Cohort Flow and Final {outcome_cfg['display_name']} Split"
    main_node_lines = {
        "identified": _consort_box_lines(layout["identified"]["title"], layout["identified"]["count"]),
        "analytic_preop": _consort_box_lines(layout["analytic_preop"]["title"], layout["analytic_preop"]["count"]),
        "final_labeled": _consort_box_lines(layout["final_labeled"]["title"], layout["final_labeled"]["count"]),
        "negative": _consort_box_lines(layout["negative"]["title"], layout["negative"]["count"]),
        "positive": _consort_box_lines(layout["positive"]["title"], layout["positive"]["count"]),
    }
    exclusion_node_lines = {
        "excluded_preop": _exclusion_box_lines(
            layout["preop_exclusions"]["title"],
            layout["preop_exclusions"]["count"],
            layout["preop_exclusions"]["items"],
        ),
        "excluded_post_preop": _exclusion_box_lines(
            layout["post_preop_exclusions"]["title"],
            layout["post_preop_exclusions"]["count"],
            layout["post_preop_exclusions"]["items"],
        ),
    }
    lines = [
        "digraph consort {",
        f'  graph [rankdir="TB", splines=ortho, newrank=true, ordering="out", pad="0.18", nodesep="0.55", ranksep="0.9", labelloc="t", labeljust="c", fontname="Times New Roman", fontsize=20, label="{title}"];',
        '  node [shape=box, style="rounded,filled", fontname="Times New Roman", fontsize=12, penwidth=1.5, color="#667d93", fillcolor="#f8fafc", margin="0.20,0.14"];',
        '  edge [color="#667d93", penwidth=1.5, arrowsize=0.75];',
        f'  identified [label="{_dot_label(main_node_lines["identified"], left_aligned=False)}", width=3.8, height=1.0];',
        f'  analytic_preop [label="{_dot_label(main_node_lines["analytic_preop"], left_aligned=False)}", width=3.8, height=1.0];',
        f'  final_labeled [label="{_dot_label(main_node_lines["final_labeled"], left_aligned=False)}", width=3.8, height=1.0, fillcolor="#f2f7fb"];',
        f'  final_negative [label="{_dot_label(main_node_lines["negative"], left_aligned=False)}", width=3.0, height=0.95, fillcolor="#f8fafc"];',
        f'  final_positive [label="{_dot_label(main_node_lines["positive"], left_aligned=False)}", width=3.0, height=0.95, fillcolor="#edf5fb"];',
        f'  excluded_preop [label="{_dot_label(exclusion_node_lines["excluded_preop"], left_aligned=True)}", width=3.35, fontsize=10.5, style="rounded,filled,dashed", fillcolor="#fbfcfd", margin="0.16,0.12"];',
        f'  excluded_post_preop [label="{_dot_label(exclusion_node_lines["excluded_post_preop"], left_aligned=True)}", width=3.35, fontsize=10.5, style="rounded,filled,dashed", fillcolor="#fbfcfd", margin="0.16,0.12"];',
        "  identified:s -> analytic_preop:n;",
        "  analytic_preop:s -> final_labeled:n;",
        "  final_labeled:s -> final_negative:n [minlen=1];",
        "  final_labeled:s -> final_positive:n [minlen=1];",
        '  analytic_preop:e -> excluded_preop:w [style=dashed, arrowhead=none, constraint=false, minlen=2];',
        '  final_labeled:e -> excluded_post_preop:w [style=dashed, arrowhead=none, constraint=false, minlen=2];',
        "  { rank=same; analytic_preop; excluded_preop; }",
        "  { rank=same; final_labeled; excluded_post_preop; }",
        "  { rank=same; final_negative; final_positive; }",
        "  final_negative -> final_positive [style=invis, weight=20];",
        "  excluded_preop -> excluded_post_preop [style=invis, weight=20];",
        "}",
    ]
    return "\n".join(lines) + "\n"


def _render_consort_graphviz(dot_path: Path, artifacts: ArtifactManager, config: dict) -> list[Path]:
    outputs: list[Path] = []
    dot_binary = shutil.which("dot")
    if dot_binary is None:
        return outputs
    png_dpi = int(config.get("reports", {}).get("figure_png_dpi", 600))
    for fmt in ("svg", "png"):
        out_path = artifacts.resolve("reports", "figures", f"consort.{fmt}")
        command = [dot_binary, f"-T{fmt}", str(dot_path), "-o", str(out_path)]
        if fmt == "png":
            command.insert(1, f"-Gdpi={png_dpi}")
        subprocess.run(command, check=True)
        outputs.append(out_path)
    return outputs


def _box_height(lines: list[str], *, line_height: float, minimum: float) -> float:
    return max(minimum, 0.05 + len(lines) * line_height)


def _draw_box(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    lines: list[str],
    facecolor: str,
    edgecolor: str,
    fontsize: float,
    dashed: bool = False,
    centered: bool = True,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=1.35,
        linestyle=(0, (4, 2)) if dashed else "solid",
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    text_x = x + width / 2 if centered else x + 0.025
    ax.text(
        text_x,
        y + height / 2,
        "\n".join(lines),
        ha="center" if centered else "left",
        va="center",
        fontsize=fontsize,
        fontweight="bold" if centered else "normal",
        linespacing=1.35,
        color="#1f2933",
    )


def _draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "lw": 1.5, "color": "#607286", "shrinkA": 2, "shrinkB": 2},
    )


def _draw_consort_figure(consort_df: pd.DataFrame, config: dict):
    layout = _build_consort_layout(consort_df, config)
    outcome_cfg = active_outcome_config(config)
    fig, ax = plt.subplots(figsize=(10.8, 9.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    main_x = 0.10
    main_w = 0.42
    side_x = 0.62
    side_w = 0.30
    left_branch_x = 0.05
    right_branch_x = 0.43
    branch_w = 0.34

    main_lines = {
        "identified": _consort_box_lines(layout["identified"]["title"], layout["identified"]["count"]),
        "analytic_preop": _consort_box_lines(layout["analytic_preop"]["title"], layout["analytic_preop"]["count"]),
        "final_labeled": _consort_box_lines(layout["final_labeled"]["title"], layout["final_labeled"]["count"]),
        "negative": _consort_box_lines(layout["negative"]["title"], layout["negative"]["count"]),
        "positive": _consort_box_lines(layout["positive"]["title"], layout["positive"]["count"]),
    }
    exclusion_lines = {
        "preop": _exclusion_box_lines(
            layout["preop_exclusions"]["title"],
            layout["preop_exclusions"]["count"],
            layout["preop_exclusions"]["items"],
        ),
        "post_preop": _exclusion_box_lines(
            layout["post_preop_exclusions"]["title"],
            layout["post_preop_exclusions"]["count"],
            layout["post_preop_exclusions"]["items"],
        ),
    }

    box_h = _box_height(main_lines["identified"], line_height=0.05, minimum=0.12)
    branch_h = _box_height(main_lines["negative"], line_height=0.047, minimum=0.11)
    preop_excl_h = _box_height(exclusion_lines["preop"], line_height=0.032, minimum=0.16)
    post_preop_excl_h = _box_height(exclusion_lines["post_preop"], line_height=0.032, minimum=0.16)

    positions = {
        "identified": (main_x, 0.82, main_w, box_h),
        "analytic_preop": (main_x, 0.56, main_w, box_h),
        "final_labeled": (main_x, 0.30, main_w, box_h),
        "negative": (left_branch_x, 0.04, branch_w, branch_h),
        "positive": (right_branch_x, 0.04, branch_w, branch_h),
        "preop_exclusions": (side_x, 0.54, side_w, preop_excl_h),
        "post_preop_exclusions": (side_x, 0.26, side_w, post_preop_excl_h),
    }

    _draw_box(
        ax,
        x=positions["identified"][0],
        y=positions["identified"][1],
        width=positions["identified"][2],
        height=positions["identified"][3],
        lines=main_lines["identified"],
        facecolor="#f8fafc",
        edgecolor="#607286",
        fontsize=10.5,
    )
    _draw_box(
        ax,
        x=positions["analytic_preop"][0],
        y=positions["analytic_preop"][1],
        width=positions["analytic_preop"][2],
        height=positions["analytic_preop"][3],
        lines=main_lines["analytic_preop"],
        facecolor="#f8fafc",
        edgecolor="#607286",
        fontsize=10.5,
    )
    _draw_box(
        ax,
        x=positions["final_labeled"][0],
        y=positions["final_labeled"][1],
        width=positions["final_labeled"][2],
        height=positions["final_labeled"][3],
        lines=main_lines["final_labeled"],
        facecolor="#f3f7fb",
        edgecolor="#607286",
        fontsize=10.5,
    )
    _draw_box(
        ax,
        x=positions["negative"][0],
        y=positions["negative"][1],
        width=positions["negative"][2],
        height=positions["negative"][3],
        lines=main_lines["negative"],
        facecolor="#f8fafc",
        edgecolor="#607286",
        fontsize=10.0,
    )
    _draw_box(
        ax,
        x=positions["positive"][0],
        y=positions["positive"][1],
        width=positions["positive"][2],
        height=positions["positive"][3],
        lines=main_lines["positive"],
        facecolor="#eef5fb",
        edgecolor="#607286",
        fontsize=10.0,
    )
    _draw_box(
        ax,
        x=positions["preop_exclusions"][0],
        y=positions["preop_exclusions"][1],
        width=positions["preop_exclusions"][2],
        height=positions["preop_exclusions"][3],
        lines=exclusion_lines["preop"],
        facecolor="#fbfcfd",
        edgecolor="#7b8b99",
        fontsize=8.7,
        dashed=True,
        centered=False,
    )
    _draw_box(
        ax,
        x=positions["post_preop_exclusions"][0],
        y=positions["post_preop_exclusions"][1],
        width=positions["post_preop_exclusions"][2],
        height=positions["post_preop_exclusions"][3],
        lines=exclusion_lines["post_preop"],
        facecolor="#fbfcfd",
        edgecolor="#7b8b99",
        fontsize=8.7,
        dashed=True,
        centered=False,
    )

    _draw_arrow(
        ax,
        (main_x + main_w / 2, positions["identified"][1]),
        (main_x + main_w / 2, positions["analytic_preop"][1] + positions["analytic_preop"][3]),
    )
    _draw_arrow(
        ax,
        (main_x + main_w / 2, positions["analytic_preop"][1]),
        (main_x + main_w / 2, positions["final_labeled"][1] + positions["final_labeled"][3]),
    )

    split_y = positions["negative"][1] + positions["negative"][3] + 0.05
    parent_x = main_x + main_w / 2
    ax.plot([parent_x, parent_x], [positions["final_labeled"][1], split_y], color="#607286", lw=1.5)
    left_center = positions["negative"][0] + positions["negative"][2] / 2
    right_center = positions["positive"][0] + positions["positive"][2] / 2
    ax.plot([left_center, right_center], [split_y, split_y], color="#607286", lw=1.5)
    _draw_arrow(ax, (left_center, split_y), (left_center, positions["negative"][1] + positions["negative"][3]))
    _draw_arrow(ax, (right_center, split_y), (right_center, positions["positive"][1] + positions["positive"][3]))

    analytic_mid_y = positions["analytic_preop"][1] + positions["analytic_preop"][3] / 2
    final_mid_y = positions["final_labeled"][1] + positions["final_labeled"][3] / 2
    ax.plot([main_x + main_w, side_x], [analytic_mid_y, analytic_mid_y], color="#7b8b99", lw=1.2, linestyle=(0, (3, 2)))
    ax.plot([main_x + main_w, side_x], [final_mid_y, final_mid_y], color="#7b8b99", lw=1.2, linestyle=(0, (3, 2)))

    ax.text(
        0.5,
        0.985,
        f"Study Cohort Flow and Final {outcome_cfg['display_name']} Split",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#1f2933",
    )
    return fig


def generate_consort_outputs(artifacts: ArtifactManager) -> list[Path]:
    config = artifacts.config
    outputs: list[Path] = []
    consort_df = _load_consort_audit(artifacts)
    table_spec = TableSpec(
        file_stem="consort_audit",
        title="CONSORT Audit",
        caption="Audit rows and derived cohort flow steps used for the manuscript consort diagram.",
        columns=[
            ColumnSpec("step", "Step", align="left"),
            ColumnSpec("count", "Count"),
            ColumnSpec("note", "Note", align="left"),
            ColumnSpec("source", "Source", align="left"),
        ],
        sections=[TableSection(title=None, display_df=consort_df, csv_df=consort_df)],
        include_section_column_in_csv=False,
    )
    outputs.extend(write_table_outputs(artifacts, table_spec, config))

    dot_path = artifacts.resolve("reports", "figures", "consort.dot")
    dot_path.write_text(_consort_dot(consort_df, config), encoding="utf-8")
    outputs.append(dot_path)

    try:
        rendered_outputs = _render_consort_graphviz(dot_path, artifacts, config)
        if rendered_outputs:
            outputs.extend(rendered_outputs)
        else:
            raise FileNotFoundError("graphviz dot binary not available")
    except (subprocess.CalledProcessError, FileNotFoundError):
        fig = _draw_consort_figure(consort_df, config)
        try:
            outputs.extend(save_figure_variants(fig, artifacts, FigureExportSpec(stem="consort"), config))
        finally:
            plt.close(fig)
    return outputs
