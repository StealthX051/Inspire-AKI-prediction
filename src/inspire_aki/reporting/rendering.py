from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from inspire_aki.io.artifacts import ArtifactManager


def report_table_formats(config: dict | None) -> tuple[str, ...]:
    if not isinstance(config, dict):
        return ("html", "md", "csv")
    return tuple(config.get("reports", {}).get("table_formats", ["html", "md", "csv"]))


def report_figure_formats(config: dict | None) -> tuple[str, ...]:
    if not isinstance(config, dict):
        return ("png", "svg")
    return tuple(config.get("reports", {}).get("figure_formats", ["png", "svg"]))


def report_figure_png_dpi(config: dict | None) -> int:
    if not isinstance(config, dict):
        return 600
    return int(config.get("reports", {}).get("figure_png_dpi", config.get("reports", {}).get("figure_dpi", 600)))


def report_primary_figure_subdir(config: dict | None) -> str | None:
    if not isinstance(config, dict):
        return "primary_figures"
    reports_cfg = config.get("reports", {})
    if not bool(reports_cfg.get("route_top_level_figures_to_primary_figures", True)):
        return None
    subdir = str(reports_cfg.get("primary_figure_subdir", "primary_figures")).strip()
    return subdir or None


def report_primary_figure_directory_parts(config: dict | None) -> tuple[str, ...]:
    subdir = report_primary_figure_subdir(config)
    if subdir is None:
        return ("reports", "figures")
    return ("reports", "figures", subdir)


@contextmanager
def report_figure_style_context(config: dict | None = None):
    import matplotlib as mpl

    style_variant = "legacy_manuscript"
    if isinstance(config, dict):
        style_variant = str(config.get("reports", {}).get("style_variant", "legacy_manuscript")).strip() or "legacy_manuscript"

    rc_params = {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelcolor": "#1f2933",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "text.color": "#1f2933",
        "axes.edgecolor": "#52606d",
        "axes.linewidth": 1.1,
        "figure.facecolor": "#ffffff",
        "axes.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "legend.frameon": False,
        "legend.fontsize": 10,
    }
    if style_variant == "legacy_manuscript":
        rc_params["axes.grid"] = False
    with mpl.rc_context(rc=rc_params):
        yield


@dataclass(frozen=True)
class ColumnSpec:
    key: str
    label: str
    align: str = "center"
    display_key: str | None = None
    ci_display_key: str | None = None
    csv_key: str | None = None
    csv_ci_lower_key: str | None = None
    csv_ci_upper_key: str | None = None
    gradient: bool = False


@dataclass(frozen=True)
class CellFormatRule:
    column_key: str
    mode: str = "max"
    predicate: Callable[[pd.Series, pd.Series | None], bool] | None = None
    css_class: str = "best-cell"
    markdown_wrapper: tuple[str, str] = ("**", "**")


@dataclass
class TableSection:
    title: str | None
    display_df: pd.DataFrame
    csv_df: pd.DataFrame | None = None


@dataclass(frozen=True)
class FigureExportSpec:
    stem: str
    directory_parts: tuple[str, ...] = ("reports", "figures")


@dataclass
class TableSpec:
    file_stem: str
    title: str
    columns: list[ColumnSpec]
    sections: list[TableSection]
    caption: str | None = None
    description: str | None = None
    rules: list[CellFormatRule] = field(default_factory=list)
    empty_message: str = "No rows available."
    markdown_two_row_ci: bool = False
    html_inline_ci: bool = False
    include_section_column_in_csv: bool = True


_HTML_STYLE = """
body {
  font-family: "Times New Roman", Times, serif;
  margin: 22px;
  color: #1f2933;
  background: #ffffff;
}
.table-title {
  font-size: 1.18rem;
  font-weight: 700;
  margin-bottom: 0.45rem;
}
.table-caption {
  margin-bottom: 0.8rem;
  color: #465362;
  max-width: 980px;
}
table.manuscript-table {
  border-collapse: collapse;
  width: 100%;
  max-width: 1200px;
  border: 1px solid #8b96a3;
  background: #ffffff;
}
table.manuscript-table th,
table.manuscript-table td {
  border: 1px solid #aeb8c2;
  padding: 8px 10px;
  vertical-align: top;
}
table.manuscript-table thead th {
  background: linear-gradient(180deg, #f2f5f8 0%, #e6edf4 100%);
  font-weight: 700;
}
tr.section-row td {
  background: linear-gradient(90deg, #dfe6ee 0%, #f1f5f9 100%);
  font-weight: 700;
  text-align: left;
}
tr.ci-row td {
  color: #5a6878;
  font-size: 0.95em;
}
td.align-left, th.align-left {
  text-align: left;
}
td.align-center, th.align-center {
  text-align: center;
}
td.align-right, th.align-right {
  text-align: right;
}
.ci-text {
  display: block;
  margin-top: 3px;
  color: #5a6878;
  font-size: 0.92em;
}
.best-cell {
  font-weight: 700;
  color: #0f2232;
}
.empty-state {
  color: #5a6878;
}
""".strip()


def _column_display_key(column: ColumnSpec) -> str:
    return column.display_key or column.key


def _column_csv_key(column: ColumnSpec) -> str:
    return column.csv_key or column.key


def _css_align(align: str) -> str:
    return f"align-{align}"


def _interpolate_rgb(start: tuple[int, int, int], end: tuple[int, int, int], weight: float) -> tuple[int, int, int]:
    clipped = min(1.0, max(0.0, float(weight)))
    return tuple(int(round(start[idx] + (end[idx] - start[idx]) * clipped)) for idx in range(3))


def _rgb_css(rgb: tuple[int, int, int]) -> str:
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def _mix_with_white(base: tuple[int, int, int], weight: float) -> tuple[int, int, int]:
    return _interpolate_rgb((255, 255, 255), base, weight)


def _rule_matches(rule: CellFormatRule, display_row: pd.Series, csv_row: pd.Series | None, section_csv: pd.DataFrame) -> bool:
    if rule.predicate is not None:
        return bool(rule.predicate(display_row, csv_row))
    if csv_row is None:
        return False
    metric_key = rule.column_key
    if metric_key not in section_csv.columns or metric_key not in csv_row.index:
        return False
    values = pd.to_numeric(section_csv[metric_key], errors="coerce")
    finite_values = values[np.isfinite(values.to_numpy(dtype=float))]
    if finite_values.empty:
        return False
    candidate = pd.to_numeric(pd.Series([csv_row[metric_key]]), errors="coerce").iloc[0]
    if not np.isfinite(candidate):
        return False
    target = finite_values.max() if rule.mode == "max" else finite_values.min()
    return bool(np.isclose(candidate, target, atol=1e-12))


def _cell_gradient_style(column: ColumnSpec, section_csv: pd.DataFrame, csv_row: pd.Series | None) -> str | None:
    if not column.gradient or csv_row is None:
        return None
    metric_key = _column_csv_key(column)
    if metric_key not in section_csv.columns or metric_key not in csv_row.index:
        return None
    values = pd.to_numeric(section_csv[metric_key], errors="coerce")
    finite_values = values[np.isfinite(values.to_numpy(dtype=float))]
    if finite_values.empty:
        return None
    candidate = pd.to_numeric(pd.Series([csv_row[metric_key]]), errors="coerce").iloc[0]
    if not np.isfinite(candidate):
        return None
    floor = float(finite_values.min())
    ceiling = float(finite_values.max())
    if np.isclose(floor, ceiling):
        normalized = 0.5
    else:
        normalized = float((candidate - floor) / (ceiling - floor))
    base_rgb = _interpolate_rgb((227, 236, 245), (183, 205, 225), normalized)
    top_rgb = _mix_with_white(base_rgb, 0.38)
    bottom_rgb = _mix_with_white(base_rgb, 0.72)
    return f"background: linear-gradient(180deg, {_rgb_css(top_rgb)} 0%, {_rgb_css(bottom_rgb)} 100%);"


def _cell_classes(
    column: ColumnSpec,
    spec: TableSpec,
    section_csv: pd.DataFrame,
    display_row: pd.Series,
    csv_row: pd.Series | None,
) -> list[str]:
    classes = [_css_align(column.align)]
    for rule in spec.rules:
        if rule.column_key != _column_csv_key(column):
            continue
        if _rule_matches(rule, display_row, csv_row, section_csv):
            classes.append(rule.css_class)
    return classes


def _cell_style(column: ColumnSpec, section_csv: pd.DataFrame, csv_row: pd.Series | None) -> str | None:
    return _cell_gradient_style(column, section_csv, csv_row)


def _wrap_markdown(value: str, classes: list[str], spec: TableSpec, column: ColumnSpec) -> str:
    wrapped = value
    if "best-cell" in classes:
        for rule in spec.rules:
            if rule.column_key == _column_csv_key(column) and rule.css_class == "best-cell":
                wrapped = f"{rule.markdown_wrapper[0]}{value}{rule.markdown_wrapper[1]}"
                break
    return wrapped


def _render_html(spec: TableSpec) -> str:
    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '  <meta charset="UTF-8">',
        f"  <title>{escape(spec.title)}</title>",
        "  <style>",
        _HTML_STYLE,
        "  </style>",
        "</head>",
        "<body>",
        f'  <div class="table-title">{escape(spec.title)}</div>',
    ]
    if spec.caption:
        lines.append(f'  <div class="table-caption">{escape(spec.caption)}</div>')
    if spec.description:
        lines.append(f'  <div class="table-caption">{escape(spec.description)}</div>')
    if not spec.sections:
        lines.append(f'  <div class="empty-state">{escape(spec.empty_message)}</div>')
    else:
        lines.append('  <table class="manuscript-table">')
        lines.append("    <thead>")
        header_cells = "".join(
            f'<th class="{_css_align(column.align)}">{escape(column.label)}</th>'
            for column in spec.columns
        )
        lines.append(f"      <tr>{header_cells}</tr>")
        lines.append("    </thead>")
        lines.append("    <tbody>")
        for section in spec.sections:
            if section.title:
                lines.append(
                    f'      <tr class="section-row"><td colspan="{len(spec.columns)}">{escape(section.title)}</td></tr>'
                )
            display_df = section.display_df.reset_index(drop=True)
            csv_df = (section.csv_df if section.csv_df is not None else section.display_df).reset_index(drop=True)
            if display_df.empty:
                lines.append(
                    f'      <tr><td colspan="{len(spec.columns)}" class="empty-state">{escape(spec.empty_message)}</td></tr>'
                )
                continue
            for row_idx in range(len(display_df)):
                display_row = display_df.iloc[row_idx]
                csv_row = csv_df.iloc[row_idx] if row_idx < len(csv_df) else None
                row_class = "ci-row" if str(display_row.get("row_kind", "")) == "ci" else ""
                cells: list[str] = []
                for column in spec.columns:
                    display_key = _column_display_key(column)
                    value = "" if display_key not in display_row.index or pd.isna(display_row[display_key]) else str(display_row[display_key])
                    classes = _cell_classes(column, spec, csv_df, display_row, csv_row)
                    inline_style = _cell_style(column, csv_df, csv_row)
                    html_value = escape(value)
                    if spec.html_inline_ci and column.ci_display_key and column.ci_display_key in display_row.index:
                        ci_value = display_row[column.ci_display_key]
                        if pd.notna(ci_value) and str(ci_value):
                            html_value = f"{html_value}<span class=\"ci-text\">{escape(str(ci_value))}</span>"
                    style_attr = f' style="{inline_style}"' if inline_style else ""
                    cells.append(f'<td class="{" ".join(classes)}"{style_attr}>{html_value}</td>')
                lines.append(f'      <tr class="{row_class}">{"".join(cells)}</tr>')
        lines.append("    </tbody>")
        lines.append("  </table>")
    lines.extend(["</body>", "</html>", ""])
    return "\n".join(lines)


def _render_markdown(spec: TableSpec) -> str:
    lines = [f"# {spec.title}", ""]
    if spec.caption:
        lines.append(spec.caption)
        lines.append("")
    if spec.description:
        lines.append(spec.description)
        lines.append("")
    if not spec.sections:
        lines.append(spec.empty_message)
        lines.append("")
        return "\n".join(lines)

    header = "| " + " | ".join(column.label for column in spec.columns) + " |"
    divider = "|" + "|".join("---" for _ in spec.columns) + "|"
    lines.extend([header, divider])
    for section in spec.sections:
        if section.title:
            lines.append("| " + " | ".join([f"**{section.title}**"] + [""] * (len(spec.columns) - 1)) + " |")
        display_df = section.display_df.reset_index(drop=True)
        csv_df = (section.csv_df if section.csv_df is not None else section.display_df).reset_index(drop=True)
        if display_df.empty:
            lines.append("| " + " | ".join([spec.empty_message] + [""] * (len(spec.columns) - 1)) + " |")
            continue
        for row_idx in range(len(display_df)):
            display_row = display_df.iloc[row_idx]
            csv_row = csv_df.iloc[row_idx] if row_idx < len(csv_df) else None
            value_cells: list[str] = []
            ci_cells: list[str] = []
            has_ci = False
            for col_idx, column in enumerate(spec.columns):
                display_key = _column_display_key(column)
                value = "" if display_key not in display_row.index or pd.isna(display_row[display_key]) else str(display_row[display_key])
                classes = _cell_classes(column, spec, csv_df, display_row, csv_row)
                value_cells.append(_wrap_markdown(value, classes, spec, column))
                ci_value = ""
                if column.ci_display_key and column.ci_display_key in display_row.index and pd.notna(display_row[column.ci_display_key]):
                    ci_value = str(display_row[column.ci_display_key])
                    has_ci = has_ci or bool(ci_value)
                ci_cells.append("" if col_idx == 0 else ci_value)
            lines.append("| " + " | ".join(value_cells) + " |")
            if spec.markdown_two_row_ci and has_ci:
                lines.append("| " + " | ".join(ci_cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _combined_csv_frame(spec: TableSpec) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for section in spec.sections:
        frame = (section.csv_df if section.csv_df is not None else section.display_df).copy()
        if spec.include_section_column_in_csv and section.title and "section" not in frame.columns:
            frame.insert(0, "section", section.title)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[_column_csv_key(column) for column in spec.columns])
    return pd.concat(frames, ignore_index=True)


def write_table_outputs(artifacts: ArtifactManager, spec: TableSpec, config: dict | None = None) -> list[Path]:
    outputs: list[Path] = []
    for fmt in report_table_formats(config):
        path = artifacts.resolve("reports", "tables", f"{spec.file_stem}.{fmt}")
        if fmt == "html":
            path.write_text(_render_html(spec), encoding="utf-8")
        elif fmt == "md":
            path.write_text(_render_markdown(spec), encoding="utf-8")
        elif fmt == "csv":
            _combined_csv_frame(spec).to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported table format: {fmt}")
        outputs.append(path)
    return outputs


def save_figure_variants(fig: object, artifacts: ArtifactManager, spec: FigureExportSpec, config: dict | None = None) -> list[Path]:
    outputs: list[Path] = []
    if not hasattr(fig, "savefig"):
        raise TypeError("save_figure_variants expects a matplotlib Figure-like object.")
    directory_parts = spec.directory_parts
    if tuple(directory_parts) == ("reports", "figures"):
        directory_parts = report_primary_figure_directory_parts(config)
    for fmt in report_figure_formats(config):
        path = artifacts.resolve(*directory_parts, f"{spec.stem}.{fmt}")
        save_kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            save_kwargs["dpi"] = report_figure_png_dpi(config)
        fig.savefig(path, **save_kwargs)
        outputs.append(path)
    return outputs
