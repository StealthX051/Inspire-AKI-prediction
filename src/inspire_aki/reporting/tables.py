from __future__ import annotations

from typing import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score

from inspire_aki.config import active_outcome_config, active_target_column
from inspire_aki.evaluation.thresholds import find_optimal_fbeta_threshold
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.registry import DATASET_REGIMES, MANUSCRIPT_MODEL_ORDER, model_display_name
from inspire_aki.reporting.rendering import CellFormatRule, ColumnSpec, TableSection, TableSpec, write_table_outputs
from inspire_aki.runtime import build_stage_runtime_plan, thread_limited_context

_DATASET_TITLES = {
    "preop": "Preop Data",
    "intraop": "Intraop Data",
    "combined": "Combined Data",
}

_PERFORMANCE_COLUMNS = [
    ColumnSpec("model_name", "Model", align="left"),
    ColumnSpec("auroc", "AUROC", ci_display_key="auroc_ci_display", gradient=True),
    ColumnSpec("auprc", "AUPRC", ci_display_key="auprc_ci_display", gradient=True),
    ColumnSpec("sensitivity", "Sensitivity", ci_display_key="sensitivity_ci_display", gradient=True),
    ColumnSpec("specificity", "Specificity", ci_display_key="specificity_ci_display", gradient=True),
    ColumnSpec("precision", "Precision", ci_display_key="precision_ci_display", gradient=True),
    ColumnSpec("f_score", "F-score", ci_display_key="f_score_ci_display", gradient=True),
    ColumnSpec("balanced_accuracy", "Balanced Accuracy", ci_display_key="balanced_accuracy_ci_display", gradient=True),
]

_PERFORMANCE_METRICS = ("auroc", "auprc", "sensitivity", "specificity", "precision", "f_score", "balanced_accuracy")
_MODEL_ORDER_LOOKUP = {model_key: idx for idx, model_key in enumerate(MANUSCRIPT_MODEL_ORDER)}
_DATASET_MODEL_ORDER = {
    "preop": ("asa_rule", "autogluon", "xgb", "knn", "log_reg", "mlp", "rf", "svm"),
    "intraop": ("autogluon", "xgb", "knn", "log_reg", "lstm_only", "mlp", "rf", "svm"),
    "combined": ("autogluon", "xgb", "knn", "log_reg", "mlp", "hybrid", "rf", "svm"),
}
_DEPARTMENT_LABELS = {
    "UR": "Urology",
    "RO": "Radiation Oncology",
    "RAD": "Radiology",
    "PS": "Plastic Surgery",
    "PED": "Pediatric Surgery",
    "OT": "Orthopedic Surgery",
    "OS": "Oral Surgery",
    "OL": "Otorhinolaryngology",
    "OG": "Obstetrics and Gynecology",
    "NS": "Neurosurgery",
    "IM": "Internal Medicine",
    "GS": "General Surgery",
    "EM": "Emergency Medicine",
    "DM": "Dermatology",
    "CTS": "Cardiothoracic Surgery",
    "AN": "Anesthesiology",
}


def _format_mean_sd(series: pd.Series) -> str:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return "N/A"
    return f"{clean.mean():.2f} +/- {clean.std(ddof=1):.2f}" if len(clean) > 1 else f"{clean.mean():.2f} +/- 0.00"


def _format_count_pct(count: float, total: int) -> str:
    if total <= 0:
        return "0 (0.00%)"
    return f"{int(count)} ({(count / total) * 100:.2f}%)"


def _female_mask(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    if clean.empty:
        return pd.Series(False, index=series.index)

    if clean.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        return series.astype("boolean").eq(False).fillna(False)

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == clean.shape[0] and set(numeric.dropna().unique()).issubset({0, 1}):
        return numeric.eq(0).fillna(False)

    normalized = series.astype(str).str.strip().str.upper()
    female_tokens = {"F", "FEMALE", "FALSE", "0"}
    male_tokens = {"M", "MALE", "TRUE", "1"}
    if normalized[series.notna()].isin(female_tokens | male_tokens).all():
        return normalized.isin(female_tokens)
    return normalized.isin({"F", "FEMALE"})


def _department_rows(cohort_df: pd.DataFrame, total: int) -> list[dict[str, object]]:
    dept_counts: dict[str, float] = {}
    dept_columns = sorted(column for column in cohort_df.columns if column.startswith("department_"))
    for column in dept_columns:
        normalized_column = column.removesuffix("_preop")
        abbreviation = normalized_column.removeprefix("department_").upper()
        count = float(pd.to_numeric(cohort_df[column], errors="coerce").fillna(0).astype(int).sum())
        if abbreviation in dept_counts:
            dept_counts[abbreviation] = max(dept_counts[abbreviation], count)
        else:
            dept_counts[abbreviation] = count

    return [
        {
            "characteristic": _DEPARTMENT_LABELS.get(abbreviation, abbreviation.replace("_", " ")),
            "finding": _format_count_pct(count, total),
        }
        for abbreviation, count in sorted(dept_counts.items())
    ]


def _model_sort_key(model_key: str) -> tuple[int, str]:
    return (_MODEL_ORDER_LOOKUP.get(model_key, len(_MODEL_ORDER_LOOKUP)), model_display_name(model_key))


def _dataset_model_sort_key(dataset_regime: str, model_key: str) -> tuple[int, tuple[int, str]]:
    preferred_order = _DATASET_MODEL_ORDER.get(dataset_regime, ())
    if model_key in preferred_order:
        return (0, (preferred_order.index(model_key), model_display_name(model_key)))
    return (1, _model_sort_key(model_key))


def _safe_trapezoidal_auprc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return float(auc(recall, precision))


def _performance_prediction_frame(predictions_df: pd.DataFrame, *, prob_col: str, config: dict, use_existing_threshold: bool) -> pd.DataFrame:
    frame = predictions_df[predictions_df["split_name"].astype(str) == "test"].copy()
    if frame.empty:
        return frame
    calibrated_probs = frame[prob_col].fillna(frame["y_prob_raw"]).astype(float)
    frame["report_y_prob"] = calibrated_probs
    groups: list[pd.DataFrame] = []
    for _, group_df in frame.groupby(["dataset_regime", "population_id", "model_key"], sort=False):
        group = group_df.copy()
        if use_existing_threshold and group["threshold"].notna().any():
            threshold = float(group["threshold"].dropna().iloc[0])
        else:
            threshold = find_optimal_fbeta_threshold(
                group["y_true"].astype(int).to_numpy(),
                group["report_y_prob"].astype(float).to_numpy(),
                beta=2.0,
                threshold_min=config["calibration"]["threshold_min"],
                threshold_max=config["calibration"]["threshold_max"],
                steps=config["calibration"]["threshold_steps"],
            )
        group["report_threshold"] = threshold
        group["report_y_pred"] = (group["report_y_prob"] >= threshold).astype(int)
        groups.append(group)
    return pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()


def _fold_metric_worker(keys: tuple, group_df: pd.DataFrame, nested_blas_threads: int) -> dict[str, object]:
    with thread_limited_context(nested_blas_threads):
        y_true = group_df["y_true"].astype(int).to_numpy()
        y_prob = group_df["report_y_prob"].astype(float).to_numpy()
        y_pred = group_df["report_y_pred"].astype(int).to_numpy()
        if len(np.unique(y_true)) < 2:
            auroc = np.nan
            balanced_accuracy = np.nan
        else:
            auroc = float(roc_auc_score(y_true, y_prob))
            balanced_accuracy = float(np.mean([
                np.mean(y_pred[y_true == 0] == 0) if np.any(y_true == 0) else np.nan,
                np.mean(y_pred[y_true == 1] == 1) if np.any(y_true == 1) else np.nan,
            ]))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "dataset_regime": keys[0],
            "population_id": keys[1],
            "model_key": keys[2],
            "repeat_id": int(keys[3]),
            "fold_id": int(keys[4]),
            "threshold": float(group_df["report_threshold"].iloc[0]),
            "auroc": auroc,
            "auprc": _safe_trapezoidal_auprc(y_true, y_prob),
            "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
            "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "f_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "balanced_accuracy": balanced_accuracy,
        }


def _manuscript_ci(values: Iterable[float]) -> tuple[float, float | None, float | None]:
    clean = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if clean.size == 0:
        return np.nan, None, None
    mean = float(np.mean(clean))
    if clean.size <= 1:
        return mean, None, None
    sem = stats.sem(clean, nan_policy="omit")
    if not np.isfinite(sem) or sem <= 0:
        return mean, None, None
    lower, upper = stats.t.interval(0.95, df=clean.size - 1, loc=mean, scale=sem)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return mean, None, None
    return mean, float(lower), float(upper)


def _format_ci(lower: float | None, upper: float | None) -> str:
    if lower is None or upper is None:
        return "N/A"
    return f"({lower:.3f}, {upper:.3f})"


def _bootstrap_ci_worker(
    keys: tuple[str, str, str],
    group_df: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_state: int,
) -> tuple[tuple[str, str, str], dict[str, tuple[float | None, float | None]]]:
    y_true_base = group_df["y_true"].astype(int).to_numpy()
    y_prob_base = group_df["report_y_prob"].astype(float).to_numpy()
    threshold = float(group_df["report_threshold"].iloc[0])
    n_samples = len(group_df)
    rng = np.random.default_rng(random_state)
    values_by_metric: dict[str, list[float]] = {metric: [] for metric in _PERFORMANCE_METRICS}

    for _ in range(n_bootstrap):
        bootstrap_idx = rng.integers(0, n_samples, n_samples)
        y_true = y_true_base[bootstrap_idx]
        if len(np.unique(y_true)) < 2:
            continue
        y_prob = y_prob_base[bootstrap_idx]
        y_pred = (y_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        values_by_metric["auroc"].append(float(roc_auc_score(y_true, y_prob)))
        values_by_metric["auprc"].append(_safe_trapezoidal_auprc(y_true, y_prob))
        values_by_metric["sensitivity"].append(float(recall_score(y_true, y_pred, zero_division=0)))
        values_by_metric["specificity"].append(float(tn / (tn + fp)) if (tn + fp) else np.nan)
        values_by_metric["precision"].append(float(precision_score(y_true, y_pred, zero_division=0)))
        values_by_metric["f_score"].append(float(f1_score(y_true, y_pred, zero_division=0)))
        values_by_metric["balanced_accuracy"].append(
            float(np.mean([
                np.mean(y_pred[y_true == 0] == 0) if np.any(y_true == 0) else np.nan,
                np.mean(y_pred[y_true == 1] == 1) if np.any(y_true == 1) else np.nan,
            ]))
        )

    ci_lookup: dict[str, tuple[float | None, float | None]] = {}
    for metric, values in values_by_metric.items():
        clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
        if clean.size == 0:
            ci_lookup[metric] = (None, None)
            continue
        ci_lookup[metric] = (
            float(np.percentile(clean, 2.5)),
            float(np.percentile(clean, 97.5)),
        )
    return keys, ci_lookup


def _bootstrap_ci_lookup(
    report_df: pd.DataFrame,
    *,
    config: dict,
    runtime_plan,
) -> dict[tuple[str, str, str], dict[str, tuple[float | None, float | None]]]:
    lookups: dict[tuple[str, str, str], dict[str, tuple[float | None, float | None]]] = {}
    if report_df.empty or int(config["evaluation"]["bootstrap_reps"]) < 2:
        return lookups

    groups = [
        ((str(keys[0]), str(keys[1]), str(keys[2])), group_df.copy())
        for keys, group_df in report_df.groupby(["dataset_regime", "population_id", "model_key"], sort=False)
        if len(group_df[["repeat_id", "fold_id"]].drop_duplicates()) <= 1
    ]
    if not groups:
        return lookups

    results = Parallel(n_jobs=max(1, min(runtime_plan.bootstrap_workers, len(groups))), backend="loky")(
        delayed(_bootstrap_ci_worker)(
            keys,
            group_df,
            n_bootstrap=int(config["evaluation"]["bootstrap_reps"]),
            random_state=int(config["splits"]["random_state"]) + idx,
        )
        for idx, (keys, group_df) in enumerate(groups)
    )
    for keys, ci_lookup in results:
        lookups[keys] = ci_lookup
    return lookups


def _performance_summary_frame(
    predictions_df: pd.DataFrame,
    *,
    prob_col: str,
    config: dict,
    use_existing_threshold: bool,
    bootstrap_ci_lookup: dict[tuple[str, str, str], dict[str, tuple[float | None, float | None]]] | None = None,
) -> pd.DataFrame:
    report_df = _performance_prediction_frame(predictions_df, prob_col=prob_col, config=config, use_existing_threshold=use_existing_threshold)
    if report_df.empty:
        return pd.DataFrame()
    groups = [
        (keys, group_df.copy())
        for keys, group_df in report_df.groupby(["dataset_regime", "population_id", "model_key", "repeat_id", "fold_id"], sort=False)
    ]
    runtime_plan = build_stage_runtime_plan(config, "report_tables", {"group_count": len(groups)})
    fold_rows = Parallel(n_jobs=max(1, runtime_plan.report_workers), backend="loky")(
        delayed(_fold_metric_worker)(keys, group_df, runtime_plan.nested_blas_threads)
        for keys, group_df in groups
    )
    fold_df = pd.DataFrame(fold_rows)
    bootstrap_ci = bootstrap_ci_lookup or _bootstrap_ci_lookup(report_df, config=config, runtime_plan=runtime_plan)
    rows: list[dict[str, object]] = []
    for (dataset_regime, population_id, model_key), group_df in fold_df.groupby(["dataset_regime", "population_id", "model_key"], sort=False):
        row: dict[str, object] = {
            "dataset_regime": dataset_regime,
            "population_id": population_id,
            "model_key": model_key,
            "model_name": model_display_name(model_key),
        }
        bootstrap_lookup = bootstrap_ci.get((str(dataset_regime), str(population_id), str(model_key)), {})
        for metric in _PERFORMANCE_METRICS:
            mean, lower, upper = _manuscript_ci(group_df[metric])
            if len(group_df) <= 1 and metric in bootstrap_lookup:
                lower, upper = bootstrap_lookup[metric]
            row[metric] = mean
            row[f"{metric}_ci_lower"] = np.nan if lower is None else lower
            row[f"{metric}_ci_upper"] = np.nan if upper is None else upper
            row[f"{metric}_display"] = "N/A" if not np.isfinite(mean) else f"{mean:.3f}"
            row[f"{metric}_ci_display"] = _format_ci(lower, upper)
        rows.append(row)
    return pd.DataFrame(rows)


def _performance_table_spec(summary_df: pd.DataFrame, *, file_stem: str, title: str, caption: str) -> TableSpec:
    sections: list[TableSection] = []
    if not summary_df.empty:
        ordered = summary_df.loc[
            ~((summary_df["model_key"].astype(str) == "asa_rule") & (summary_df["dataset_regime"].astype(str) != "preop"))
        ].copy()
        for dataset_regime in DATASET_REGIMES:
            section_df = ordered[ordered["dataset_regime"] == dataset_regime].copy()
            if section_df.empty:
                continue
            section_df["model_order"] = section_df["model_key"].map(
                lambda key, dataset_regime=dataset_regime: _dataset_model_sort_key(dataset_regime, str(key))
            )
            section_df = section_df.sort_values("model_order", kind="stable").reset_index(drop=True)
            display_cols = {
                "model_name": section_df["model_name"],
            }
            for metric in _PERFORMANCE_METRICS:
                display_cols[metric] = section_df[f"{metric}_display"]
                display_cols[f"{metric}_ci_display"] = section_df[f"{metric}_ci_display"]
            display_df = pd.DataFrame(display_cols)
            csv_df = section_df.drop(columns=["model_order"])
            sections.append(TableSection(title=_DATASET_TITLES[dataset_regime], display_df=display_df, csv_df=csv_df))

    return TableSpec(
        file_stem=file_stem,
        title=title,
        caption=caption,
        columns=_PERFORMANCE_COLUMNS,
        sections=sections,
        rules=[CellFormatRule(metric, mode="max") for metric in _PERFORMANCE_METRICS],
        markdown_two_row_ci=True,
        html_inline_ci=True,
    )


def _cohort_sections(merged_df: pd.DataFrame, config: dict) -> list[TableSection]:
    if merged_df.empty:
        return []
    cohort_df = merged_df.sort_values("op_id", kind="stable")
    if "subject_id" in cohort_df.columns:
        cohort_df = cohort_df.drop_duplicates(subset=["subject_id"], keep="last")
    total = int(len(cohort_df))

    summary_rows: list[dict[str, object]] = []
    numeric_rows = [
        ("Age, y, mean (SD)", "age"),
        ("Weight, kg, mean (SD)", "weight"),
        ("Height, cm, mean (SD)", "height"),
        ("BMI, kg/m^2, mean (SD)", "BMI"),
        ("BSA, m^2, mean (SD)", "BSA"),
        ("ASA, mean (SD)", "asa"),
        ("Number of Preexisting Cardiac Diagnoses, mean (SD)", "num_card_events"),
        ("Booking Case Length, min, mean (SD)", "booking_case_length"),
    ]
    for label, column in numeric_rows:
        if column in cohort_df.columns:
            summary_rows.append({"characteristic": label, "finding": _format_mean_sd(cohort_df[column])})
    if "sex" in cohort_df.columns:
        female_mask = _female_mask(cohort_df["sex"])
        summary_rows.append({"characteristic": "Female sex, n (%)", "finding": _format_count_pct(float(female_mask.sum()), total)})
    sections = [TableSection(title=None, display_df=pd.DataFrame(summary_rows), csv_df=pd.DataFrame(summary_rows))]

    if "asa" in cohort_df.columns:
        asa_rows = []
        for value, count in cohort_df["asa"].dropna().astype(int).value_counts().sort_index().items():
            asa_rows.append({"characteristic": str(value), "finding": _format_count_pct(float(count), total)})
        sections.append(TableSection(title="ASA classification, n (%)", display_df=pd.DataFrame(asa_rows), csv_df=pd.DataFrame(asa_rows)))

    target_column = active_target_column(config)
    if target_column in cohort_df.columns:
        outcome_cfg = active_outcome_config(config)
        positive_count = float(cohort_df[target_column].astype(int).sum())
        outcome_rows = pd.DataFrame(
            [
                {
                    "characteristic": f"{outcome_cfg['display_name']}, n (%)",
                    "finding": _format_count_pct(positive_count, total),
                }
            ]
        )
        sections.append(TableSection(title=None, display_df=outcome_rows, csv_df=outcome_rows))

    dept_columns = [column for column in cohort_df.columns if column.startswith("department_")]
    if dept_columns:
        dept_rows = _department_rows(cohort_df, total)
        sections.append(
            TableSection(
                title="Department Surgery type, n (%)",
                display_df=pd.DataFrame(dept_rows),
                csv_df=pd.DataFrame(dept_rows),
            )
        )
    elif "department" in cohort_df.columns:
        dept_counts = cohort_df["department"].astype(str).value_counts().sort_index()
        dept_rows = [
            {"characteristic": _DEPARTMENT_LABELS.get(label.upper(), label), "finding": _format_count_pct(float(count), total)}
            for label, count in dept_counts.items()
        ]
        sections.append(TableSection(title="Department Surgery type, n (%)", display_df=pd.DataFrame(dept_rows), csv_df=pd.DataFrame(dept_rows)))
    return sections


def _labels_artifact_path(artifacts: ArtifactManager) -> Path:
    candidates = [
        artifacts.paths.artifact_path("cohort", "labels.csv"),
        artifacts.paths.artifact_path("cohort", "aki_labels.csv"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _build_fill_rate_frame(cohort_df: pd.DataFrame, artifacts: ArtifactManager) -> pd.DataFrame:
    isna_cols = [column for column in cohort_df.columns if column.endswith("_isna")]
    if isna_cols:
        rows = []
        for column in isna_cols:
            base_name = column.removesuffix("_isna")
            for prefix in ("mean_", "sum_", "max_", "min_"):
                if base_name.startswith(prefix):
                    base_name = base_name.removeprefix(prefix)
            fill_rate = 1.0 - pd.to_numeric(cohort_df[column], errors="coerce").fillna(1.0).mean()
            rows.append({"variable": base_name, "fill_rate": float(fill_rate)})
        fill_df = pd.DataFrame(rows).groupby("variable", as_index=False)["fill_rate"].mean()
    else:
        fill_path = artifacts.paths.artifact_path("features", "fill_rates.csv")
        if not fill_path.exists():
            return pd.DataFrame()
        fill_df = pd.read_csv(fill_path).rename(columns={"feature": "variable"})
    fill_df["fill_rate_pct"] = fill_df["fill_rate"] * 100.0
    return fill_df.sort_values(["fill_rate_pct", "variable"], ascending=[False, True], kind="stable").reset_index(drop=True)


def generate_table_outputs(artifacts: ArtifactManager) -> list[Path]:
    config = artifacts.config
    outputs: list[Path] = []

    predictions_path = artifacts.paths.artifact_path("predictions", "raw_predictions.parquet")
    calibrated_path = artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet")
    if predictions_path.exists():
        raw_predictions = pd.read_parquet(predictions_path)
        raw_summary = _performance_summary_frame(
            raw_predictions,
            prob_col="y_prob_raw",
            config=config,
            use_existing_threshold=False,
        )
        outputs.extend(
            write_table_outputs(
                artifacts,
                _performance_table_spec(
                    raw_summary,
                    file_stem="performance_table",
                    title="Performance Metrics",
                    caption="Legacy-style fold/run aggregation over corrected refactor predictions.",
                ),
                config,
            )
        )
    if calibrated_path.exists():
        calibrated_predictions = pd.read_parquet(calibrated_path)
        calibrated_summary = _performance_summary_frame(
            calibrated_predictions,
            prob_col="y_prob_calibrated",
            config=config,
            use_existing_threshold=True,
        )
        outputs.extend(
            write_table_outputs(
                artifacts,
                _performance_table_spec(
                    calibrated_summary,
                    file_stem="performance_table_calibrated",
                    title="Calibrated Performance Metrics",
                    caption="Legacy-style calibrated manuscript table using grouped isotonic refactor outputs.",
                ),
                config,
            )
        )

    cohort_path = artifacts.paths.artifact_path("datasets", "tabular", "tabular_combined_unnormalized.csv")
    preop_path = artifacts.paths.artifact_path("features", "preop", "preop_features.csv")
    labels_path = _labels_artifact_path(artifacts)
    if labels_path.exists() and (cohort_path.exists() or preop_path.exists()):
        if cohort_path.exists():
            merged_df = pd.read_csv(cohort_path)
        else:
            merged_df = pd.read_csv(preop_path)
        merged_df = merged_df.merge(pd.read_csv(labels_path), on="op_id", how="inner")
        cohort_spec = TableSpec(
            file_stem="cohort_characteristics",
            title="Table 1. Characteristics of Cohort",
            caption="Manuscript-ready descriptive summary.",
            columns=[ColumnSpec("characteristic", "Characteristic", align="left"), ColumnSpec("finding", "Finding", align="left")],
            sections=_cohort_sections(merged_df, config),
            include_section_column_in_csv=False,
        )
        outputs.extend(write_table_outputs(artifacts, cohort_spec, config))
        fill_df = _build_fill_rate_frame(merged_df, artifacts)
        if not fill_df.empty:
            fill_display = fill_df.assign(fill_rate_pct=fill_df["fill_rate_pct"].map(lambda value: f"{value:.2f}%"))
            fill_spec = TableSpec(
                file_stem="fill_rate_table",
                title="Table 2. Variable Fill Rates",
                caption="Variable completeness after preprocessing, consolidated across missingness indicators where available.",
                columns=[ColumnSpec("variable", "Variable", align="left"), ColumnSpec("fill_rate_pct", "Fill Rate (%)")],
                sections=[TableSection(title=None, display_df=fill_display[["variable", "fill_rate_pct"]], csv_df=fill_df)],
                rules=[CellFormatRule("fill_rate_pct", mode="max")],
                include_section_column_in_csv=False,
            )
            outputs.extend(write_table_outputs(artifacts, fill_spec, config))
    elif cohort_path.exists() != labels_path.exists():
        missing = cohort_path if not cohort_path.exists() else labels_path
        raise FileNotFoundError(f"Required cohort reporting input was not found: {missing}")

    return outputs
