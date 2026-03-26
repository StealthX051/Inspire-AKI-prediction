from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.registry import model_colors, model_display_name, model_safe_name
from inspire_aki.reporting.rendering import FigureExportSpec, save_figure_variants
from inspire_aki.runtime import build_stage_runtime_plan

_DATASET_LABELS = {"preop": "Preop", "intraop": "Intraop", "combined": "Combined"}


def _probability_column(df: pd.DataFrame) -> pd.Series:
    return df["y_prob_calibrated"].fillna(df["y_prob_raw"]).astype(float)


def _roc_curve_stats(model_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    mean_fpr = np.linspace(0.0, 1.0, 101)
    interpolated = []
    aucs: list[float] = []
    for _, fold_df in model_df.groupby(["repeat_id", "fold_id"], sort=False):
        y_true = fold_df["y_true"].astype(int).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        y_prob = _probability_column(fold_df).to_numpy()
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        interpolated.append(interp_tpr)
        aucs.append(float(auc(fpr, tpr)))
    if not interpolated:
        raise ValueError("Unable to compute ROC statistics for a single-class fold set.")
    stacked = np.vstack(interpolated)
    return mean_fpr, stacked.mean(axis=0), stacked.std(axis=0), float(np.mean(aucs)), float(np.std(aucs, ddof=0))


def _pr_curve_stats(model_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    mean_recall = np.linspace(0.0, 1.0, 101)
    interpolated = []
    aucs: list[float] = []
    for _, fold_df in model_df.groupby(["repeat_id", "fold_id"], sort=False):
        y_true = fold_df["y_true"].astype(int).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        y_prob = _probability_column(fold_df).to_numpy()
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        order = np.argsort(recall)
        recall_sorted = recall[order]
        precision_sorted = precision[order]
        interpolated.append(np.interp(mean_recall, recall_sorted, precision_sorted))
        aucs.append(float(auc(recall, precision)))
    if not interpolated:
        raise ValueError("Unable to compute PR statistics for a single-class fold set.")
    stacked = np.vstack(interpolated)
    return mean_recall, stacked.mean(axis=0), stacked.std(axis=0), float(np.mean(aucs)), float(np.std(aucs, ddof=0))


def _sorted_model_groups(dataset_df: pd.DataFrame, metric_fn) -> list[tuple[str, pd.DataFrame, tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]]:
    rows = []
    for model_key, model_df in dataset_df.groupby("model_key", sort=False):
        try:
            stats_payload = metric_fn(model_df)
        except ValueError:
            continue
        rows.append((model_key, model_df, stats_payload))
    return sorted(rows, key=lambda item: item[2][3], reverse=True)


def _curve_figure(dataset_regime: str, dataset_df: pd.DataFrame, *, curve_kind: str):
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    metric_fn = _roc_curve_stats if curve_kind == "roc" else _pr_curve_stats
    sorted_groups = _sorted_model_groups(dataset_df, metric_fn)
    if curve_kind == "roc":
        x_label, y_label = "False Positive Rate", "True Positive Rate"
        title_prefix = "ROC Curves"
        reference_line = ([0, 1], [0, 1], {"linestyle": "--", "color": "#7d8793", "label": "Chance"})
        legend_prefix = "AUC"
    else:
        x_label, y_label = "Recall", "Precision"
        title_prefix = "Precision-Recall Curves"
        prevalence = float(dataset_df["y_true"].astype(int).mean()) if not dataset_df.empty else 0.0
        reference_line = ([0, 1], [prevalence, prevalence], {"linestyle": "--", "color": "#7d8793", "label": "Prevalence"})
        legend_prefix = "AUPRC"
    for model_key, _model_df, stats_payload in sorted_groups:
        x_axis, mean_curve, std_curve, mean_auc, std_auc = stats_payload
        color = model_colors.get(model_display_name(model_key))
        lower = np.clip(mean_curve - std_curve, 0.0, 1.0)
        upper = np.clip(mean_curve + std_curve, 0.0, 1.0)
        label = f"{model_display_name(model_key)} ({legend_prefix} = {mean_auc:.3f} +/- {std_auc:.3f})"
        ax.plot(x_axis, mean_curve, color=color, lw=2.0, label=label)
        ax.fill_between(x_axis, lower, upper, color=color, alpha=0.18)
    ax.plot(reference_line[0], reference_line[1], **reference_line[2])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title_prefix}: {_DATASET_LABELS.get(dataset_regime, dataset_regime.title())}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.22)
    return fig


def _calibration_figure(dataset_regime: str, dataset_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    plotted = False
    for model_key, model_df in dataset_df.groupby("model_key", sort=False):
        y_true = model_df["y_true"].astype(int).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        y_prob = _probability_column(model_df).to_numpy()
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=15, strategy="uniform")
        color = model_colors.get(model_display_name(model_key))
        ax.plot(mean_pred, frac_pos, marker="o", lw=1.8, label=model_display_name(model_key), color=color)
        plotted = True
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7d8793", label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Observed Fraction Positive")
    ax.set_title(f"Calibration Curves: {_DATASET_LABELS.get(dataset_regime, dataset_regime.title())}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.22)
    if not plotted:
        raise ValueError("No calibration curves could be computed for this dataset.")
    return fig


def _curve_outputs_for_dataset(dataset_regime: str, dataset_df: pd.DataFrame, config: dict, artifacts: ArtifactManager) -> list[Path]:
    outputs: list[Path] = []
    for curve_kind, stem in [("roc", f"roc_curves_{dataset_regime}"), ("pr", f"pr_curves_{dataset_regime}")]:
        fig = _curve_figure(dataset_regime, dataset_df, curve_kind=curve_kind)
        try:
            outputs.extend(save_figure_variants(fig, artifacts, FigureExportSpec(stem=stem), config))
        finally:
            plt.close(fig)
    fig = _calibration_figure(dataset_regime, dataset_df)
    try:
        outputs.extend(save_figure_variants(fig, artifacts, FigureExportSpec(stem=f"calibration_curves_{dataset_regime}"), config))
    finally:
        plt.close(fig)
    return outputs


def _dca_figure(dca_df: pd.DataFrame, *, title: str, color: str, show_ci: bool) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.plot(dca_df["threshold_prob"] * 100.0, dca_df["net_benefit_model"], color=color, lw=2.0, label="Model")
    if show_ci and {"ci_lower_95", "ci_upper_95"}.issubset(dca_df.columns):
        ax.fill_between(
            dca_df["threshold_prob"] * 100.0,
            dca_df["ci_lower_95"],
            dca_df["ci_upper_95"],
            color=color,
            alpha=0.18,
            label="95% Confidence Interval",
        )
    ax.plot(dca_df["threshold_prob"] * 100.0, dca_df["net_benefit_treat_all"], linestyle="--", color="#222222", label="Treat All")
    ax.plot(dca_df["threshold_prob"] * 100.0, dca_df["net_benefit_treat_none"], linestyle="-", color="#7d8793", label="Treat None")
    threshold_series = dca_df["threshold_optimal"].dropna() if "threshold_optimal" in dca_df.columns else pd.Series(dtype=float)
    if not threshold_series.empty:
        threshold = float(threshold_series.iloc[0])
        ax.axvline(threshold * 100.0, linestyle=":", color="#bf3b3b", lw=1.8, label=f"F2-Optimal Threshold (tau* = {threshold:.2f})")
    ax.set_xlabel("Risk Threshold Probability (%)")
    ax.set_ylabel("Net Benefit")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.22)
    return fig


def _dca_outputs_for_group(keys: tuple[str, str, str], dca_df: pd.DataFrame, config: dict, artifacts: ArtifactManager) -> list[Path]:
    dataset_regime, population_id, model_key = keys
    display_name = model_display_name(model_key)
    color = model_colors.get(display_name, "#2d6ba3")
    title = f"Decision-Curve Analysis: {display_name} ({_DATASET_LABELS.get(dataset_regime, dataset_regime.title())})"
    fig = _dca_figure(dca_df, title=title, color=color, show_ci="ci_lower_95" in dca_df.columns)
    try:
        stem = f"dca_curve_{dataset_regime}_{model_safe_name(model_key)}"
        if dca_df["population_id"].nunique() > 1 or population_id != dataset_regime:
            stem = f"{stem}_{population_id}"
        return save_figure_variants(fig, artifacts, FigureExportSpec(stem=stem), config)
    finally:
        plt.close(fig)


def _dca_comparison_figure(model_key: str, model_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    for dataset_regime, dataset_curve_df in model_df.groupby("dataset_regime", sort=False):
        label = _DATASET_LABELS.get(dataset_regime, dataset_regime.title())
        ax.plot(
            dataset_curve_df["threshold_prob"] * 100.0,
            dataset_curve_df["net_benefit_model"],
            lw=2.0,
            label=label,
        )
    ax.set_xlabel("Risk Threshold Probability (%)")
    ax.set_ylabel("Net Benefit")
    ax.set_title(f"Decision-Curve Comparison Across Data Sources: {model_display_name(model_key)}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.22)
    return fig


def _dca_comparison_outputs(model_key: str, model_df: pd.DataFrame, config: dict, artifacts: ArtifactManager) -> list[Path]:
    fig = _dca_comparison_figure(model_key, model_df)
    try:
        return save_figure_variants(fig, artifacts, FigureExportSpec(stem=f"dca_datasource_comparison_{model_safe_name(model_key)}"), config)
    finally:
        plt.close(fig)


def generate_curve_outputs(artifacts: ArtifactManager, config: dict) -> list[Path]:
    predictions_path = artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet")
    if not predictions_path.exists():
        return []
    predictions_df = pd.read_parquet(predictions_path)
    predictions_df = predictions_df[predictions_df["split_name"].astype(str) == "test"].copy()
    if predictions_df.empty:
        return []
    runtime_plan = build_stage_runtime_plan(config, "report_curves")

    outputs: list[Path] = []
    dataset_groups = [(dataset_regime, dataset_df.copy()) for dataset_regime, dataset_df in predictions_df.groupby("dataset_regime", sort=False)]
    dataset_outputs = Parallel(n_jobs=max(1, runtime_plan.report_workers), backend="loky")(
        delayed(_curve_outputs_for_dataset)(dataset_regime, dataset_df, config, artifacts)
        for dataset_regime, dataset_df in dataset_groups
    )
    outputs.extend(path for group in dataset_outputs for path in group)

    dca_ci_path = artifacts.paths.artifact_path("evaluation", "dca_bootstrap_ci.csv")
    dca_path = artifacts.paths.artifact_path("evaluation", "dca_curves.csv")
    dca_df = pd.read_csv(dca_ci_path) if dca_ci_path.exists() else (pd.read_csv(dca_path) if dca_path.exists() else pd.DataFrame())
    if not dca_df.empty:
        dca_groups = [
            (keys, group_df.copy())
            for keys, group_df in dca_df.groupby(["dataset_regime", "population_id", "model_key"], sort=False)
        ]
        dca_outputs = Parallel(n_jobs=max(1, runtime_plan.report_workers), backend="loky")(
            delayed(_dca_outputs_for_group)(keys, group_df, config, artifacts)
            for keys, group_df in dca_groups
        )
        outputs.extend(path for group in dca_outputs for path in group)

        comparison_source = pd.read_csv(dca_path) if dca_path.exists() else dca_df
        comparison_groups = [
            (model_key, model_df.copy())
            for model_key, model_df in comparison_source.groupby("model_key", sort=False)
            if model_df["dataset_regime"].nunique() > 1
        ]
        comparison_outputs = Parallel(n_jobs=max(1, runtime_plan.report_workers), backend="loky")(
            delayed(_dca_comparison_outputs)(model_key, model_df, config, artifacts)
            for model_key, model_df in comparison_groups
        )
        outputs.extend(path for group in comparison_outputs for path in group)

    return outputs
