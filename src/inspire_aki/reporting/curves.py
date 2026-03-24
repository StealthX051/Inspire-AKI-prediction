from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.registry import model_colors, model_display_name
from inspire_aki.runtime import build_stage_runtime_plan


def _save_plot(path: Path, figure_dpi: int) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=figure_dpi, bbox_inches="tight")
    plt.close()


def _curve_outputs_for_dataset(dataset_regime: str, dataset_df: pd.DataFrame, config: dict, artifacts: ArtifactManager) -> list[Path]:
    import matplotlib.pyplot as plt

    figure_dpi = config["reports"]["figure_dpi"]
    outputs: list[Path] = []
    plot_groups = list(dataset_df.groupby("model_key", sort=False))
    if not plot_groups:
        return outputs

    plt.figure(figsize=(8, 6))
    for model_key, model_df in plot_groups:
        y_true = model_df["y_true"].astype(int).to_numpy()
        y_prob = model_df["y_prob_calibrated"].fillna(model_df["y_prob_raw"]).astype(float).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.plot(fpr, tpr, label=model_display_name(model_key), color=model_colors.get(model_display_name(model_key)))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves: {dataset_regime}")
    plt.legend(loc="lower right")
    roc_path = artifacts.resolve("reports", "figures", f"roc_curves_{dataset_regime}.png")
    _save_plot(roc_path, figure_dpi)
    outputs.append(roc_path)

    plt.figure(figsize=(8, 6))
    for model_key, model_df in plot_groups:
        y_true = model_df["y_true"].astype(int).to_numpy()
        y_prob = model_df["y_prob_calibrated"].fillna(model_df["y_prob_raw"]).astype(float).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.plot(recall, precision, label=model_display_name(model_key), color=model_colors.get(model_display_name(model_key)))
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curves: {dataset_regime}")
    plt.legend(loc="lower left")
    pr_path = artifacts.resolve("reports", "figures", f"pr_curves_{dataset_regime}.png")
    _save_plot(pr_path, figure_dpi)
    outputs.append(pr_path)

    plt.figure(figsize=(8, 6))
    for model_key, model_df in plot_groups:
        y_true = model_df["y_true"].astype(int).to_numpy()
        y_prob = model_df["y_prob_calibrated"].fillna(model_df["y_prob_raw"]).astype(float).to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", label=model_display_name(model_key), color=model_colors.get(model_display_name(model_key)))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Fraction Positive")
    plt.title(f"Calibration Curves: {dataset_regime}")
    plt.legend(loc="upper left")
    calib_path = artifacts.resolve("reports", "figures", f"calibration_curves_{dataset_regime}.png")
    _save_plot(calib_path, figure_dpi)
    outputs.append(calib_path)
    return outputs


def _dca_outputs_for_dataset(dataset_regime: str, dataset_df: pd.DataFrame, config: dict, artifacts: ArtifactManager) -> list[Path]:
    import matplotlib.pyplot as plt

    figure_dpi = config["reports"]["figure_dpi"]
    if dataset_df.empty:
        return []
    plt.figure(figsize=(8, 6))
    for model_key, model_df in dataset_df.groupby("model_key", sort=False):
        plt.plot(
            model_df["threshold_prob"] * 100.0,
            model_df["net_benefit_model"],
            label=model_display_name(model_key),
            color=model_colors.get(model_display_name(model_key)),
        )
    base_df = next(iter(dataset_df.groupby("model_key")))[1]
    plt.plot(base_df["threshold_prob"] * 100.0, base_df["net_benefit_treat_all"], linestyle="--", color="black", label="Treat All")
    plt.plot(base_df["threshold_prob"] * 100.0, base_df["net_benefit_treat_none"], linestyle="-", color="gray", label="Treat None")
    plt.xlabel("Threshold Probability (%)")
    plt.ylabel("Net Benefit")
    plt.title(f"Decision Curve Analysis: {dataset_regime}")
    plt.legend(loc="upper right")
    out_path = artifacts.resolve("reports", "figures", f"dca_curves_{dataset_regime}.png")
    _save_plot(out_path, figure_dpi)
    return [out_path]


def generate_curve_outputs(artifacts: ArtifactManager, config: dict) -> list[Path]:
    predictions_path = artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet")
    if not predictions_path.exists():
        return []
    predictions_df = pd.read_parquet(predictions_path)
    runtime_plan = build_stage_runtime_plan(config, "report_curves")

    outputs: list[Path] = []
    prediction_groups = [(dataset_regime, dataset_df.copy()) for dataset_regime, dataset_df in predictions_df.groupby("dataset_regime", sort=False)]
    if prediction_groups:
        curve_outputs_nested = Parallel(n_jobs=max(1, runtime_plan.report_workers), backend="loky")(
            delayed(_curve_outputs_for_dataset)(dataset_regime, dataset_df, config, artifacts)
            for dataset_regime, dataset_df in prediction_groups
        )
        outputs.extend(path for group in curve_outputs_nested for path in group)

    dca_path = artifacts.paths.artifact_path("evaluation", "dca_curves.csv")
    if dca_path.exists():
        dca_df = pd.read_csv(dca_path)
        dca_groups = [(dataset_regime, dataset_df.copy()) for dataset_regime, dataset_df in dca_df.groupby("dataset_regime", sort=False)]
        if dca_groups:
            dca_outputs_nested = Parallel(n_jobs=max(1, runtime_plan.report_workers), backend="loky")(
                delayed(_dca_outputs_for_dataset)(dataset_regime, dataset_df, config, artifacts)
                for dataset_regime, dataset_df in dca_groups
            )
            outputs.extend(path for group in dca_outputs_nested for path in group)

    return outputs
