from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.registry import model_colors, model_display_name


def _save_plot(path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def generate_curve_outputs(artifacts: ArtifactManager) -> list[Path]:
    import matplotlib.pyplot as plt

    predictions_path = artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet")
    if not predictions_path.exists():
        return []
    predictions_df = pd.read_parquet(predictions_path)
    outputs: list[Path] = []

    for dataset_regime, dataset_df in predictions_df.groupby("dataset_regime", sort=False):
        plot_groups = list(dataset_df.groupby("model_key", sort=False))
        if not plot_groups:
            continue

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
        _save_plot(roc_path)
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
        _save_plot(pr_path)
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
        _save_plot(calib_path)
        outputs.append(calib_path)

    dca_path = artifacts.paths.artifact_path("evaluation", "dca_curves.csv")
    if dca_path.exists():
        dca_df = pd.read_csv(dca_path)
        for dataset_regime, dataset_df in dca_df.groupby("dataset_regime", sort=False):
            plt.figure(figsize=(8, 6))
            for model_key, model_df in dataset_df.groupby("model_key", sort=False):
                plt.plot(
                    model_df["threshold_prob"] * 100.0,
                    model_df["net_benefit_model"],
                    label=model_display_name(model_key),
                    color=model_colors.get(model_display_name(model_key)),
                )
            if not dataset_df.empty:
                base_df = next(iter(dataset_df.groupby("model_key")))[1]
                plt.plot(base_df["threshold_prob"] * 100.0, base_df["net_benefit_treat_all"], linestyle="--", color="black", label="Treat All")
                plt.plot(base_df["threshold_prob"] * 100.0, base_df["net_benefit_treat_none"], linestyle="-", color="gray", label="Treat None")
            plt.xlabel("Threshold Probability (%)")
            plt.ylabel("Net Benefit")
            plt.title(f"Decision Curve Analysis: {dataset_regime}")
            plt.legend(loc="upper right")
            out_path = artifacts.resolve("reports", "figures", f"dca_curves_{dataset_regime}.png")
            _save_plot(out_path)
            outputs.append(out_path)

    return outputs
