from __future__ import annotations

from time import perf_counter

import pandas as pd

from inspire_aki.evaluation.calibration import calibrate_prediction_groups
from inspire_aki.evaluation.dca import decision_curve_outputs
from inspire_aki.evaluation.delong import delong_comparison_outputs
from inspire_aki.evaluation.metrics import compute_group_metrics, summarize_group_metrics
from inspire_aki.evaluation.reclassification import compute_reclassification_summary
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.predictions import read_raw_predictions
from inspire_aki.runtime import build_stage_runtime_plan


def run_calibration(config: dict) -> dict[str, str]:
    stage_name = "evaluate_calibration"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    predictions_df = read_raw_predictions(artifacts)
    group_count = len(predictions_df.groupby(["dataset_regime", "population_id", "model_key"]))
    result = calibrate_prediction_groups(predictions_df, config)
    pred_path = artifacts.write_dataframe(result.predictions, "predictions", "calibrated_predictions.parquet")
    outputs = {"predictions": str(pred_path)}
    if not result.thresholds.empty:
        threshold_path = artifacts.write_dataframe(result.thresholds, "evaluation", "thresholds.csv")
        outputs["thresholds"] = str(threshold_path)
    artifacts.write_manifest(
        "evaluate_calibration",
        ["manifests", "evaluate_calibration.json"],
            inputs=[artifacts.relative(artifacts.paths.artifact_path("predictions", "raw_predictions.parquet"))],
            outputs=[artifacts.relative(artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet"))],
            metadata={"n_rows": len(result.predictions)},
            stage_runtime_plan=build_stage_runtime_plan(config, stage_name, {"group_count": group_count}).as_dict(),
            wall_time_seconds=perf_counter() - start,
    )
    return outputs


def run_metrics(config: dict) -> dict[str, str]:
    stage_name = "evaluate_metrics"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    predictions_df = pd.read_parquet(artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet"))
    fold_metrics = compute_group_metrics(predictions_df, config)
    summary_metrics, bootstrap_metrics = summarize_group_metrics(predictions_df, config)
    fold_path = artifacts.write_dataframe(fold_metrics, "evaluation", "metrics_by_fold.csv")
    summary_path = artifacts.write_dataframe(summary_metrics, "evaluation", "metrics_summary.csv")
    outputs = {"fold_metrics": str(fold_path), "summary_metrics": str(summary_path)}
    if not bootstrap_metrics.empty:
        bootstrap_path = artifacts.write_dataframe(bootstrap_metrics, "evaluation", "metrics_bootstrap_ci.csv")
        outputs["bootstrap_metrics"] = str(bootstrap_path)
    artifacts.write_manifest(
        stage_name,
        ["manifests", "evaluate_metrics.json"],
        inputs=[artifacts.relative(artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet"))],
        outputs=[artifacts.relative(fold_path), artifacts.relative(summary_path)]
        + ([artifacts.relative(bootstrap_path)] if "bootstrap_metrics" in outputs else []),
        metadata={"n_rows": len(predictions_df)},
        stage_runtime_plan=build_stage_runtime_plan(
            config,
            stage_name,
            {"group_count": len(predictions_df.groupby(["dataset_regime", "population_id", "model_key"]))},
        ).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return outputs


def run_delong(config: dict) -> dict[str, str]:
    stage_name = "evaluate_delong"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    predictions_df = pd.read_parquet(artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet"))
    matrix_df, long_df, corrected_matrix_df, corrected_long_df = delong_comparison_outputs(predictions_df, config)
    matrix_path = artifacts.write_dataframe(matrix_df.reset_index().rename(columns={"index": "model_name"}), "evaluation", "delong_matrix.csv")
    outputs = {"matrix": str(matrix_path)}
    if not long_df.empty:
        long_path = artifacts.write_dataframe(long_df, "evaluation", "delong_long.csv")
        outputs["long"] = str(long_path)
    if not corrected_matrix_df.empty:
        corrected_matrix_path = artifacts.write_dataframe(
            corrected_matrix_df.reset_index().rename(columns={"index": "model_name"}),
            "evaluation",
            "delong_fdr_corrected.csv",
        )
        outputs["corrected_matrix"] = str(corrected_matrix_path)
    if not corrected_long_df.empty:
        corrected_long_path = artifacts.write_dataframe(corrected_long_df, "evaluation", "delong_fdr_corrected_long.csv")
        outputs["corrected_long"] = str(corrected_long_path)
    artifacts.write_manifest(
        stage_name,
        ["manifests", "evaluate_delong.json"],
        inputs=[artifacts.relative(artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet"))],
        outputs=[artifacts.relative(matrix_path)]
        + ([artifacts.relative(long_path)] if "long" in outputs else [])
        + ([artifacts.relative(corrected_matrix_path)] if "corrected_matrix" in outputs else [])
        + ([artifacts.relative(corrected_long_path)] if "corrected_long" in outputs else []),
        metadata={"n_rows": len(predictions_df)},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return outputs


def run_dca(config: dict) -> dict[str, str]:
    stage_name = "evaluate_dca"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    predictions_df = pd.read_parquet(artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet"))
    dca_result = decision_curve_outputs(predictions_df, config)
    path = artifacts.write_dataframe(dca_result.curves, "evaluation", "dca_curves.csv")
    outputs = {"dca": str(path)}
    if not dca_result.bootstrap_ci.empty:
        bootstrap_path = artifacts.write_dataframe(dca_result.bootstrap_ci, "evaluation", "dca_bootstrap_ci.csv")
        outputs["bootstrap_ci"] = str(bootstrap_path)
    artifacts.write_manifest(
        stage_name,
        ["manifests", "evaluate_dca.json"],
        inputs=[artifacts.relative(artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet"))],
        outputs=[artifacts.relative(path)] + ([artifacts.relative(bootstrap_path)] if "bootstrap_ci" in outputs else []),
        metadata={"n_rows": len(dca_result.curves)},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return outputs


def run_reclassification(config: dict) -> dict[str, str]:
    stage_name = "evaluate_reclassification"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    predictions_df = pd.read_parquet(artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet"))
    summary_df = compute_reclassification_summary(predictions_df)
    path = artifacts.write_dataframe(summary_df, "evaluation", "reclassification_summary.csv")
    artifacts.write_manifest(
        stage_name,
        ["manifests", "evaluate_reclassification.json"],
        inputs=[artifacts.relative(artifacts.paths.artifact_path("predictions", "calibrated_predictions.parquet"))],
        outputs=[artifacts.relative(path)],
        metadata={"n_rows": len(summary_df)},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return {"reclassification": str(path)}
