from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from inspire_aki.config import REPO_ROOT, active_target_column, config_hash, load_config
from inspire_aki.datasets.tabular import assemble_tabular_base_frame, tabular_ignore_columns
from inspire_aki.evaluation.calibration import calibrate_prediction_groups
from inspire_aki.evaluation.metrics import compute_group_metrics, summarize_group_metrics
from inspire_aki.evaluation.split_manager import evaluation_runs, subset_generated_manifest
from inspire_aki.features.normalization import apply_outlier_replacement_plan, fit_outlier_replacement_plan
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.predictions import materialize_raw_predictions, write_prediction_partition
from inspire_aki.models.tabular import (
    PreparedTabularFold,
    fit_tabular_model,
    load_tabular_bundle,
    predict_tabular_bundle,
    raw_prediction_rows,
    tabular_feature_columns,
)
from inspire_aki.models.weighting import balance_sample_weights
from inspire_aki.reporting.shap import (
    ShapExplanationBundle,
    _beeswarm_outputs,
    _build_importance_table,
    _compute_shap_values,
    _sample_frame,
    _write_importance_csv,
)
from inspire_aki.runtime import build_stage_runtime_plan


DEFAULT_REVIEWER_CONFIG_PATH = REPO_ROOT / "configs" / "aki" / "reviewer_combined_xgb_baseline.yaml"
SUMMARY_REPORT_NAME = "missingness_sensitivity_summary.md"
PERFORMANCE_COMPARISON_NAME = "missingness_sensitivity_performance_comparison.csv"
SHAP_COMPARISON_NAME = "missingness_sensitivity_shap_comparison.csv"
CONVERTED_FEATURES_NAME = "missingness_sensitivity_converted_features.csv"
INDICATOR_RANKS_NAME = "missingness_sensitivity_indicator_ranks.csv"
BASELINE_FIGURE_STEM = "shap_beeswarm_combined_xgb"
SENSITIVITY_STRATEGY_NAME = "median_plus_indicator_gt10"
_SHAP_BACKGROUND_SAMPLE_N = 200


@dataclass(frozen=True)
class ReviewerContext:
    config: dict[str, Any]
    config_path: Path
    baseline_artifacts: ArtifactManager
    sensitivity_artifacts: ArtifactManager
    out_dir: Path
    target_column: str
    threshold_pct: float
    baseline_config_hash: str


@dataclass(frozen=True)
class PreparedSensitivityFold:
    train_model_df: pd.DataFrame
    test_model_df: pd.DataFrame
    display_test_df: pd.DataFrame
    model_feature_cols: list[str]
    low_missing_cols: list[str]
    high_missing_cols: list[str]
    indicator_columns: list[str]
    scaling_columns: list[str]
    scaled_medians: dict[str, float]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reviewer-specific missingness sensitivity analysis for the combined GBT model "
            "using median imputation plus explicit missingness indicators for >10% missingness features."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_REVIEWER_CONFIG_PATH),
        help="Baseline reviewer config path. Defaults to configs/aki/reviewer_combined_xgb_baseline.yaml.",
    )
    parser.add_argument(
        "--baseline-artifacts-dir",
        default=None,
        help="Optional override for the baseline artifact root. Defaults to paths.artifacts_dir from the config.",
    )
    parser.add_argument(
        "--sensitivity-artifacts-dir",
        default=None,
        help="Optional output artifact root for the sensitivity rerun. Defaults to a sibling reviewer artifact directory.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional repo-local comparison output directory. Defaults to repo-root reports/.",
    )
    return parser


def missing_indicator_name(feature_name: str) -> str:
    return f"{feature_name}_missing_flag"


def _markdown_table(frame: pd.DataFrame, *, columns: list[str] | None = None) -> str:
    subset = frame.loc[:, columns] if columns is not None else frame
    if subset.empty:
        return "No rows were available."
    return subset.to_markdown(index=False)


def build_converted_feature_frame(
    fill_rates: pd.DataFrame,
    feature_cols: list[str],
    *,
    threshold_pct: float,
) -> pd.DataFrame:
    required = {"feature", "fill_rate"}
    missing = sorted(required - set(fill_rates.columns))
    if missing:
        raise ValueError(f"fill_rates.csv is missing required columns: {missing}")
    frame = fill_rates.loc[fill_rates["feature"].isin(feature_cols), ["feature", "fill_rate"]].copy()
    frame["fill_rate"] = pd.to_numeric(frame["fill_rate"], errors="coerce")
    frame["missing_rate"] = 1.0 - frame["fill_rate"]
    frame = frame.loc[(frame["missing_rate"] * 100.0) >= float(threshold_pct)].copy()
    frame = frame.sort_values(["missing_rate", "feature"], ascending=[False, True], kind="stable").reset_index(drop=True)
    frame = frame.rename(columns={"feature": "feature_name"})
    frame["indicator_column_name"] = frame["feature_name"].map(missing_indicator_name)
    return frame[["feature_name", "fill_rate", "missing_rate", "indicator_column_name"]]


def _low_missing_feature_names(
    fill_rates: pd.DataFrame,
    feature_cols: list[str],
    *,
    threshold_pct: float,
) -> list[str]:
    frame = fill_rates.loc[fill_rates["feature"].isin(feature_cols), ["feature", "fill_rate"]].copy()
    frame["fill_rate"] = pd.to_numeric(frame["fill_rate"], errors="coerce")
    frame["missing_pct"] = (1.0 - frame["fill_rate"]) * 100.0
    low_missing = frame.loc[(frame["missing_pct"] > 0.0) & (frame["missing_pct"] < float(threshold_pct)), "feature"]
    return low_missing.astype(str).tolist()


def _scaling_ignore_columns(frame: pd.DataFrame, config: dict, *, extra_ignore: set[str] | None = None) -> set[str]:
    ignore_cols = tabular_ignore_columns(frame, config)
    if extra_ignore:
        ignore_cols.update(extra_ignore)
    return ignore_cols


def prepare_missingness_sensitivity_fold(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    fill_rates: pd.DataFrame,
    config: dict,
) -> PreparedSensitivityFold:
    threshold_pct = float(config["features"]["high_missing_threshold_pct"])
    converted = build_converted_feature_frame(fill_rates, feature_cols, threshold_pct=threshold_pct)
    high_missing_cols = converted["feature_name"].astype(str).tolist()
    low_missing_cols = _low_missing_feature_names(fill_rates, feature_cols, threshold_pct=threshold_pct)

    indicator_map = {feature_name: indicator_name for feature_name, indicator_name in converted[["feature_name", "indicator_column_name"]].itertuples(index=False)}
    indicator_columns = [indicator_map[feature_name] for feature_name in high_missing_cols]

    train_model = train_df[["op_id"]].copy()
    test_model = test_df[["op_id"]].copy()
    if config["models"]["target"] in train_df.columns:
        train_model[config["models"]["target"]] = train_df[config["models"]["target"]].astype(int).to_numpy()
    if config["models"]["target"] in test_df.columns:
        test_model[config["models"]["target"]] = test_df[config["models"]["target"]].astype(int).to_numpy()

    train_features = train_df[feature_cols].copy()
    test_features = test_df[feature_cols].copy()
    outlier_ignore_cols = tabular_ignore_columns(train_features, config)
    outlier_columns = [
        col
        for col in feature_cols
        if col not in outlier_ignore_cols and pd.api.types.is_numeric_dtype(train_features[col])
    ]
    if outlier_columns:
        outlier_plan = fit_outlier_replacement_plan(train_features, outlier_columns, config)
        train_features = apply_outlier_replacement_plan(
            train_features,
            columns=outlier_columns,
            plan=outlier_plan,
            config=config,
        )
        test_features = apply_outlier_replacement_plan(
            test_features,
            columns=outlier_columns,
            plan=outlier_plan,
            config=config,
        )

    display_test = test_features.copy()
    for feature_name, indicator_name in indicator_map.items():
        train_model[indicator_name] = train_df[feature_name].isna().astype(int)
        test_model[indicator_name] = test_df[feature_name].isna().astype(int)
        display_test[indicator_name] = test_df[feature_name].isna().astype(int)

    ignore_cols = _scaling_ignore_columns(train_features, config, extra_ignore=set(indicator_columns))
    scaling_columns = [
        col
        for col in feature_cols
        if col not in ignore_cols and pd.api.types.is_numeric_dtype(train_features[col])
    ]
    if scaling_columns:
        scaler = StandardScaler()
        scaler.fit(train_features[scaling_columns])
        train_features.loc[:, scaling_columns] = scaler.transform(train_features[scaling_columns])
        test_features.loc[:, scaling_columns] = scaler.transform(test_features[scaling_columns])

    scaled_medians: dict[str, float] = {}
    for feature_name in high_missing_cols:
        median_value = float(pd.to_numeric(train_features[feature_name], errors="coerce").median())
        if not np.isfinite(median_value):
            median_value = 0.0
        scaled_medians[feature_name] = median_value
        train_features.loc[:, feature_name] = pd.to_numeric(train_features[feature_name], errors="coerce").fillna(median_value)
        test_features.loc[:, feature_name] = pd.to_numeric(test_features[feature_name], errors="coerce").fillna(median_value)

    if low_missing_cols:
        imputer = KNNImputer(n_neighbors=int(config["features"]["knn_neighbors"]))
        train_features.loc[:, low_missing_cols] = imputer.fit_transform(train_features[low_missing_cols])
        test_features.loc[:, low_missing_cols] = imputer.transform(test_features[low_missing_cols])

    train_model = pd.concat([train_model, train_features], axis=1)
    test_model = pd.concat([test_model, test_features], axis=1)
    model_feature_cols = [*feature_cols, *indicator_columns]
    return PreparedSensitivityFold(
        train_model_df=train_model,
        test_model_df=test_model,
        display_test_df=display_test[model_feature_cols].copy(),
        model_feature_cols=model_feature_cols,
        low_missing_cols=low_missing_cols,
        high_missing_cols=high_missing_cols,
        indicator_columns=indicator_columns,
        scaling_columns=scaling_columns,
        scaled_medians=scaled_medians,
    )


def _resolve_context(
    *,
    config_path: str | Path | None,
    baseline_artifacts_dir: str | Path | None,
    sensitivity_artifacts_dir: str | Path | None,
    out_dir: str | Path | None,
) -> ReviewerContext:
    resolved_config_path = Path(config_path) if config_path is not None else DEFAULT_REVIEWER_CONFIG_PATH
    config = copy.deepcopy(load_config(resolved_config_path))
    if config.get("evaluation_mode", "legacy_repeated_cv") == "legacy_repeated_cv":
        raise ValueError("This reviewer workflow requires a grouped evaluation mode, not legacy_repeated_cv.")

    if baseline_artifacts_dir is not None:
        config["paths"]["artifacts_dir"] = str(Path(baseline_artifacts_dir))
    baseline_config_hash = config_hash(config)
    baseline_artifacts = ArtifactManager(config)

    sensitivity_config = copy.deepcopy(config)
    if sensitivity_artifacts_dir is None:
        baseline_root = baseline_artifacts.paths.artifacts_root
        default_sensitivity_root = baseline_root.parent / f"{baseline_root.name}_{SENSITIVITY_STRATEGY_NAME}"
        sensitivity_config["paths"]["artifacts_dir"] = str(default_sensitivity_root)
    else:
        sensitivity_config["paths"]["artifacts_dir"] = str(Path(sensitivity_artifacts_dir))
    sensitivity_artifacts = ArtifactManager(sensitivity_config)

    resolved_out_dir = Path(out_dir) if out_dir is not None else (REPO_ROOT / "reports")
    if not resolved_out_dir.is_absolute():
        resolved_out_dir = REPO_ROOT / resolved_out_dir
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    return ReviewerContext(
        config=sensitivity_config,
        config_path=resolved_config_path,
        baseline_artifacts=baseline_artifacts,
        sensitivity_artifacts=sensitivity_artifacts,
        out_dir=resolved_out_dir,
        target_column=active_target_column(sensitivity_config),
        threshold_pct=float(sensitivity_config["features"]["high_missing_threshold_pct"]),
        baseline_config_hash=baseline_config_hash,
    )


def _combined_manifest_path(artifacts: ArtifactManager, config: dict) -> Path:
    evaluation_mode = str(config.get("evaluation_mode", "legacy_repeated_cv"))
    return artifacts.paths.artifact_path("datasets", "splits", f"{evaluation_mode}_combined.parquet")


def _load_combined_inputs(context: ReviewerContext) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_artifacts = context.baseline_artifacts
    preop_path = baseline_artifacts.paths.artifact_path("features", "preop", "preop_features.csv")
    intraop_path = baseline_artifacts.paths.artifact_path("features", "intraop", "feature_engineered.csv")
    labels_path = baseline_artifacts.paths.artifact_path("cohort", "labels.csv")
    fill_path = baseline_artifacts.paths.artifact_path("features", "fill_rates.csv")
    manifest_path = _combined_manifest_path(baseline_artifacts, context.config)

    for path in (preop_path, intraop_path, labels_path, fill_path, manifest_path):
        if not path.exists():
            raise FileNotFoundError(f"Required baseline artifact was not found: {path}")

    preop_df = pd.read_csv(preop_path)
    intraop_df = pd.read_csv(intraop_path)
    labels = pd.read_csv(labels_path)
    fill_rates = pd.read_csv(fill_path)
    manifest = pd.read_parquet(manifest_path)

    if context.target_column not in labels.columns:
        raise KeyError(f"Target column '{context.target_column}' was not present in {labels_path}.")

    label_cols = ["op_id", context.target_column]
    combined_base = assemble_tabular_base_frame(preop_df, intraop_df)
    modeling_df = combined_base.merge(labels[label_cols], on="op_id", how="inner", validate="one_to_one")
    return modeling_df, fill_rates, manifest


def _baseline_metric_row(context: ReviewerContext) -> pd.Series:
    metrics_path = context.baseline_artifacts.paths.artifact_path("evaluation", "metrics_summary.csv")
    if not metrics_path.exists():
        raise FileNotFoundError(f"Baseline metrics summary was not found: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    subset = metrics.loc[(metrics["dataset_regime"].astype(str) == "combined") & (metrics["model_key"].astype(str) == "xgb")].copy()
    if subset.empty:
        raise ValueError(f"Baseline metrics summary did not contain a combined/xgb row: {metrics_path}")
    return subset.iloc[0]


def _baseline_shap_importance(context: ReviewerContext) -> pd.DataFrame:
    path = context.baseline_artifacts.paths.artifact_path("explainability", "shap_importance_combined_xgb.csv")
    if not path.exists():
        raise FileNotFoundError(f"Baseline SHAP importance CSV was not found: {path}")
    frame = pd.read_csv(path)
    frame["original_rank"] = np.arange(1, len(frame) + 1)
    return frame


def _xgb_params(config: dict) -> dict[str, Any]:
    params = copy.deepcopy(config["models"]["tabular_hpo_params"]["combined"]["xgb"])
    if not isinstance(params, dict):
        raise ValueError("Expected models.tabular_hpo_params.combined.xgb to be a mapping.")
    return params


def _write_sensitivity_artifacts(
    context: ReviewerContext,
    *,
    raw_predictions: pd.DataFrame,
    calibrated_predictions: pd.DataFrame,
    thresholds: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    summary_metrics: pd.DataFrame,
    bootstrap_metrics: pd.DataFrame,
) -> dict[str, Path]:
    artifacts = context.sensitivity_artifacts
    partition_path = write_prediction_partition("tabular", raw_predictions, artifacts)
    raw_path = materialize_raw_predictions(artifacts)
    calibrated_path = artifacts.write_dataframe(calibrated_predictions, "predictions", "calibrated_predictions.parquet")
    outputs = {
        "raw_partition": partition_path,
        "raw_predictions": raw_path,
        "calibrated_predictions": calibrated_path,
        "metrics_by_fold": artifacts.write_dataframe(fold_metrics, "evaluation", "metrics_by_fold.csv"),
        "metrics_summary": artifacts.write_dataframe(summary_metrics, "evaluation", "metrics_summary.csv"),
    }
    if not thresholds.empty:
        outputs["thresholds"] = artifacts.write_dataframe(thresholds, "evaluation", "thresholds.csv")
    if not bootstrap_metrics.empty:
        outputs["metrics_bootstrap_ci"] = artifacts.write_dataframe(bootstrap_metrics, "evaluation", "metrics_bootstrap_ci.csv")
    return outputs


def _fit_sensitivity_bundle_for_fold(
    *,
    prepared_fold: PreparedSensitivityFold,
    run,
    config: dict,
    artifacts: ArtifactManager,
    target: str,
    params: dict[str, Any],
) -> tuple[pd.DataFrame, Path]:
    y_train = prepared_fold.train_model_df[target].to_numpy()
    manual_fold = PreparedTabularFold(
        feature_cols=list(prepared_fold.model_feature_cols),
        target=target,
        train_df=prepared_fold.train_model_df.copy(),
        test_df=prepared_fold.test_model_df.copy(),
        x_train_scaled=prepared_fold.train_model_df[prepared_fold.model_feature_cols].copy(),
        x_test_scaled=prepared_fold.test_model_df[prepared_fold.model_feature_cols].copy(),
        y_train=y_train,
        sample_weights=balance_sample_weights(y_train),
        scaler=None,
    )
    model_dir = artifacts.paths.artifact_path(
        "models",
        "tabular",
        "combined",
        "xgb",
        f"repeat_{run.repeat_id}",
        f"fold_{run.fold_id}",
    )
    bundle = fit_tabular_model(
        model_key="xgb",
        train_df=prepared_fold.train_model_df,
        feature_cols=prepared_fold.model_feature_cols,
        target=target,
        params=params,
        config=config,
        model_output_dir=model_dir,
        seed=int(config["splits"]["random_state"]) + int(run.repeat_id) * 100 + int(run.fold_id),
        prepared_fold=manual_fold,
    )
    y_pred, y_prob = predict_tabular_bundle(bundle, prepared_fold.test_model_df, target, prepared_fold=manual_fold)
    predictions = raw_prediction_rows(
        dataset_regime="combined",
        population_id="combined",
        model_key="xgb",
        target=target,
        repeat_id=int(run.repeat_id),
        fold_id=int(run.fold_id),
        test_df=prepared_fold.test_model_df,
        y_pred=y_pred,
        y_prob=y_prob,
    )
    return predictions, model_dir


def _sensitivity_shap_bundle(
    *,
    config: dict,
    artifacts: ArtifactManager,
    model_dir: Path,
    prepared_fold: PreparedSensitivityFold,
    target: str,
) -> ShapExplanationBundle:
    runtime_plan = build_stage_runtime_plan(config, "report_shap")
    fitted_bundle = load_tabular_bundle("xgb", model_dir)
    feature_cols = list(prepared_fold.model_feature_cols)
    x_background = _sample_frame(prepared_fold.train_model_df[feature_cols], limit=_SHAP_BACKGROUND_SAMPLE_N)
    x_explain = prepared_fold.test_model_df[feature_cols].copy()
    display_frame = prepared_fold.display_test_df[feature_cols].copy()
    y_true = prepared_fold.test_model_df[target].astype(bool).copy()
    shap_values = _compute_shap_values(
        "xgb",
        fitted_bundle.model,
        x_background,
        x_explain,
        feature_cols,
        worker_budget=runtime_plan.shap_workers,
    )
    importance = _build_importance_table(feature_cols, shap_values)
    return ShapExplanationBundle(
        dataset_regime="combined",
        model_key="xgb",
        feature_names=tuple(feature_cols),
        display_frame=display_frame,
        shap_values=shap_values,
        importance=importance,
        y_true=y_true,
    )


def _metric_value(row: pd.Series, metric_name: str) -> float:
    return float(pd.to_numeric(pd.Series([row[metric_name]]), errors="coerce").iloc[0])


def build_performance_comparison_frame(baseline_row: pd.Series, sensitivity_row: pd.Series) -> pd.DataFrame:
    metric_map = [
        ("auroc", "auroc"),
        ("auprc", "auprc"),
        ("sensitivity", "recall"),
        ("specificity", "specificity"),
        ("precision", "precision"),
        ("f1", "f1"),
        ("accuracy", "accuracy"),
        ("threshold", "threshold"),
    ]
    rows: list[dict[str, object]] = []
    for metric_name, source_metric in metric_map:
        baseline_value = _metric_value(baseline_row, source_metric)
        sensitivity_value = _metric_value(sensitivity_row, source_metric)
        rows.append(
            {
                "metric": metric_name,
                "source_metric": source_metric,
                "baseline_value": baseline_value,
                "sensitivity_value": sensitivity_value,
                "delta": sensitivity_value - baseline_value,
                "absolute_delta": abs(sensitivity_value - baseline_value),
            }
        )
    return pd.DataFrame(rows)


def build_shap_comparison_frame(
    *,
    baseline_importance: pd.DataFrame,
    sensitivity_importance: pd.DataFrame,
    fill_rates: pd.DataFrame,
    converted_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = baseline_importance.copy().rename(columns={"feature": "feature_name", "mean_abs_shap": "original_mean_abs_shap"})
    sensitivity = sensitivity_importance.copy().rename(columns={"feature": "feature_name", "mean_abs_shap": "sensitivity_mean_abs_shap"})
    baseline["original_rank"] = np.arange(1, len(baseline) + 1)
    sensitivity["sensitivity_rank"] = np.arange(1, len(sensitivity) + 1)

    fill = fill_rates.loc[:, ["feature", "fill_rate"]].copy().rename(columns={"feature": "feature_name"})
    fill["fill_rate"] = pd.to_numeric(fill["fill_rate"], errors="coerce")
    fill["missing_rate"] = 1.0 - fill["fill_rate"]
    indicator_lookup = {row.feature_name: row.indicator_column_name for row in converted_features.itertuples(index=False)}

    comparison = baseline.merge(sensitivity, on="feature_name", how="outer")
    comparison = comparison.merge(fill, on="feature_name", how="left")
    comparison["rank_change"] = comparison["sensitivity_rank"] - comparison["original_rank"]
    comparison["had_gt10_missingness_before"] = comparison["feature_name"].isin(set(converted_features["feature_name"])).map({True: "yes", False: "no"})
    comparison["corresponding_missing_flag_rank"] = comparison["feature_name"].map(
        {
            feature_name: int(sensitivity.loc[sensitivity["feature_name"] == indicator_name, "sensitivity_rank"].iloc[0])
            for feature_name, indicator_name in indicator_lookup.items()
            if indicator_name in set(sensitivity["feature_name"])
        }
    )
    comparison = comparison.sort_values(
        ["original_rank", "sensitivity_rank", "feature_name"],
        ascending=[True, True, True],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    indicator_rows = sensitivity.loc[sensitivity["feature_name"].isin(set(converted_features["indicator_column_name"]))].copy()
    if indicator_rows.empty:
        indicator_rows = pd.DataFrame(columns=["indicator_column_name", "feature_name", "fill_rate", "missing_rate", "sensitivity_mean_abs_shap", "sensitivity_rank"])
    else:
        indicator_rows = indicator_rows.rename(columns={"feature_name": "indicator_column_name"})
        indicator_rows = indicator_rows.merge(
            converted_features.loc[:, ["feature_name", "indicator_column_name", "fill_rate", "missing_rate"]],
            on="indicator_column_name",
            how="left",
        )
    indicator_rows = indicator_rows.loc[:, ["indicator_column_name", "feature_name", "fill_rate", "missing_rate", "sensitivity_mean_abs_shap", "sensitivity_rank"]]
    return comparison, indicator_rows


def _reviewer_draft_sections(
    *,
    performance_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    indicator_df: pd.DataFrame,
    converted_features: pd.DataFrame,
) -> tuple[str, str]:
    metric_lookup = performance_df.set_index("metric")
    auroc_delta = float(metric_lookup.loc["auroc", "delta"])
    auprc_delta = float(metric_lookup.loc["auprc", "delta"])

    focus_features = shap_df.loc[shap_df["feature_name"].isin({"sum_uo", "sum_ebl", "preop_crp"})].copy()
    focus_lines = []
    for row in focus_features.itertuples(index=False):
        focus_lines.append(
            f"{row.feature_name}: rank {row.original_rank} to {row.sensitivity_rank}, "
            f"mean(|SHAP|) {row.original_mean_abs_shap:.4f} to {row.sensitivity_mean_abs_shap:.4f}"
        )
    focus_summary = "; ".join(focus_lines) if focus_lines else "Key reviewer-named features were not present in the exported SHAP table."

    top_indicator = (
        indicator_df.sort_values("sensitivity_rank", kind="stable").head(3)["indicator_column_name"].astype(str).tolist()
        if not indicator_df.empty
        else []
    )
    indicator_summary = ", ".join(top_indicator) if top_indicator else "No converted-feature indicators appeared in the sensitivity SHAP ranking."

    stable_version = (
        "We agree that missingness may itself be informative in routinely collected perioperative data. "
        "Our concern was therefore not whether the model could learn from missingness, but whether encoding missing values "
        "as a fixed sentinel could conflate missingness with the continuous value itself and affect tree-based interpretability. "
        "To address this, we performed a targeted sensitivity analysis in the combined GBT model, replacing the sentinel encoding "
        "used for variables with >10% missingness with median imputation plus explicit missingness indicators, while preserving the "
        "original KNN-based handling for variables with <10% missingness and otherwise keeping the grouped evaluation and grouped "
        "calibration framework unchanged. Model discrimination was similar under the alternative handling "
        f"(AUROC change {auroc_delta:+.4f}; AUPRC change {auprc_delta:+.4f}), and the main SHAP conclusions were broadly preserved. "
        f"For key reviewer-named features, {focus_summary}. The explicit indicator columns were ranked separately ({indicator_summary}), "
        "which supports the interpretation that the sensitivity run disentangled missingness from the continuous value rather than relying on a single extreme sentinel."
    )

    changed_version = (
        "We agree that missingness may itself be informative in routinely collected perioperative data. "
        "Our concern was therefore not whether the model could learn from missingness, but whether encoding missing values "
        "as a fixed sentinel could conflate missingness with the continuous value itself and affect tree-based interpretability. "
        "To address this, we performed a targeted sensitivity analysis in the combined GBT model, replacing the sentinel encoding "
        "used for variables with >10% missingness with median imputation plus explicit missingness indicators, while preserving the "
        "original KNN-based handling for variables with <10% missingness and otherwise keeping the grouped evaluation and grouped "
        "calibration framework unchanged. Under this alternative handling, model discrimination remained in the same overall range "
        f"but shifted modestly (AUROC change {auroc_delta:+.4f}; AUPRC change {auprc_delta:+.4f}), and some SHAP attributions changed. "
        f"For key reviewer-named features, {focus_summary}. The explicit indicator columns were ranked separately ({indicator_summary}), "
        "which suggests that part of the prior attribution for some incompletely observed variables was sensitive to the missingness encoding. "
        "We would therefore retain the overall predictive result while applying more caution to mechanistic interpretation of the affected features."
    )
    return stable_version, changed_version


def write_summary_report(
    *,
    context: ReviewerContext,
    baseline_row: pd.Series,
    sensitivity_row: pd.Series,
    performance_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    indicator_df: pd.DataFrame,
    converted_features: pd.DataFrame,
) -> Path:
    stable_version, changed_version = _reviewer_draft_sections(
        performance_df=performance_df,
        shap_df=shap_df,
        indicator_df=indicator_df,
        converted_features=converted_features,
    )
    top_indicator_frame = (
        indicator_df.sort_values("sensitivity_rank", kind="stable")
        .head(10)
        .loc[:, ["indicator_column_name", "feature_name", "sensitivity_rank", "sensitivity_mean_abs_shap"]]
        if not indicator_df.empty
        else pd.DataFrame()
    )
    key_feature_frame = (
        shap_df.loc[shap_df["feature_name"].isin({"sum_uo", "sum_ebl", "preop_crp"})]
        .loc[:, ["feature_name", "original_rank", "sensitivity_rank", "rank_change", "corresponding_missing_flag_rank"]]
        if not shap_df.empty
        else pd.DataFrame()
    )
    text = "\n".join(
        [
            "# Missingness Sensitivity Summary",
            "",
            "## Design Decision",
            "",
            "This reviewer analysis is intentionally separate from the default CLI preprocessing path. "
            "The current maintained pipeline applies missingness handling during tabular preprocessing before grouped outer train/test "
            "manifests are materialized, so a public CLI-wide missingness toggle would require a broader architectural refactor to remain "
            "fully fold-fit and leakage-safe. For this reviewer response, we therefore implemented a focused grouped-analysis workflow "
            "that leaves the manuscript default unchanged while fitting the alternative missingness handling inside each outer training fold only.",
            "",
            "## Leakage Safeguards",
            "",
            "- Outer evaluation reused the maintained patient-grouped manifests.",
            "- Outlier handling, scaling, and imputation were each fit on outer-train rows only for every fold.",
            "- Missingness indicators were generated from raw missingness before imputation.",
            "- Any outlier replacement reused train-fit quantiles on the held-out fold rather than full-cohort clipping.",
            "- Median imputation for >10% missingness features used outer-train medians only.",
            "- KNN imputation for <10% missingness features used outer-train fitting only.",
            "- Binary missingness indicators were kept as 0/1 and excluded from scaling.",
            "- No new HPO was run; the grouped sensitivity rerun reused the pinned combined XGB parameters from config.",
            "- Calibration reused the maintained grouped isotonic path on held-out predictions with op_id grouping.",
            "",
            "## Affected Features",
            "",
            _markdown_table(converted_features) if not converted_features.empty else "No >10% missingness features were identified.",
            "",
            "## Performance",
            "",
            _markdown_table(performance_df),
            "",
            "## Key SHAP Comparison",
            "",
            _markdown_table(key_feature_frame) if not key_feature_frame.empty else "No SHAP comparison rows were available.",
            "",
            "## Indicator Rankings",
            "",
            _markdown_table(top_indicator_frame)
            if not top_indicator_frame.empty
            else "No indicator columns were ranked in the exported sensitivity SHAP table.",
            "",
            "## Paths",
            "",
            f"- Baseline artifacts: `{context.baseline_artifacts.paths.artifacts_root}`",
            f"- Sensitivity artifacts: `{context.sensitivity_artifacts.paths.artifacts_root}`",
            f"- Baseline SHAP figure: `{context.baseline_artifacts.paths.artifact_path('reports', 'figures', f'{BASELINE_FIGURE_STEM}.png')}`",
            f"- Sensitivity SHAP figure: `{context.sensitivity_artifacts.paths.artifact_path('reports', 'figures', f'{BASELINE_FIGURE_STEM}.png')}`",
            "",
            "## Draft Reviewer Response",
            "",
            "### Version A: Stable Results",
            "",
            stable_version,
            "",
            "### Version B: Meaningful Shifts",
            "",
            changed_version,
            "",
            "## Run Metadata",
            "",
            f"- Config path: `{context.config_path}`",
            f"- Baseline config hash: `{context.baseline_config_hash}`",
            f"- Sensitivity config hash: `{config_hash(context.config)}`",
            f"- Sensitivity strategy: `{SENSITIVITY_STRATEGY_NAME}`",
            f"- Baseline combined/xgb AUROC: `{float(baseline_row['auroc']):.6f}`",
            f"- Sensitivity combined/xgb AUROC: `{float(sensitivity_row['auroc']):.6f}`",
        ]
    )
    path = context.out_dir / SUMMARY_REPORT_NAME
    path.write_text(text + "\n", encoding="utf-8")
    return path


def run_missingness_sensitivity_analysis(
    *,
    config_path: str | Path | None = None,
    baseline_artifacts_dir: str | Path | None = None,
    sensitivity_artifacts_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, str]:
    context = _resolve_context(
        config_path=config_path,
        baseline_artifacts_dir=baseline_artifacts_dir,
        sensitivity_artifacts_dir=sensitivity_artifacts_dir,
        out_dir=out_dir,
    )
    modeling_df, fill_rates, manifest = _load_combined_inputs(context)
    target = context.target_column
    feature_cols = tabular_feature_columns(modeling_df, target)
    converted_features = build_converted_feature_frame(fill_rates, feature_cols, threshold_pct=context.threshold_pct)
    params = _xgb_params(context.config)

    prediction_frames: list[pd.DataFrame] = []
    shap_fold: PreparedSensitivityFold | None = None
    shap_model_dir: Path | None = None
    for run in evaluation_runs(manifest):
        train_df = subset_generated_manifest(modeling_df, manifest, split_name="train", run_id=run.run_id)
        test_df = subset_generated_manifest(modeling_df, manifest, split_name="test", run_id=run.run_id)
        prepared_fold = prepare_missingness_sensitivity_fold(
            train_df=train_df,
            test_df=test_df,
            feature_cols=feature_cols,
            fill_rates=fill_rates,
            config=context.config,
        )
        predictions, model_dir = _fit_sensitivity_bundle_for_fold(
            prepared_fold=prepared_fold,
            run=run,
            config=context.config,
            artifacts=context.sensitivity_artifacts,
            target=target,
            params=params,
        )
        prediction_frames.append(predictions)
        if int(run.repeat_id) == 0 and int(run.fold_id) == 0:
            shap_fold = prepared_fold
            shap_model_dir = model_dir

    raw_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    calibration_result = calibrate_prediction_groups(raw_predictions, context.config)
    fold_metrics = compute_group_metrics(calibration_result.predictions, context.config)
    summary_metrics, bootstrap_metrics = summarize_group_metrics(calibration_result.predictions, context.config)
    artifact_outputs = _write_sensitivity_artifacts(
        context,
        raw_predictions=raw_predictions,
        calibrated_predictions=calibration_result.predictions,
        thresholds=calibration_result.thresholds,
        fold_metrics=fold_metrics,
        summary_metrics=summary_metrics,
        bootstrap_metrics=bootstrap_metrics,
    )

    if shap_fold is None or shap_model_dir is None:
        raise ValueError("The sensitivity run did not produce a repeat_0/fold_0 model for SHAP export.")
    shap_bundle = _sensitivity_shap_bundle(
        config=context.config,
        artifacts=context.sensitivity_artifacts,
        model_dir=shap_model_dir,
        prepared_fold=shap_fold,
        target=target,
    )
    shap_importance_path = _write_importance_csv(context.sensitivity_artifacts, shap_bundle)
    shap_figure_outputs = _beeswarm_outputs(context.sensitivity_artifacts, shap_bundle, context.config)

    baseline_row = _baseline_metric_row(context)
    sensitivity_row = summary_metrics.loc[
        (summary_metrics["dataset_regime"].astype(str) == "combined") & (summary_metrics["model_key"].astype(str) == "xgb")
    ].iloc[0]
    baseline_importance = _baseline_shap_importance(context)
    shap_comparison, indicator_ranks = build_shap_comparison_frame(
        baseline_importance=baseline_importance,
        sensitivity_importance=shap_bundle.importance,
        fill_rates=fill_rates,
        converted_features=converted_features,
    )
    performance_df = build_performance_comparison_frame(baseline_row, sensitivity_row)

    performance_path = context.out_dir / PERFORMANCE_COMPARISON_NAME
    performance_df.to_csv(performance_path, index=False)
    shap_comparison_path = context.out_dir / SHAP_COMPARISON_NAME
    shap_comparison.to_csv(shap_comparison_path, index=False)
    converted_features_path = context.out_dir / CONVERTED_FEATURES_NAME
    converted_features.to_csv(converted_features_path, index=False)
    indicator_ranks_path = context.out_dir / INDICATOR_RANKS_NAME
    indicator_ranks.to_csv(indicator_ranks_path, index=False)
    summary_path = write_summary_report(
        context=context,
        baseline_row=baseline_row,
        sensitivity_row=sensitivity_row,
        performance_df=performance_df,
        shap_df=shap_comparison,
        indicator_df=indicator_ranks,
        converted_features=converted_features,
    )

    context.sensitivity_artifacts.write_json(
        {
            "config_path": str(context.config_path),
            "baseline_artifacts_dir": str(context.baseline_artifacts.paths.artifacts_root),
            "sensitivity_artifacts_dir": str(context.sensitivity_artifacts.paths.artifacts_root),
            "out_dir": str(context.out_dir),
            "baseline_config_hash": context.baseline_config_hash,
            "sensitivity_config_hash": config_hash(context.config),
            "strategy": SENSITIVITY_STRATEGY_NAME,
            "n_converted_features": int(len(converted_features)),
            "converted_features": converted_features.to_dict(orient="records"),
        },
        "manifests",
        "reviewer_missingness_sensitivity.json",
    )

    return {
        "baseline_artifacts_dir": str(context.baseline_artifacts.paths.artifacts_root),
        "baseline_metrics_summary": str(context.baseline_artifacts.paths.artifact_path("evaluation", "metrics_summary.csv")),
        "baseline_shap_importance": str(
            context.baseline_artifacts.paths.artifact_path("explainability", "shap_importance_combined_xgb.csv")
        ),
        "baseline_shap_beeswarm": str(
            context.baseline_artifacts.paths.artifact_path("reports", "figures", f"{BASELINE_FIGURE_STEM}.png")
        ),
        "sensitivity_artifacts_dir": str(context.sensitivity_artifacts.paths.artifacts_root),
        "raw_predictions": str(artifact_outputs["raw_predictions"]),
        "calibrated_predictions": str(artifact_outputs["calibrated_predictions"]),
        "metrics_summary": str(artifact_outputs["metrics_summary"]),
        "shap_importance": str(shap_importance_path),
        "shap_beeswarm": str(next((path for path in shap_figure_outputs if path.suffix == ".png"), shap_figure_outputs[0])),
        "performance_comparison": str(performance_path),
        "shap_comparison": str(shap_comparison_path),
        "converted_features": str(converted_features_path),
        "indicator_ranks": str(indicator_ranks_path),
        "summary": str(summary_path),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    outputs = run_missingness_sensitivity_analysis(
        config_path=args.config,
        baseline_artifacts_dir=args.baseline_artifacts_dir,
        sensitivity_artifacts_dir=args.sensitivity_artifacts_dir,
        out_dir=args.out_dir,
    )
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0
