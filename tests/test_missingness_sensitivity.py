from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

from inspire_aki.config import load_config
from inspire_aki.datasets.tabular import assemble_tabular_base_frame
from inspire_aki.evaluation.split_manager import evaluation_runs
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.pipelines.evaluate import run_calibration, run_metrics
from inspire_aki.pipelines.evaluate_generate import run_evaluate_generate
from inspire_aki.pipelines.preprocess import run_intraop, run_labels, run_preop, run_tabular
from inspire_aki.pipelines.report import run_shap
from inspire_aki.pipelines.train import run_train_tabular
from inspire_aki.reporting.missingness_sensitivity import (
    _load_combined_inputs,
    _resolve_context,
    missing_indicator_name,
    prepare_missingness_sensitivity_fold,
    run_missingness_sensitivity_analysis,
)


def _write_full_config(path: Path, config: dict) -> Path:
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _write_executable(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)
    return path


def _reviewer_test_config(synthetic_config: Path, artifact_root: Path) -> Path:
    config = copy.deepcopy(load_config(synthetic_config))
    config["paths"]["artifacts_dir"] = str(artifact_root)
    config["evaluation_mode"] = "grouped_nested_cv"
    config["splits"]["n_cv_folds"] = 2
    config["evaluation"]["bootstrap_reps"] = 5
    config["calibration"]["cv_folds"] = 2
    config["calibration"]["threshold_steps"] = 11
    config["models"]["tabular_enabled"] = ["xgb"]
    config["models"]["sequence_enabled"] = []
    config["models"]["tabular_hpo_enabled"] = []
    config["models"]["sequence_hpo_enabled"] = []
    config["reports"]["shap_jobs"] = [
        {
            "run_name": "XGBoost_Combined_Reviewer_Test",
            "model_key": "xgb",
            "dataset_regime": "combined",
            "plots": ["beeswarm"],
            "scatter_features": [],
            "dependence_pairs": [],
        }
    ]
    return _write_full_config(artifact_root.parent / "reviewer_missingness_test.yaml", config)


def _inject_missingness_for_test(config: dict) -> tuple[str, str]:
    artifacts = ArtifactManager(config)
    intraop_path = artifacts.paths.artifact_path("features", "intraop", "feature_engineered.csv")
    intraop_df = pd.read_csv(intraop_path)
    candidate_columns = [
        column
        for column in intraop_df.columns
        if column != "op_id"
        and pd.api.types.is_numeric_dtype(intraop_df[column])
        and "department" not in str(column)
        and "aki" not in str(column)
    ]
    if len(candidate_columns) < 2:
        raise ValueError("The synthetic intraop feature table did not expose enough numeric columns for the reviewer test.")

    high_missing_feature = "sum_uo" if "sum_uo" in candidate_columns else candidate_columns[0]
    low_missing_feature = next(column for column in candidate_columns if column != high_missing_feature)

    high_missing_rows = intraop_df.index[:2]
    low_missing_row = intraop_df.index[-1]
    intraop_df.loc[high_missing_rows, high_missing_feature] = np.nan
    intraop_df.loc[[low_missing_row], low_missing_feature] = np.nan
    intraop_df.to_csv(intraop_path, index=False)
    return high_missing_feature, low_missing_feature


def _prepare_reviewer_baseline(synthetic_config: Path, artifact_root: Path) -> tuple[Path, str, str]:
    config_path = _reviewer_test_config(synthetic_config, artifact_root)
    config = load_config(config_path)

    run_preop(config)
    run_intraop(config)
    high_missing_feature, low_missing_feature = _inject_missingness_for_test(config)
    run_tabular(config)
    run_labels(config)
    run_evaluate_generate(config)
    run_train_tabular(config)
    run_calibration(config)
    run_metrics(config)
    run_shap(config)
    return config_path, high_missing_feature, low_missing_feature


def test_prepare_missingness_sensitivity_fold_is_train_fit_and_keeps_indicators_binary(loaded_synthetic_config) -> None:
    config = copy.deepcopy(loaded_synthetic_config)
    config["features"]["outlier_quantiles"] = {
        "lower_extreme": 0.0,
        "upper_extreme": 1.0,
        "lower_fill_low": 0.0,
        "lower_fill_high": 0.0,
        "upper_fill_low": 1.0,
        "upper_fill_high": 1.0,
    }
    target = config["models"]["target"]
    train_df = pd.DataFrame(
        [
            {"op_id": 1, target: 0, "high_feature": 1.0, "low_feature": 2.0, "stable_feature": 1.0, "department_GS": 1.0},
            {"op_id": 2, target: 1, "high_feature": np.nan, "low_feature": 4.0, "stable_feature": 2.0, "department_GS": 0.0},
            {"op_id": 3, target: 0, "high_feature": 5.0, "low_feature": 6.0, "stable_feature": 3.0, "department_GS": 1.0},
        ]
    )
    test_df = pd.DataFrame(
        [
            {"op_id": 4, target: 1, "high_feature": np.nan, "low_feature": np.nan, "stable_feature": 2.5, "department_GS": 0.0},
            {"op_id": 5, target: 0, "high_feature": 5.0, "low_feature": 6.0, "stable_feature": 3.0, "department_GS": 1.0},
        ]
    )
    feature_cols = ["high_feature", "low_feature", "stable_feature", "department_GS"]
    fill_rates = pd.DataFrame(
        [
            {"feature": "high_feature", "fill_rate": 0.50},
            {"feature": "low_feature", "fill_rate": 0.95},
            {"feature": "stable_feature", "fill_rate": 1.0},
            {"feature": "department_GS", "fill_rate": 1.0},
        ]
    )

    prepared = prepare_missingness_sensitivity_fold(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
        fill_rates=fill_rates,
        config=config,
    )

    expected_indicator = missing_indicator_name("high_feature")
    assert prepared.high_missing_cols == ["high_feature"]
    assert prepared.low_missing_cols == ["low_feature"]
    assert expected_indicator in prepared.model_feature_cols
    assert expected_indicator not in prepared.scaling_columns
    assert set(prepared.train_model_df[expected_indicator].astype(int).tolist()) == {0, 1}
    assert set(prepared.test_model_df[expected_indicator].astype(int).tolist()) == {0, 1}
    assert float(prepared.display_test_df.loc[prepared.display_test_df[expected_indicator] == 1, expected_indicator].iloc[0]) == 1.0
    assert bool(prepared.display_test_df["high_feature"].isna().iloc[0])

    scaler = StandardScaler()
    scaling_columns = ["high_feature", "low_feature", "stable_feature"]
    train_scaled = train_df[feature_cols].copy()
    test_scaled = test_df[feature_cols].copy()
    scaler.fit(train_scaled[scaling_columns])
    train_scaled.loc[:, scaling_columns] = scaler.transform(train_scaled[scaling_columns])
    test_scaled.loc[:, scaling_columns] = scaler.transform(test_scaled[scaling_columns])

    expected_median = float(pd.to_numeric(train_scaled["high_feature"], errors="coerce").median())
    assert prepared.scaled_medians["high_feature"] == pytest.approx(expected_median)
    assert prepared.test_model_df["high_feature"].iloc[0] == pytest.approx(expected_median)

    knn = KNNImputer(n_neighbors=int(config["features"]["knn_neighbors"]))
    expected_low = knn.fit_transform(train_scaled[["low_feature"]])
    expected_test_low = knn.transform(test_scaled[["low_feature"]])
    assert prepared.train_model_df["low_feature"].to_numpy() == pytest.approx(expected_low[:, 0])
    assert prepared.test_model_df["low_feature"].to_numpy() == pytest.approx(expected_test_low[:, 0])


def test_prepare_missingness_sensitivity_fold_reuses_train_fit_outlier_handling(loaded_synthetic_config) -> None:
    config = copy.deepcopy(loaded_synthetic_config)
    target = config["models"]["target"]
    train_df = pd.DataFrame(
        [
            {"op_id": 1, target: 0, "stable_feature": 1.0},
            {"op_id": 2, target: 1, "stable_feature": 2.0},
            {"op_id": 3, target: 0, "stable_feature": 3.0},
            {"op_id": 4, target: 1, "stable_feature": 4.0},
            {"op_id": 5, target: 0, "stable_feature": 5.0},
        ]
    )
    test_df = pd.DataFrame([{"op_id": 6, target: 1, "stable_feature": 1000.0}])
    fill_rates = pd.DataFrame([{"feature": "stable_feature", "fill_rate": 1.0}])

    prepared = prepare_missingness_sensitivity_fold(
        train_df=train_df,
        test_df=test_df,
        feature_cols=["stable_feature"],
        fill_rates=fill_rates,
        config=config,
    )

    assert float(prepared.display_test_df["stable_feature"].iloc[0]) != pytest.approx(1000.0)
    assert abs(float(prepared.test_model_df["stable_feature"].iloc[0])) < 10.0


def test_missingness_sensitivity_requires_grouped_mode(tmp_path: Path, loaded_synthetic_config) -> None:
    config = copy.deepcopy(loaded_synthetic_config)
    config["evaluation_mode"] = "legacy_repeated_cv"
    config_path = _write_full_config(tmp_path / "legacy_reviewer_missingness.yaml", config)

    with pytest.raises(ValueError, match="grouped evaluation mode"):
        run_missingness_sensitivity_analysis(config_path=config_path)


def test_missingness_sensitivity_workflow_writes_comparison_outputs(
    synthetic_config: Path,
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "baseline_artifacts"
    config_path, high_missing_feature, _low_missing_feature = _prepare_reviewer_baseline(synthetic_config, baseline_root)
    config = load_config(config_path)
    artifacts = ArtifactManager(config)
    manifest_path = artifacts.paths.artifact_path("datasets", "splits", "grouped_nested_cv_combined.parquet")
    manifest = pd.read_parquet(manifest_path)

    assert set(manifest["evaluation_mode"]) == {"grouped_nested_cv"}
    assert len(evaluation_runs(manifest)) == int(config["splits"]["n_cv_folds"])

    sensitivity_root = tmp_path / "sensitivity_artifacts"
    reports_dir = tmp_path / "reports"
    outputs = run_missingness_sensitivity_analysis(
        config_path=config_path,
        sensitivity_artifacts_dir=sensitivity_root,
        out_dir=reports_dir,
    )

    output_paths = {key: Path(value) for key, value in outputs.items()}
    for required_key in [
        "baseline_metrics_summary",
        "baseline_shap_importance",
        "raw_predictions",
        "calibrated_predictions",
        "metrics_summary",
        "shap_importance",
        "shap_beeswarm",
        "performance_comparison",
        "shap_comparison",
        "converted_features",
        "indicator_ranks",
        "summary",
    ]:
        assert output_paths[required_key].exists(), required_key

    performance_df = pd.read_csv(output_paths["performance_comparison"])
    assert {"metric", "source_metric", "baseline_value", "sensitivity_value", "delta", "absolute_delta"}.issubset(
        performance_df.columns
    )
    assert set(performance_df["metric"]) == {
        "accuracy",
        "auprc",
        "auroc",
        "f1",
        "precision",
        "sensitivity",
        "specificity",
        "threshold",
    }

    converted_df = pd.read_csv(output_paths["converted_features"])
    expected_indicator = missing_indicator_name(high_missing_feature)
    assert {"feature_name", "fill_rate", "missing_rate", "indicator_column_name"}.issubset(converted_df.columns)
    assert high_missing_feature in set(converted_df["feature_name"])
    assert expected_indicator in set(converted_df["indicator_column_name"])

    shap_df = pd.read_csv(output_paths["shap_comparison"])
    assert {
        "feature_name",
        "original_rank",
        "sensitivity_rank",
        "rank_change",
        "original_mean_abs_shap",
        "sensitivity_mean_abs_shap",
        "fill_rate",
        "missing_rate",
        "had_gt10_missingness_before",
        "corresponding_missing_flag_rank",
    }.issubset(shap_df.columns)
    target_row = shap_df.loc[shap_df["feature_name"] == high_missing_feature].iloc[0]
    assert target_row["had_gt10_missingness_before"] == "yes"
    assert pd.notna(target_row["corresponding_missing_flag_rank"])

    indicator_df = pd.read_csv(output_paths["indicator_ranks"])
    assert {"indicator_column_name", "feature_name", "sensitivity_rank", "sensitivity_mean_abs_shap"}.issubset(
        indicator_df.columns
    )
    assert expected_indicator in set(indicator_df["indicator_column_name"])

    summary_text = output_paths["summary"].read_text(encoding="utf-8")
    assert "This reviewer analysis is intentionally separate from the default CLI preprocessing path." in summary_text
    assert high_missing_feature in summary_text
    assert "Outer evaluation reused the maintained patient-grouped manifests." in summary_text


def test_missingness_sensitivity_rebuilds_inputs_from_upstream_feature_artifacts(
    synthetic_config: Path,
    tmp_path: Path,
) -> None:
    baseline_root = tmp_path / "baseline_artifacts"
    config_path, high_missing_feature, _low_missing_feature = _prepare_reviewer_baseline(synthetic_config, baseline_root)
    config = load_config(config_path)
    artifacts = ArtifactManager(config)

    preop_df = pd.read_csv(artifacts.paths.artifact_path("features", "preop", "preop_features.csv"))
    intraop_df = pd.read_csv(artifacts.paths.artifact_path("features", "intraop", "feature_engineered.csv"))
    expected_base = assemble_tabular_base_frame(preop_df, intraop_df).set_index("op_id")

    combined_path = artifacts.paths.artifact_path("datasets", "tabular", "tabular_combined_unnormalized.csv")
    combined_df = pd.read_csv(combined_path)
    target_op_id = int(combined_df.loc[combined_df[high_missing_feature].notna(), "op_id"].iloc[0])
    combined_df.loc[combined_df["op_id"] == target_op_id, high_missing_feature] = 987654.0
    combined_df.to_csv(combined_path, index=False)

    context = _resolve_context(config_path=config_path, baseline_artifacts_dir=None, sensitivity_artifacts_dir=None, out_dir=tmp_path / "reports")
    modeling_df, _fill_rates, _manifest = _load_combined_inputs(context)
    loaded_value = float(modeling_df.set_index("op_id").loc[target_op_id, high_missing_feature])

    assert loaded_value == pytest.approx(float(expected_base.loc[target_op_id, high_missing_feature]))
    assert loaded_value != pytest.approx(987654.0)


def test_reviewer_missingness_wrapper_uses_active_environment_bins(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cli_log = tmp_path / "cli_calls.log"
    python_log = tmp_path / "python_calls.log"
    _write_executable(
        bin_dir / "inspire-aki",
        f"""#!/usr/bin/env bash
echo "$@" >> "{cli_log}"
exit 0
""",
    )
    _write_executable(
        bin_dir / "python",
        f"""#!/usr/bin/env bash
echo "$@" >> "{python_log}"
if [[ "${{1:-}}" == "-c" ]]; then
  echo "/tmp/fake-baseline-artifacts"
fi
exit 0
""",
    )

    result = subprocess.run(
        [
            "bash",
            str(repo_root / "scripts" / "run_reviewer_missingness_sensitivity.sh"),
            str(repo_root / "configs" / "aki" / "reviewer_combined_xgb_baseline.yaml"),
            str(tmp_path / "sensitivity_artifacts"),
            str(tmp_path / "reports"),
        ],
        cwd=repo_root,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    cli_calls = cli_log.read_text(encoding="utf-8").strip().splitlines()
    python_calls = python_log.read_text(encoding="utf-8").strip().splitlines()

    assert any("preprocess preop --config" in call for call in cli_calls)
    assert any("explain shap --config" in call for call in cli_calls)
    assert any("combined_xgb_missingness_sensitivity.py" in call for call in python_calls)
    assert any(call.startswith("-c ") for call in python_calls)
