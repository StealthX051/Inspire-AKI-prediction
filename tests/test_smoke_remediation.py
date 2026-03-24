from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inspire_aki.cohort.filters import apply_preop_filters
from inspire_aki.config import load_config
from inspire_aki.features.intraop_tabular import build_intraop_features, safe_entropy, safe_kurtosis, safe_skew, safe_trend
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.models.tabular import fit_tabular_model, predict_tabular_bundle
from inspire_aki.pipelines.preprocess import run_intraop, run_labels, run_preop, run_tabular
from inspire_aki.pipelines.report import run_manuscript
from inspire_aki.pipelines.train import run_train_tabular


def _prepare_reporting_inputs(config_path: Path) -> dict:
    config = load_config(config_path)
    run_preop(config)
    run_intraop(config)
    run_tabular(config)
    run_labels(config)
    return config


def test_apply_preop_filters_excludes_nonpositive_op_len(loaded_synthetic_config) -> None:
    df_preop = pd.DataFrame(
        {
            "op_id": [1, 2, 3],
            "subject_id": [11, 12, 13],
            "age": [40, 50, 60],
            "asa": [2, 2, 2],
            "opstart_time": [10.0, 20.0, 30.0],
            "opend_time": [15.0, 20.0, 25.0],
            "sex": ["M", "F", "M"],
            "weight": [70.0, 80.0, 90.0],
            "height": [170.0, 175.0, 180.0],
        }
    )
    df_ops = pd.DataFrame(
        {
            "op_id": [1, 2, 3],
            "subject_id": [11, 12, 13],
            "antype": ["General", "General", "General"],
            "department": ["GS", "GS", "GS"],
        }
    )

    filtered_df, _, audit = apply_preop_filters(df_preop, df_ops, loaded_synthetic_config, [])

    assert filtered_df["op_id"].tolist() == [1]
    assert pd.DataFrame(audit)["step"].tolist().count("positive_op_len_only") == 1


def test_build_preop_features_excludes_prefix_ops_from_output(synthetic_config: Path) -> None:
    config = load_config(synthetic_config)
    raw_dir = synthetic_config.parent / "raw"
    operations_path = raw_dir / "operations.csv"
    operations = pd.read_csv(operations_path)
    excluded_op_id = int(operations.loc[0, "op_id"])
    operations.loc[0, "icd10_pcs"] = "10ZZ"
    operations.to_csv(operations_path, index=False)

    preop_module = __import__("inspire_aki.cohort.preop", fromlist=["build_preop_features", "_extract_preop_item_feature"])
    preop_df, _ = preop_module.build_preop_features(config, raw_dir)

    assert excluded_op_id not in preop_df["op_id"].tolist()


def test_safe_intraop_statistics_are_finite() -> None:
    constant = np.array([1.0, 1.0, 1.0, 1.0])
    zeros = np.array([0.0, 0.0, 0.0, 0.0])

    assert safe_entropy(constant) == 0.0
    assert safe_entropy(zeros) == 0.0
    assert safe_kurtosis(constant) == 0.0
    assert safe_skew(constant) == 0.0
    assert safe_trend(constant) == 0.0


def test_intraop_features_do_not_emit_infinite_values(synthetic_config: Path) -> None:
    config = load_config(synthetic_config)
    raw_dir = synthetic_config.parent / "raw"
    preop_df, _ = __import__("inspire_aki.cohort.preop", fromlist=["build_preop_features"]).build_preop_features(config, raw_dir)
    vitals_df = pd.read_csv(raw_dir / "vitals.csv")

    intraop_df = build_intraop_features(vitals_df, preop_df, config)
    numeric = intraop_df.select_dtypes(include=[np.number])

    assert not np.isinf(numeric.to_numpy()).any()


def test_run_intraop_manifest_records_finite_artifact_metadata(synthetic_config: Path) -> None:
    config = load_config(synthetic_config)
    artifacts = ArtifactManager(config)

    run_preop(config)
    run_intraop(config)

    manifest = artifacts.read_json("manifests", "preprocess_intraop.json")
    assert manifest["metadata"]["n_inf_values"] == 0
    assert manifest["metadata"]["n_nan_values"] >= 0


def test_cohort_characteristics_use_unnormalized_units(synthetic_config: Path) -> None:
    config = _prepare_reporting_inputs(synthetic_config)
    artifacts = ArtifactManager(config)

    run_manuscript(config)
    table = pd.read_csv(artifacts.paths.artifact_path("reports", "tables", "cohort_characteristics.csv"))
    bmi_row = table.loc[table["feature"] == "BMI"].iloc[0]

    assert "-0." not in str(bmi_row["overall"])
    numeric_mean = float(str(bmi_row["overall"]).split(" +/- ")[0])
    assert numeric_mean > 10.0


def test_report_manuscript_manifest_includes_outputs(synthetic_config: Path) -> None:
    config = _prepare_reporting_inputs(synthetic_config)
    artifacts = ArtifactManager(config)

    run_train_tabular(config)
    config["reports"]["shap_jobs"] = [{"dataset_regime": "combined", "model_key": "log_reg"}]
    run_manuscript(config)

    manifest = artifacts.read_json("manifests", "report_manuscript.json")
    assert manifest["metadata"]["n_outputs"] > 0
    assert len(manifest["outputs"]) == manifest["metadata"]["n_outputs"]


def test_tabular_bundle_prediction_emits_no_scaler_feature_name_warning(synthetic_config: Path) -> None:
    config = _prepare_reporting_inputs(synthetic_config)
    artifacts = ArtifactManager(config)
    dataset = pd.read_csv(artifacts.paths.artifact_path("datasets", "tabular", "tabular_preop_labeled.csv"))
    target = config["models"]["target"]
    feature_cols = [col for col in dataset.columns if col not in ["op_id", target]]
    bundle_dir = artifacts.paths.artifact_path("models", "tabular", "preop", "log_reg", "repeat_0", "fold_0")
    bundle = fit_tabular_model(
        model_key="log_reg",
        train_df=dataset.head(512),
        feature_cols=feature_cols,
        target=target,
        params={},
        config=config,
        model_output_dir=bundle_dir,
        seed=config["splits"]["random_state"],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predict_tabular_bundle(bundle, dataset.tail(32), target)

    matching = [warning for warning in caught if "feature names" in str(warning.message)]
    assert not matching
