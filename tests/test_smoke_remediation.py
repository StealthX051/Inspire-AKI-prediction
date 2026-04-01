from __future__ import annotations

import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inspire_aki.cohort.filters import apply_preop_filters
from inspire_aki.config import load_config
from inspire_aki.features.intraop_tabular import build_intraop_features, safe_entropy, safe_kurtosis, safe_skew, safe_trend
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.models.tabular import FittedTabularBundle, _AUTOGLUON_SAMPLE_WEIGHT_COLUMN, fit_tabular_model, predict_tabular_bundle
from inspire_aki.models.weighting import balance_sample_weights, positive_balance_weight
from inspire_aki.pipelines.evaluate_generate import run_evaluate_generate
from inspire_aki.pipelines.preprocess import run_intraop, run_labels, run_preop, run_tabular
from inspire_aki.pipelines.report import run_manuscript
from inspire_aki.pipelines.train import run_train_tabular
from inspire_aki.reporting.reclassification import generate_reclassification_outputs
from inspire_aki.reporting.shap import generate_shap_outputs


def _prepare_reporting_inputs(config_path: Path, *, include_generated_evaluation: bool = True) -> dict:
    config = load_config(config_path)
    run_preop(config)
    run_intraop(config)
    run_tabular(config)
    run_labels(config)
    if include_generated_evaluation and config.get("evaluation_mode", "legacy_repeated_cv") != "legacy_repeated_cv":
        run_evaluate_generate(config)
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
    bmi_row = table.loc[table["characteristic"].astype(str).str.contains("BMI", regex=False)].iloc[0]

    assert "-0." not in str(bmi_row["finding"])
    numeric_mean = float(str(bmi_row["finding"]).split(" +/- ")[0])
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


def test_generate_shap_outputs_accepts_grouped_nested_manifest_names(synthetic_config: Path) -> None:
    config = _prepare_reporting_inputs(synthetic_config)
    artifacts = ArtifactManager(config)

    run_train_tabular(config)
    grouped_path = artifacts.paths.artifact_path("datasets", "splits", "grouped_nested_cv_combined.parquet")
    bootstrap_path = artifacts.paths.artifact_path("datasets", "splits", "bootstrap_combined.parquet")

    config["reports"]["shap_jobs"] = [{"dataset_regime": "combined", "model_key": "log_reg"}]
    outputs = generate_shap_outputs(artifacts, config)

    assert grouped_path.exists()
    assert not bootstrap_path.exists()
    assert artifacts.paths.artifact_path("explainability", "shap_importance_combined_log_reg.csv") in outputs
    assert artifacts.paths.artifact_path("reports", "figures", "shap_beeswarm_combined_log_reg.png") in outputs


def test_generate_reclassification_outputs_skips_empty_summary(loaded_synthetic_config) -> None:
    artifacts = ArtifactManager(loaded_synthetic_config)
    path = artifacts.resolve("evaluation", "reclassification_summary.csv")
    path.write_text("\n", encoding="utf-8")

    outputs = generate_reclassification_outputs(artifacts)

    assert outputs == []


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


def test_mlp_prediction_accepts_dataframe_input() -> None:
    torch = pytest.importorskip("torch")

    class FakeMLP(torch.nn.Module):
        def forward(self, tensor):
            return torch.zeros((tensor.shape[0], 1), device=tensor.device)

    bundle = FittedTabularBundle(
        model_key="mlp",
        feature_names=["feature_a", "feature_b"],
        scaler=None,
        model=FakeMLP(),
        metadata={},
    )
    test_df = pd.DataFrame({"feature_a": [0.1, 0.2], "feature_b": [1.0, 2.0]})

    y_pred, y_prob = predict_tabular_bundle(bundle, test_df, target="aki_boolean")

    assert y_pred.tolist() == [1, 1]
    assert y_prob.tolist() == pytest.approx([0.5, 0.5])


def test_autogluon_uses_non_reserved_sample_weight_column(loaded_synthetic_config, tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTabularPredictor:
        def __init__(self, *, label, eval_metric, path, sample_weight):
            captured["label"] = label
            captured["eval_metric"] = eval_metric
            captured["path"] = path
            captured["sample_weight"] = sample_weight

        def fit(self, *, train_data, time_limit, presets, num_cpus, num_gpus, dynamic_stacking, fit_strategy, excluded_model_types=None):
            captured["columns"] = list(train_data.columns)
            captured["train_data"] = train_data.copy()
            captured["time_limit"] = time_limit
            captured["presets"] = presets
            captured["num_cpus"] = num_cpus
            captured["num_gpus"] = num_gpus
            captured["dynamic_stacking"] = dynamic_stacking
            captured["fit_strategy"] = fit_strategy
            captured["excluded_model_types"] = excluded_model_types
            return self

    monkeypatch.setitem(sys.modules, "autogluon.tabular", types.SimpleNamespace(TabularPredictor=FakeTabularPredictor))
    monkeypatch.setattr("inspire_aki.models.tabular._autogluon_model_type_available", lambda *_args: True)

    train_df = pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feature_b": [1.0, 2.0, 3.0, 4.0, 5.0],
            "aki_boolean": [0, 0, 0, 1, 0],
        }
    )
    bundle = fit_tabular_model(
        model_key="autogluon",
        train_df=train_df,
        feature_cols=["feature_a", "feature_b"],
        target="aki_boolean",
        params={},
        config=loaded_synthetic_config,
        model_output_dir=tmp_path / "autogluon_model",
        seed=42,
    )

    assert bundle.model_key == "autogluon"
    assert captured["eval_metric"] == "balanced_accuracy"
    assert captured["sample_weight"] == _AUTOGLUON_SAMPLE_WEIGHT_COLUMN
    assert _AUTOGLUON_SAMPLE_WEIGHT_COLUMN in captured["columns"]
    assert "balance_weight" not in captured["columns"]
    assert captured["num_gpus"] == "auto"
    assert captured["dynamic_stacking"] is False
    assert captured["fit_strategy"] == "sequential"
    assert captured["excluded_model_types"] is None
    train_data = captured["train_data"]
    positive_weights = train_data.loc[train_data["aki_boolean"] == 1, _AUTOGLUON_SAMPLE_WEIGHT_COLUMN]
    negative_weights = train_data.loc[train_data["aki_boolean"] == 0, _AUTOGLUON_SAMPLE_WEIGHT_COLUMN]
    assert positive_weights.nunique() == 1
    assert positive_weights.iloc[0] == pytest.approx(4.0)
    assert negative_weights.eq(1.0).all()


def test_autogluon_excludes_missing_optional_model_types(loaded_synthetic_config, tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeTabularPredictor:
        def __init__(self, *, label, eval_metric, path, sample_weight):
            return None

        def fit(self, **kwargs):
            captured.update(kwargs)
            return self

    monkeypatch.setitem(sys.modules, "autogluon.tabular", types.SimpleNamespace(TabularPredictor=FakeTabularPredictor))
    availability = {"CAT": False, "FASTAI": False, "XGB": True}
    monkeypatch.setattr("inspire_aki.models.tabular._autogluon_model_type_available", lambda model_type, *_args: availability[model_type])

    train_df = pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.3, 0.4, 0.5],
            "feature_b": [1.0, 2.0, 3.0, 4.0, 5.0],
            "aki_boolean": [0, 0, 0, 1, 0],
        }
    )
    fit_tabular_model(
        model_key="autogluon",
        train_df=train_df,
        feature_cols=["feature_a", "feature_b"],
        target="aki_boolean",
        params={},
        config=loaded_synthetic_config,
        model_output_dir=tmp_path / "autogluon_model",
        seed=42,
    )

    assert captured["excluded_model_types"] == ["CAT", "FASTAI"]


def test_balance_sample_weights_upweight_positive_class() -> None:
    y = np.array([0, 0, 0, 1])

    weights = balance_sample_weights(y)

    assert positive_balance_weight(y) == pytest.approx(3.0)
    assert weights.tolist() == pytest.approx([1.0, 1.0, 1.0, 3.0])


def test_log_reg_training_uses_balance_sample_weights(loaded_synthetic_config, tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeLogisticRegression:
        def __init__(self, **kwargs):
            captured["init_kwargs"] = kwargs

        def fit(self, x, y, sample_weight=None):
            captured["x_shape"] = getattr(x, "shape", None)
            captured["y"] = np.asarray(y)
            captured["sample_weight"] = np.asarray(sample_weight)
            return self

    monkeypatch.setattr("inspire_aki.models.tabular.LogisticRegression", FakeLogisticRegression)
    monkeypatch.setattr("inspire_aki.models.tabular.save_tabular_bundle", lambda *_args, **_kwargs: None)

    train_df = pd.DataFrame(
        {
            "feature_a": [0.1, 0.2, 0.3, 0.4],
            "feature_b": [1.0, 2.0, 3.0, 4.0],
            "aki_boolean": [0, 0, 0, 1],
        }
    )
    bundle = fit_tabular_model(
        model_key="log_reg",
        train_df=train_df,
        feature_cols=["feature_a", "feature_b"],
        target="aki_boolean",
        params={"C": 1.0},
        config=loaded_synthetic_config,
        model_output_dir=tmp_path / "log_reg_model",
        seed=42,
    )

    assert bundle.model_key == "log_reg"
    assert captured["sample_weight"].tolist() == pytest.approx([1.0, 1.0, 1.0, 3.0])
    assert "class_weight" not in captured["init_kwargs"]
