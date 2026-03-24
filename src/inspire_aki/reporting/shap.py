from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from inspire_aki.datasets.splits import subset_from_manifest
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.models.tabular import load_tabular_bundle
from inspire_aki.registry import SUPPORTED_SHAP_MODELS
from inspire_aki.runtime import build_stage_runtime_plan


def _load_dataset_for_regime(artifacts: ArtifactManager, dataset_regime: str) -> pd.DataFrame:
    return pd.read_csv(artifacts.paths.artifact_path("datasets", "tabular", f"tabular_{dataset_regime}_labeled.csv"))


def _load_split(artifacts: ArtifactManager, dataset_regime: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(artifacts.paths.artifact_path("datasets", "splits", f"bootstrap_{dataset_regime}.parquet"))
    dataset_df = _load_dataset_for_regime(artifacts, dataset_regime)
    train_df = subset_from_manifest(dataset_df, manifest, repeat_id=0, fold_id=0, split_name="train")
    test_df = subset_from_manifest(dataset_df, manifest, repeat_id=0, fold_id=0, split_name="test")
    return train_df, test_df


def _shap_bundle_dir(artifacts: ArtifactManager, dataset_regime: str, model_key: str) -> Path:
    return artifacts.paths.artifact_path("models", "tabular", dataset_regime, model_key, "repeat_0", "fold_0")


def _generate_shap_job(config: dict, dataset_regime: str, model_key: str) -> list[str]:
    artifacts = ArtifactManager(config)
    bundle_dir = _shap_bundle_dir(artifacts, dataset_regime, model_key)
    train_df, test_df = _load_split(artifacts, dataset_regime)
    bundle = load_tabular_bundle(model_key, bundle_dir)
    feature_cols = bundle.feature_names
    x_train = train_df[feature_cols].copy()
    x_test = test_df[feature_cols].copy()
    sample_n = min(len(x_train), 200)
    explain_n = min(len(x_test), 200)
    x_background = x_train.sample(n=sample_n, random_state=42) if sample_n and len(x_train) > sample_n else x_train
    x_explain = x_test.sample(n=explain_n, random_state=42) if explain_n and len(x_test) > explain_n else x_test
    if x_background.empty or x_explain.empty:
        return []
    if bundle.scaler is not None:
        x_background = pd.DataFrame(
            bundle.scaler.transform(x_background),
            columns=feature_cols,
            index=x_background.index,
        )
        x_explain = pd.DataFrame(
            bundle.scaler.transform(x_explain),
            columns=feature_cols,
            index=x_explain.index,
        )

    import matplotlib.pyplot as plt
    import shap

    if model_key in {"xgb", "rf"}:
        explainer = shap.TreeExplainer(bundle.model)
        shap_values = explainer.shap_values(x_explain)
        values = shap_values if not isinstance(shap_values, list) else shap_values[1]
    elif model_key == "log_reg":
        explainer = shap.LinearExplainer(bundle.model, x_background)
        values = explainer.shap_values(x_explain)
    else:
        raise ValueError(f"Unsupported SHAP model_key '{model_key}'.")

    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    csv_path = artifacts.write_dataframe(
        importance,
        "explainability",
        f"shap_importance_{dataset_regime}_{model_key}.csv",
    )

    plt.figure()
    shap.summary_plot(values, x_explain, show=False, max_display=config["reports"]["max_display_features"])
    png_path = artifacts.resolve("reports", "figures", f"shap_beeswarm_{dataset_regime}_{model_key}.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=config["reports"]["figure_dpi"], bbox_inches="tight")
    plt.close()
    return [str(csv_path), str(png_path)]


def generate_shap_outputs(artifacts: ArtifactManager, config: dict) -> list[Path]:
    outputs: list[Path] = []
    jobs = config["reports"]["shap_jobs"]
    if not jobs:
        return outputs

    for job in jobs:
        dataset_regime = job["dataset_regime"]
        model_key = job["model_key"]
        if model_key not in SUPPORTED_SHAP_MODELS:
            raise ValueError(f"Unsupported SHAP model_key '{model_key}'. Supported SHAP models: {list(SUPPORTED_SHAP_MODELS)}.")
        bundle_dir = _shap_bundle_dir(artifacts, dataset_regime, model_key)
        bundle_path = bundle_dir / "bundle.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Expected SHAP model artifact was not found: {bundle_path}")

    runtime_plan = build_stage_runtime_plan(config, "report_shap")
    outputs_nested = Parallel(n_jobs=max(1, runtime_plan.shap_workers), backend="loky")(
        delayed(_generate_shap_job)(config, job["dataset_regime"], job["model_key"])
        for job in jobs
    )
    for job_outputs in outputs_nested:
        outputs.extend(Path(path) for path in job_outputs)
    return outputs
