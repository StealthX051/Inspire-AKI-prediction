from __future__ import annotations

from pathlib import Path

import pandas as pd

from inspire_aki.datasets.splits import build_bootstrap_split_manifest, subset_from_manifest
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.predictions import materialize_raw_predictions, write_prediction_partition
from inspire_aki.models.registry import sequence_dataset_for_model, sequence_population_for_model
from inspire_aki.models.sequence import fit_sequence_model, predict_sequence_bundle, raw_prediction_rows as sequence_prediction_rows
from inspire_aki.models.tabular import (
    fit_tabular_model,
    predict_tabular_bundle,
    raw_prediction_rows as tabular_prediction_rows,
    tabular_feature_columns,
)


def _tabular_params(config: dict, artifacts: ArtifactManager, dataset_regime: str, model_key: str) -> dict:
    tuning_path = artifacts.paths.artifact_path("tuning", "tabular_best_params.json")
    if tuning_path.exists():
        tuning = artifacts.read_json("tuning", "tabular_best_params.json")
        if dataset_regime in tuning and model_key in tuning[dataset_regime]:
            return tuning[dataset_regime][model_key]
    if model_key in config["models"]["tabular_hpo_params"].get(dataset_regime, {}):
        return config["models"]["tabular_hpo_params"][dataset_regime][model_key]
    return config["models"]["tabular_defaults"].get(model_key, {})


def _sequence_params(config: dict, artifacts: ArtifactManager, model_key: str) -> dict:
    params = dict(config["models"]["sequence_defaults"])
    tuning_path = artifacts.paths.artifact_path("tuning", "sequence_best_params.json")
    if tuning_path.exists():
        tuned = artifacts.read_json("tuning", "sequence_best_params.json")
        if model_key in tuned:
            tuned_params = dict(tuned[model_key])
            if "lr" in tuned_params:
                tuned_params["learning_rate"] = tuned_params.pop("lr")
            params.update(tuned_params)
    else:
        configured = dict(config["models"]["sequence_hpo_params"].get(model_key, {}))
        if "lr" in configured:
            configured["learning_rate"] = configured.pop("lr")
        params.update(configured)
    return params


def run_train_tabular(config: dict) -> dict[str, str]:
    artifacts = ArtifactManager(config)
    prediction_frames: list[pd.DataFrame] = []
    for dataset_regime in ["preop", "intraop", "combined"]:
        dataset_df = pd.read_csv(artifacts.paths.artifact_path("datasets", "tabular", f"tabular_{dataset_regime}_labeled.csv"))
        target = config["models"]["target"]
        manifest = build_bootstrap_split_manifest(
            dataset_df,
            target=target,
            dataset_regime=dataset_regime,
            population_id=dataset_regime,
            random_state=config["splits"]["random_state"],
            n_iterations=config["splits"]["n_bootstrap_iterations"],
            n_cv_folds=config["splits"]["n_cv_folds"],
            use_bootstrapping=config["splits"]["use_bootstrapping"],
        )
        manifest_path = artifacts.write_dataframe(manifest, "datasets", "splits", f"bootstrap_{dataset_regime}.parquet")
        feature_cols = tabular_feature_columns(dataset_df, target)
        split_keys = manifest[["repeat_id", "fold_id"]].drop_duplicates().sort_values(["repeat_id", "fold_id"])

        for row in split_keys.itertuples(index=False):
            train_df = subset_from_manifest(dataset_df, manifest, repeat_id=row.repeat_id, fold_id=row.fold_id, split_name="train")
            test_df = subset_from_manifest(dataset_df, manifest, repeat_id=row.repeat_id, fold_id=row.fold_id, split_name="test")
            for model_key in config["models"]["tabular_enabled"]:
                if model_key == "asa_rule" and "asa" not in feature_cols:
                    continue
                model_dir = artifacts.paths.artifact_path("models", "tabular", dataset_regime, model_key, f"repeat_{row.repeat_id}", f"fold_{row.fold_id}")
                params = _tabular_params(config, artifacts, dataset_regime, model_key)
                bundle = fit_tabular_model(
                    model_key=model_key,
                    train_df=train_df,
                    feature_cols=feature_cols,
                    target=target,
                    params=params,
                    config=config,
                    model_output_dir=model_dir,
                    seed=config["splits"]["random_state"] + row.repeat_id * 100 + row.fold_id,
                )
                y_pred, y_prob = predict_tabular_bundle(bundle, test_df, target)
                prediction_frames.append(
                    tabular_prediction_rows(
                        dataset_regime=dataset_regime,
                        population_id=dataset_regime,
                        model_key=model_key,
                        target=target,
                        repeat_id=row.repeat_id,
                        fold_id=row.fold_id,
                        test_df=test_df,
                        y_pred=y_pred,
                        y_prob=y_prob,
                    )
                )
        artifacts.write_manifest(
            f"train_tabular_{dataset_regime}",
            ["manifests", f"train_tabular_{dataset_regime}.json"],
            inputs=[artifacts.relative(artifacts.paths.artifact_path("datasets", "tabular", f"tabular_{dataset_regime}_labeled.csv"))],
            outputs=[artifacts.relative(manifest_path)],
            metadata={"n_models": len(config["models"]["tabular_enabled"])},
        )

    raw_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    partition_path = write_prediction_partition("tabular", raw_predictions, artifacts)
    out_path = materialize_raw_predictions(artifacts)
    return {"partition": str(partition_path), "predictions": str(out_path)}


def run_train_sequence(config: dict) -> dict[str, str]:
    artifacts = ArtifactManager(config)
    if not config["models"]["sequence_enabled"]:
        return {}
    sequence_path = artifacts.paths.artifact_path("datasets", "sequence", "lstm_trainable.pkl")
    if not sequence_path.exists():
        return {}

    sequence_df = artifacts.read_pickle("datasets", "sequence", "lstm_trainable.pkl")
    target = config["models"]["target"]
    manifest = build_bootstrap_split_manifest(
        sequence_df,
        target=target,
        dataset_regime="sequence",
        population_id="sequence_common",
        random_state=config["splits"]["random_state"],
        n_iterations=config["splits"]["n_bootstrap_iterations"],
        n_cv_folds=config["splits"]["n_cv_folds"],
        use_bootstrapping=config["splits"]["use_bootstrapping"],
    )
    manifest_path = artifacts.write_dataframe(manifest, "datasets", "splits", "bootstrap_sequence.parquet")
    split_keys = manifest[["repeat_id", "fold_id"]].drop_duplicates().sort_values(["repeat_id", "fold_id"])
    feature_cols_tab = [col for col in sequence_df.columns if col not in ["op_id", "time_tensors", "seq_len", target]]
    prediction_frames: list[pd.DataFrame] = []

    for model_key in config["models"]["sequence_enabled"]:
        params = _sequence_params(config, artifacts, model_key)
        dataset_regime = sequence_dataset_for_model(model_key)
        population_id = sequence_population_for_model(model_key)
        for row in split_keys.itertuples(index=False):
            train_df = subset_from_manifest(sequence_df, manifest, repeat_id=row.repeat_id, fold_id=row.fold_id, split_name="train")
            test_df = subset_from_manifest(sequence_df, manifest, repeat_id=row.repeat_id, fold_id=row.fold_id, split_name="test")
            model_dir = artifacts.paths.artifact_path("models", "sequence", model_key, f"repeat_{row.repeat_id}", f"fold_{row.fold_id}")
            bundle = fit_sequence_model(
                model_key=model_key,
                train_df=train_df,
                feature_cols_tab=feature_cols_tab,
                params=params,
                config=config,
                model_output_dir=model_dir,
                seed=config["splits"]["random_state"] + row.repeat_id * 100 + row.fold_id,
            )
            y_pred, y_prob = predict_sequence_bundle(bundle, test_df)
            prediction_frames.append(
                sequence_prediction_rows(
                    dataset_regime=dataset_regime,
                    population_id=population_id,
                    model_key=model_key,
                    repeat_id=row.repeat_id,
                    fold_id=row.fold_id,
                    test_df=test_df,
                    y_pred=y_pred,
                    y_prob=y_prob,
                )
            )

    if not prediction_frames:
        return {}
    raw_predictions = pd.concat(prediction_frames, ignore_index=True)
    partition_path = write_prediction_partition("sequence", raw_predictions, artifacts)
    out_path = materialize_raw_predictions(artifacts)
    artifacts.write_manifest(
        "train_sequence",
        ["manifests", "train_sequence.json"],
        inputs=[artifacts.relative(sequence_path)],
        outputs=[artifacts.relative(manifest_path), artifacts.relative(partition_path), artifacts.relative(out_path)],
        metadata={"n_models": len(config["models"]["sequence_enabled"])},
    )
    return {"partition": str(partition_path), "predictions": str(out_path)}
