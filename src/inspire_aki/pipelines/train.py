from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd

from inspire_aki.datasets.splits import build_bootstrap_split_manifest, subset_from_manifest
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.predictions import materialize_raw_predictions, write_prediction_partition
from inspire_aki.io.progress import ProgressLogger
from inspire_aki.models.registry import sequence_dataset_for_model, sequence_population_for_model
from inspire_aki.models.sequence import fit_sequence_model, predict_sequence_bundle, raw_prediction_rows as sequence_prediction_rows
from inspire_aki.models.tabular import (
    fit_tabular_model,
    predict_tabular_bundle,
    prepare_tabular_fold,
    raw_prediction_rows as tabular_prediction_rows,
    selected_tabular_dataset_regimes,
    selected_tabular_models,
    tabular_execution_policy,
    tabular_feature_columns,
)
from inspire_aki.runtime import build_stage_runtime_plan


@dataclass(frozen=True)
class TabularRepeatTask:
    dataset_regime: str
    dataset_path: str
    manifest_path: str
    repeat_id: int
    target: str
    params: dict
    random_state: int


def _predict_tabular_bundle_compat(bundle, test_df: pd.DataFrame, target: str, *, prepared_fold=None):
    try:
        return predict_tabular_bundle(bundle, test_df, target, prepared_fold=prepared_fold)
    except TypeError as exc:
        if "prepared_fold" not in str(exc):
            raise
        return predict_tabular_bundle(bundle, test_df, target)


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


def _run_svm_repeat_worker(task: TabularRepeatTask, config: dict) -> dict[str, object]:
    artifacts = ArtifactManager(config)
    dataset_df = pd.read_csv(task.dataset_path)
    manifest = pd.read_parquet(task.manifest_path)
    feature_cols = tabular_feature_columns(dataset_df, task.target)
    prediction_frames: list[pd.DataFrame] = []
    progress_events: list[dict[str, object]] = []
    repeat_manifest = manifest[manifest["repeat_id"] == task.repeat_id]
    fold_ids = sorted(repeat_manifest["fold_id"].unique().tolist())
    task_started = perf_counter()

    for fold_id in fold_ids:
        train_df = subset_from_manifest(dataset_df, manifest, repeat_id=task.repeat_id, fold_id=fold_id, split_name="train")
        test_df = subset_from_manifest(dataset_df, manifest, repeat_id=task.repeat_id, fold_id=fold_id, split_name="test")
        prepared_fold = prepare_tabular_fold(train_df=train_df, test_df=test_df, feature_cols=feature_cols, target=task.target)
        model_dir = artifacts.paths.artifact_path(
            "models",
            "tabular",
            task.dataset_regime,
            "svm",
            f"repeat_{task.repeat_id}",
            f"fold_{fold_id}",
        )
        bundle = fit_tabular_model(
            model_key="svm",
            train_df=train_df,
            feature_cols=feature_cols,
            target=task.target,
            params=task.params,
            config=config,
            model_output_dir=model_dir,
            seed=task.random_state + task.repeat_id * 100 + fold_id,
            prepared_fold=prepared_fold,
        )
        y_pred, y_prob = _predict_tabular_bundle_compat(bundle, test_df, task.target, prepared_fold=prepared_fold)
        prediction_frames.append(
            tabular_prediction_rows(
                dataset_regime=task.dataset_regime,
                population_id=task.dataset_regime,
                model_key="svm",
                target=task.target,
                repeat_id=task.repeat_id,
                fold_id=fold_id,
                test_df=test_df,
                y_pred=y_pred,
                y_prob=y_prob,
            )
        )
        progress_events.append(
            {
                "dataset_regime": task.dataset_regime,
                "model_key": "svm",
                "repeat_id": int(task.repeat_id),
                "fold_id": int(fold_id),
                "elapsed_seconds": perf_counter() - task_started,
                "model_output_dir": str(model_dir),
                "worker_type": "low_parallel",
                "task_key": f"{task.dataset_regime}::svm::repeat_{task.repeat_id}",
            }
        )
    return {
        "dataset_regime": task.dataset_regime,
        "repeat_id": int(task.repeat_id),
        "predictions": pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame(),
        "progress_events": progress_events,
    }


def run_train_tabular(config: dict) -> dict[str, str]:
    stage_name = "train_tabular"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    progress = ProgressLogger(artifacts, ("logs", "train_tabular_progress.jsonl"), stdout=False)
    progress.stage_start(stage_name, message="tabular training started")
    prediction_frames: list[pd.DataFrame] = []
    dataset_inputs: list[str] = []
    split_outputs: list[str] = []
    selected_models = selected_tabular_models(config, "train")
    dataset_regimes = selected_tabular_dataset_regimes()
    svm_policy = tabular_execution_policy("svm")

    for dataset_regime in dataset_regimes:
        dataset_path = artifacts.paths.artifact_path("datasets", "tabular", f"tabular_{dataset_regime}_labeled.csv")
        dataset_df = pd.read_csv(dataset_path)
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
        dataset_inputs.append(artifacts.relative(dataset_path))
        split_outputs.append(artifacts.relative(manifest_path))
        feature_cols = tabular_feature_columns(dataset_df, target)
        split_keys = manifest[["repeat_id", "fold_id"]].drop_duplicates().sort_values(["repeat_id", "fold_id"])
        serial_models = [model_key for model_key in selected_models if model_key != "svm" or not svm_policy.train_parallel_by_repeat]

        if serial_models:
            for row in split_keys.itertuples(index=False):
                train_df = subset_from_manifest(dataset_df, manifest, repeat_id=row.repeat_id, fold_id=row.fold_id, split_name="train")
                test_df = subset_from_manifest(dataset_df, manifest, repeat_id=row.repeat_id, fold_id=row.fold_id, split_name="test")
                prepared_fold = prepare_tabular_fold(train_df=train_df, test_df=test_df, feature_cols=feature_cols, target=target)
                for model_key in serial_models:
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
                        prepared_fold=prepared_fold,
                    )
                    y_pred, y_prob = _predict_tabular_bundle_compat(bundle, test_df, target, prepared_fold=prepared_fold)
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
                    progress.emit_event(
                        event_type="model_fit_complete",
                        stage=stage_name,
                        status="running",
                        dataset_regime=dataset_regime,
                        model_key=model_key,
                        repeat_id=int(row.repeat_id),
                        fold_id=int(row.fold_id),
                        elapsed_seconds=perf_counter() - start,
                        model_output_dir=str(model_dir),
                        worker_type="serial",
                        task_key=f"{dataset_regime}::{model_key}::repeat_{row.repeat_id}::fold_{row.fold_id}",
                    )

        if "svm" in selected_models and svm_policy.train_parallel_by_repeat:
            svm_params = _tabular_params(config, artifacts, dataset_regime, "svm")
            repeat_ids = sorted(manifest["repeat_id"].unique().tolist())
            tasks = [
                TabularRepeatTask(
                    dataset_regime=dataset_regime,
                    dataset_path=str(dataset_path),
                    manifest_path=str(manifest_path),
                    repeat_id=int(repeat_id),
                    target=target,
                    params=svm_params,
                    random_state=int(config["splits"]["random_state"]),
                )
                for repeat_id in repeat_ids
            ]
            with ProcessPoolExecutor(max_workers=min(8, len(tasks))) as executor:
                futures = {executor.submit(_run_svm_repeat_worker, task, config): task for task in tasks}
                for future in as_completed(futures):
                    result = future.result()
                    if not result["predictions"].empty:
                        prediction_frames.append(result["predictions"])
                    for event in result["progress_events"]:
                        progress.emit_event(
                            event_type="model_fit_complete",
                            stage=stage_name,
                            status="running",
                            **event,
                        )

        artifacts.write_manifest(
            f"train_tabular_{dataset_regime}",
            ["manifests", f"train_tabular_{dataset_regime}.json"],
            inputs=[artifacts.relative(dataset_path)],
            outputs=[artifacts.relative(manifest_path)],
            metadata={"n_models": len(selected_models)},
            stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
            wall_time_seconds=perf_counter() - start,
        )

    raw_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    partition_path = write_prediction_partition("tabular", raw_predictions, artifacts)
    out_path = materialize_raw_predictions(artifacts)
    artifacts.write_manifest(
        "train_tabular",
        ["manifests", "train_tabular.json"],
        inputs=dataset_inputs,
        outputs=split_outputs + [artifacts.relative(partition_path), artifacts.relative(out_path)],
        metadata={"dataset_regimes": dataset_regimes, "n_models": len(selected_models)},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    progress.stage_end(stage_name, wall_time_seconds=perf_counter() - start, message="tabular training finished")
    return {"partition": str(partition_path), "predictions": str(out_path)}


def run_train_sequence(config: dict) -> dict[str, str]:
    stage_name = "train_sequence"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    progress = ProgressLogger(artifacts, ("logs", "train_sequence_progress.jsonl"), stdout=False)
    progress.stage_start(stage_name, message="sequence training started")
    if not config["models"]["sequence_enabled"]:
        progress.stage_end(stage_name, wall_time_seconds=perf_counter() - start, message="sequence training skipped; no sequence models enabled")
        return {}
    sequence_path = artifacts.paths.artifact_path("datasets", "sequence", "lstm_trainable.pkl")
    if not sequence_path.exists():
        progress.stage_end(stage_name, wall_time_seconds=perf_counter() - start, message="sequence training skipped; no sequence dataset")
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
                progress_callback=lambda **payload: progress.emit_event(
                    event_type="validation_checkpoint",
                    stage=stage_name,
                    status="running",
                    model_key=model_key,
                    repeat_id=int(row.repeat_id),
                    fold_id=int(row.fold_id),
                    **payload,
                ),
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
        progress.stage_end(stage_name, wall_time_seconds=perf_counter() - start, message="sequence training skipped; no predictions produced")
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
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    progress.stage_end(stage_name, wall_time_seconds=perf_counter() - start, message="sequence training finished")
    return {"partition": str(partition_path), "predictions": str(out_path)}
