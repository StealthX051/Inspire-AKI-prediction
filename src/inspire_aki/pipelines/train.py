from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import pandas as pd
from inspire_aki.datasets.splits import build_bootstrap_split_manifest, grouped_manifest_to_training_manifest, subset_from_manifest
from inspire_aki.evaluation.split_manager import evaluation_runs, subset_generated_manifest
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.predictions import materialize_raw_predictions, write_prediction_partition
from inspire_aki.io.progress import ProgressLogger
from inspire_aki.models.registry import sequence_dataset_for_model, sequence_population_for_model
from inspire_aki.models.sequence import (
    fit_sequence_model,
    predict_sequence_bundle,
    raw_prediction_rows as sequence_prediction_rows,
    sequence_feature_columns,
)
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


@dataclass(frozen=True)
class TabularGeneratedRunTask:
    dataset_regime: str
    dataset_path: str
    manifest_path: str
    run_id: int
    repeat_id: int
    fold_id: int
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


def _tabular_params(config: dict, artifacts: ArtifactManager, dataset_regime: str, model_key: str, *, run_id: int = 0) -> dict:
    tuning_path = artifacts.paths.artifact_path("tuning", "tabular_best_params.json")
    if tuning_path.exists():
        tuning = artifacts.read_json("tuning", "tabular_best_params.json")
        scoped = tuning.get(f"run_{run_id}", tuning)
        if dataset_regime in scoped and model_key in scoped[dataset_regime]:
            return scoped[dataset_regime][model_key]
    if model_key in config["models"]["tabular_hpo_params"].get(dataset_regime, {}):
        return config["models"]["tabular_hpo_params"][dataset_regime][model_key]
    return config["models"]["tabular_defaults"].get(model_key, {})


def _sequence_params(config: dict, artifacts: ArtifactManager, model_key: str, *, run_id: int = 0) -> dict:
    params = dict(config["models"]["sequence_defaults"])
    tuning_path = artifacts.paths.artifact_path("tuning", "sequence_best_params.json")
    if tuning_path.exists():
        tuned = artifacts.read_json("tuning", "sequence_best_params.json")
        scoped = tuned.get(f"run_{run_id}", tuned)
        if model_key in scoped:
            tuned_params = dict(scoped[model_key])
            if "lr" in tuned_params:
                tuned_params["learning_rate"] = tuned_params.pop("lr")
            params.update(tuned_params)
    else:
        configured = dict(config["models"]["sequence_hpo_params"].get(model_key, {}))
        if "lr" in configured:
            configured["learning_rate"] = configured.pop("lr")
        params.update(configured)
    return params


def _precomputed_manifest_path(artifacts: ArtifactManager, config: dict, dataset_regime: str) -> Path | None:
    evaluation_mode = config.get("evaluation_mode", "legacy_repeated_cv")
    if evaluation_mode == "legacy_repeated_cv":
        return None
    return artifacts.paths.artifact_path("datasets", "splits", f"{evaluation_mode}_{dataset_regime}.parquet")


def _load_grouped_evaluation_manifest(artifacts: ArtifactManager, config: dict, dataset_regime: str) -> tuple[pd.DataFrame, Path]:
    manifest_path = _precomputed_manifest_path(artifacts, config, dataset_regime)
    if manifest_path is None:
        raise ValueError("Grouped evaluation manifest requested for legacy_repeated_cv mode.")
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Expected grouped evaluation manifest was not found: {manifest_path}. "
            "Run `inspire-aki evaluate generate --config ...` before training."
        )
    return pd.read_parquet(manifest_path), manifest_path


def _tabular_training_feature_columns(dataset_df: pd.DataFrame, target: str, model_key: str) -> list[str]:
    if model_key == "gs_aki_rule":
        if "gs_aki_count" not in dataset_df.columns:
            raise ValueError("gs_aki_rule training requires a gs_aki_count column in the dedicated labeled dataset.")
        return ["gs_aki_count"]
    return tabular_feature_columns(dataset_df, target)


def _tabular_model_dataset(
    *,
    artifacts: ArtifactManager,
    dataset_regime: str,
    model_key: str,
    default_dataset_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Path]:
    if model_key == "gs_aki_rule":
        if dataset_regime != "preop":
            raise ValueError("gs_aki_rule is only supported for the preop dataset regime.")
        dataset_path = artifacts.paths.artifact_path("datasets", "tabular", "tabular_gs_aki_labeled.csv")
        return pd.read_csv(dataset_path), dataset_path
    dataset_path = artifacts.paths.artifact_path("datasets", "tabular", f"tabular_{dataset_regime}_labeled.csv")
    return default_dataset_df, dataset_path


def _load_or_build_training_manifest(
    *,
    artifacts: ArtifactManager,
    config: dict,
    dataset_df: pd.DataFrame,
    target: str,
    dataset_regime: str,
    population_id: str,
) -> tuple[pd.DataFrame, Path, bool]:
    precomputed_path = _precomputed_manifest_path(artifacts, config, dataset_regime)
    if precomputed_path is not None:
        if not precomputed_path.exists():
            raise FileNotFoundError(
                f"Expected grouped evaluation manifest was not found: {precomputed_path}. "
                "Run `inspire-aki evaluate generate --config ...` before training."
            )
        manifest = pd.read_parquet(precomputed_path)
        return grouped_manifest_to_training_manifest(manifest), precomputed_path, False

    manifest = build_bootstrap_split_manifest(
        dataset_df,
        target=target,
        dataset_regime=dataset_regime,
        population_id=population_id,
        random_state=config["splits"]["random_state"],
        n_iterations=config["splits"]["n_bootstrap_iterations"],
        n_cv_folds=config["splits"]["n_cv_folds"],
        use_bootstrapping=config["splits"]["use_bootstrapping"],
    )
    manifest_path = artifacts.write_dataframe(manifest, "datasets", "splits", f"bootstrap_{dataset_regime}.parquet")
    return manifest, manifest_path, True


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


def _run_svm_generated_run_worker(task: TabularGeneratedRunTask, config: dict) -> dict[str, object]:
    artifacts = ArtifactManager(config)
    dataset_df = pd.read_csv(task.dataset_path)
    manifest = pd.read_parquet(task.manifest_path)
    feature_cols = tabular_feature_columns(dataset_df, task.target)
    task_started = perf_counter()
    train_df = subset_generated_manifest(
        dataset_df,
        manifest,
        split_name="train",
        run_id=task.run_id,
    )
    test_df = subset_generated_manifest(
        dataset_df,
        manifest,
        split_name="test",
        run_id=task.run_id,
    )
    prepared_fold = prepare_tabular_fold(train_df=train_df, test_df=test_df, feature_cols=feature_cols, target=task.target)
    model_dir = artifacts.paths.artifact_path(
        "models",
        "tabular",
        task.dataset_regime,
        "svm",
        f"repeat_{task.repeat_id}",
        f"fold_{task.fold_id}",
    )
    bundle = fit_tabular_model(
        model_key="svm",
        train_df=train_df,
        feature_cols=feature_cols,
        target=task.target,
        params=task.params,
        config=config,
        model_output_dir=model_dir,
        seed=task.random_state + task.repeat_id * 100 + task.fold_id,
        prepared_fold=prepared_fold,
    )
    y_pred, y_prob = _predict_tabular_bundle_compat(bundle, test_df, task.target, prepared_fold=prepared_fold)
    prediction_frame = tabular_prediction_rows(
        dataset_regime=task.dataset_regime,
        population_id=task.dataset_regime,
        model_key="svm",
        target=task.target,
        repeat_id=task.repeat_id,
        fold_id=task.fold_id,
        test_df=test_df,
        y_pred=y_pred,
        y_prob=y_prob,
    )
    progress_event = {
        "dataset_regime": task.dataset_regime,
        "model_key": "svm",
        "run_id": int(task.run_id),
        "repeat_id": int(task.repeat_id),
        "fold_id": int(task.fold_id),
        "elapsed_seconds": perf_counter() - task_started,
        "model_output_dir": str(model_dir),
        "worker_type": "low_parallel",
        "task_key": f"{task.dataset_regime}::svm::run_{task.run_id}",
    }
    return {
        "dataset_regime": task.dataset_regime,
        "run_id": int(task.run_id),
        "repeat_id": int(task.repeat_id),
        "predictions": prediction_frame,
        "progress_events": [progress_event],
    }


def run_train_tabular(config: dict) -> dict[str, str]:
    stage_name = "train_tabular"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    progress = ProgressLogger(artifacts, ("logs", "train_tabular_progress.jsonl"), stdout=False)
    progress.stage_start(stage_name, message="tabular training started")
    prediction_frames: list[pd.DataFrame] = []
    dataset_inputs: list[str] = []
    split_inputs: list[str] = []
    split_outputs: list[str] = []
    selected_models = selected_tabular_models(config, "train")
    dataset_regimes = selected_tabular_dataset_regimes()
    svm_policy = tabular_execution_policy("svm")
    evaluation_mode = config.get("evaluation_mode", "legacy_repeated_cv")

    if evaluation_mode != "legacy_repeated_cv":
        for dataset_regime in dataset_regimes:
            dataset_path = artifacts.paths.artifact_path("datasets", "tabular", f"tabular_{dataset_regime}_labeled.csv")
            dataset_df = pd.read_csv(dataset_path)
            manifest, manifest_path = _load_grouped_evaluation_manifest(artifacts, config, dataset_regime)
            target = config["models"]["target"]
            dataset_inputs.append(artifacts.relative(dataset_path))
            split_inputs.append(artifacts.relative(manifest_path))
            split_runs = evaluation_runs(manifest)
            dataset_models = [model_key for model_key in selected_models if model_key != "gs_aki_rule" or dataset_regime == "preop"]
            serial_models = [model_key for model_key in dataset_models if model_key != "svm" or not svm_policy.train_parallel_by_repeat]

            if serial_models:
                for run in split_runs:
                    for model_key in serial_models:
                        model_dataset_df, model_dataset_path = _tabular_model_dataset(
                            artifacts=artifacts,
                            dataset_regime=dataset_regime,
                            model_key=model_key,
                            default_dataset_df=dataset_df,
                        )
                        if model_key == "gs_aki_rule":
                            model_input = artifacts.relative(model_dataset_path)
                            if model_input not in dataset_inputs:
                                dataset_inputs.append(model_input)
                        feature_cols = _tabular_training_feature_columns(model_dataset_df, target, model_key)
                        if model_key == "asa_rule" and "asa" not in feature_cols:
                            continue
                        train_df = subset_generated_manifest(model_dataset_df, manifest, split_name="train", run_id=run.run_id)
                        test_df = subset_generated_manifest(model_dataset_df, manifest, split_name="test", run_id=run.run_id)
                        prepared_fold = prepare_tabular_fold(
                            train_df=train_df,
                            test_df=test_df,
                            feature_cols=feature_cols,
                            target=target,
                        )
                        model_dir = artifacts.paths.artifact_path(
                            "models",
                            "tabular",
                            dataset_regime,
                            model_key,
                            f"repeat_{run.repeat_id}",
                            f"fold_{run.fold_id}",
                        )
                        params = _tabular_params(config, artifacts, dataset_regime, model_key, run_id=int(run.run_id))
                        bundle = fit_tabular_model(
                            model_key=model_key,
                            train_df=train_df,
                            feature_cols=feature_cols,
                            target=target,
                            params=params,
                            config=config,
                            model_output_dir=model_dir,
                            seed=config["splits"]["random_state"] + run.repeat_id * 100 + run.fold_id,
                            prepared_fold=prepared_fold,
                        )
                        y_pred, y_prob = _predict_tabular_bundle_compat(bundle, test_df, target, prepared_fold=prepared_fold)
                        prediction_frames.append(
                            tabular_prediction_rows(
                                dataset_regime=dataset_regime,
                                population_id=dataset_regime,
                                model_key=model_key,
                                target=target,
                                repeat_id=run.repeat_id,
                                fold_id=run.fold_id,
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
                            run_id=int(run.run_id),
                            repeat_id=int(run.repeat_id),
                            fold_id=int(run.fold_id),
                            elapsed_seconds=perf_counter() - start,
                            model_output_dir=str(model_dir),
                            worker_type="serial",
                            task_key=f"{dataset_regime}::{model_key}::run_{run.run_id}",
                        )

            if "svm" in dataset_models and svm_policy.train_parallel_by_repeat:
                tasks = [
                    TabularGeneratedRunTask(
                        dataset_regime=dataset_regime,
                        dataset_path=str(dataset_path),
                        manifest_path=str(manifest_path),
                        run_id=int(run.run_id),
                        repeat_id=int(run.repeat_id),
                        fold_id=int(run.fold_id),
                        target=target,
                        params=_tabular_params(config, artifacts, dataset_regime, "svm", run_id=int(run.run_id)),
                        random_state=int(config["splits"]["random_state"]),
                    )
                    for run in split_runs
                ]
                with ProcessPoolExecutor(max_workers=min(8, len(tasks))) as executor:
                    futures = {executor.submit(_run_svm_generated_run_worker, task, config): task for task in tasks}
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
                inputs=[artifacts.relative(dataset_path), artifacts.relative(manifest_path)]
                + (
                    [artifacts.relative(artifacts.paths.artifact_path("datasets", "tabular", "tabular_gs_aki_labeled.csv"))]
                    if "gs_aki_rule" in dataset_models
                    else []
                ),
                outputs=[],
                metadata={"n_models": len(dataset_models)},
                stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
                wall_time_seconds=perf_counter() - start,
            )

        raw_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
        partition_path = write_prediction_partition("tabular", raw_predictions, artifacts)
        out_path = materialize_raw_predictions(artifacts)
        artifacts.write_manifest(
            "train_tabular",
            ["manifests", "train_tabular.json"],
            inputs=dataset_inputs + split_inputs,
            outputs=[artifacts.relative(partition_path), artifacts.relative(out_path)],
            metadata={"dataset_regimes": dataset_regimes, "n_models": len(selected_models)},
            stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
            wall_time_seconds=perf_counter() - start,
        )
        progress.stage_end(stage_name, wall_time_seconds=perf_counter() - start, message="tabular training finished")
        return {"partition": str(partition_path), "predictions": str(out_path)}

    for dataset_regime in dataset_regimes:
        dataset_path = artifacts.paths.artifact_path("datasets", "tabular", f"tabular_{dataset_regime}_labeled.csv")
        dataset_df = pd.read_csv(dataset_path)
        target = config["models"]["target"]
        manifest, manifest_path, manifest_generated = _load_or_build_training_manifest(
            artifacts=artifacts,
            config=config,
            dataset_df=dataset_df,
            target=target,
            dataset_regime=dataset_regime,
            population_id=dataset_regime,
        )
        dataset_inputs.append(artifacts.relative(dataset_path))
        if manifest_generated:
            split_outputs.append(artifacts.relative(manifest_path))
        else:
            split_inputs.append(artifacts.relative(manifest_path))
        split_keys = manifest[["repeat_id", "fold_id"]].drop_duplicates().sort_values(["repeat_id", "fold_id"])
        dataset_models = [model_key for model_key in selected_models if model_key != "gs_aki_rule" or dataset_regime == "preop"]
        serial_models = [model_key for model_key in dataset_models if model_key != "svm" or not svm_policy.train_parallel_by_repeat]

        if serial_models:
            for row in split_keys.itertuples(index=False):
                for model_key in serial_models:
                    model_dataset_df, model_dataset_path = _tabular_model_dataset(
                        artifacts=artifacts,
                        dataset_regime=dataset_regime,
                        model_key=model_key,
                        default_dataset_df=dataset_df,
                    )
                    if model_key == "gs_aki_rule":
                        model_input = artifacts.relative(model_dataset_path)
                        if model_input not in dataset_inputs:
                            dataset_inputs.append(model_input)
                    feature_cols = _tabular_training_feature_columns(model_dataset_df, target, model_key)
                    if model_key == "asa_rule" and "asa" not in feature_cols:
                        continue
                    train_df = subset_from_manifest(
                        model_dataset_df,
                        manifest,
                        repeat_id=row.repeat_id,
                        fold_id=row.fold_id,
                        split_name="train",
                    )
                    test_df = subset_from_manifest(
                        model_dataset_df,
                        manifest,
                        repeat_id=row.repeat_id,
                        fold_id=row.fold_id,
                        split_name="test",
                    )
                    prepared_fold = prepare_tabular_fold(
                        train_df=train_df,
                        test_df=test_df,
                        feature_cols=feature_cols,
                        target=target,
                    )
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

        if "svm" in dataset_models and svm_policy.train_parallel_by_repeat:
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
            inputs=[artifacts.relative(dataset_path)]
            + ([] if manifest_generated else [artifacts.relative(manifest_path)])
            + (
                [artifacts.relative(artifacts.paths.artifact_path("datasets", "tabular", "tabular_gs_aki_labeled.csv"))]
                if "gs_aki_rule" in dataset_models
                else []
            ),
            outputs=([artifacts.relative(manifest_path)] if manifest_generated else []),
            metadata={"n_models": len(dataset_models)},
            stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
            wall_time_seconds=perf_counter() - start,
        )

    raw_predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    partition_path = write_prediction_partition("tabular", raw_predictions, artifacts)
    out_path = materialize_raw_predictions(artifacts)
    artifacts.write_manifest(
        "train_tabular",
        ["manifests", "train_tabular.json"],
        inputs=dataset_inputs + split_inputs,
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
    evaluation_mode = config.get("evaluation_mode", "legacy_repeated_cv")
    if evaluation_mode != "legacy_repeated_cv":
        manifest, manifest_path = _load_grouped_evaluation_manifest(artifacts, config, "sequence")
        split_runs = evaluation_runs(manifest)
        feature_cols_tab = sequence_feature_columns(sequence_df, target)
        prediction_frames: list[pd.DataFrame] = []

        for model_key in config["models"]["sequence_enabled"]:
            dataset_regime = sequence_dataset_for_model(model_key)
            population_id = sequence_population_for_model(model_key)
            for run in split_runs:
                params = _sequence_params(config, artifacts, model_key, run_id=int(run.run_id))
                train_df = subset_generated_manifest(sequence_df, manifest, split_name="train", run_id=run.run_id)
                test_df = subset_generated_manifest(sequence_df, manifest, split_name="test", run_id=run.run_id)
                model_dir = artifacts.paths.artifact_path("models", "sequence", model_key, f"repeat_{run.repeat_id}", f"fold_{run.fold_id}")
                bundle = fit_sequence_model(
                    model_key=model_key,
                    train_df=train_df,
                    feature_cols_tab=feature_cols_tab,
                    target=target,
                    params=params,
                    config=config,
                    model_output_dir=model_dir,
                    seed=config["splits"]["random_state"] + run.repeat_id * 100 + run.fold_id,
                    progress_callback=lambda _run=run, **payload: progress.emit_event(
                        event_type="validation_checkpoint",
                        stage=stage_name,
                        status="running",
                        model_key=model_key,
                        run_id=int(_run.run_id),
                        repeat_id=int(_run.repeat_id),
                        fold_id=int(_run.fold_id),
                        **payload,
                    ),
                )
                y_pred, y_prob = predict_sequence_bundle(bundle, test_df)
                prediction_frames.append(
                    sequence_prediction_rows(
                        dataset_regime=dataset_regime,
                        population_id=population_id,
                        model_key=model_key,
                        target=target,
                        repeat_id=run.repeat_id,
                        fold_id=run.fold_id,
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
            inputs=[artifacts.relative(sequence_path), artifacts.relative(manifest_path)],
            outputs=[artifacts.relative(partition_path), artifacts.relative(out_path)],
            metadata={"n_models": len(config["models"]["sequence_enabled"])},
            stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
            wall_time_seconds=perf_counter() - start,
        )
        progress.stage_end(stage_name, wall_time_seconds=perf_counter() - start, message="sequence training finished")
        return {"partition": str(partition_path), "predictions": str(out_path)}

    manifest, manifest_path, manifest_generated = _load_or_build_training_manifest(
        artifacts=artifacts,
        config=config,
        dataset_df=sequence_df,
        target=target,
        dataset_regime="sequence",
        population_id="sequence_common",
    )
    split_keys = manifest[["repeat_id", "fold_id"]].drop_duplicates().sort_values(["repeat_id", "fold_id"])
    feature_cols_tab = sequence_feature_columns(sequence_df, target)
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
                target=target,
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
                    target=target,
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
        inputs=[artifacts.relative(sequence_path)] + ([] if manifest_generated else [artifacts.relative(manifest_path)]),
        outputs=([artifacts.relative(manifest_path)] if manifest_generated else []) + [artifacts.relative(partition_path), artifacts.relative(out_path)],
        metadata={"n_models": len(config["models"]["sequence_enabled"])},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    progress.stage_end(stage_name, wall_time_seconds=perf_counter() - start, message="sequence training finished")
    return {"partition": str(partition_path), "predictions": str(out_path)}
