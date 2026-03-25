from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from inspire_aki.config import config_hash
from inspire_aki.datasets.splits import build_hpo_split_manifest
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.manifest import build_manifest
from inspire_aki.io.progress import ProgressLogger
from inspire_aki.models.hpo import tune_sequence_dataset, tune_tabular_dataset
from inspire_aki.models.tabular import (
    selected_tabular_dataset_regimes,
    selected_tabular_models,
    tabular_execution_policy,
)
from inspire_aki.runtime import build_stage_runtime_plan


@dataclass(frozen=True)
class TabularStudySpec:
    dataset_regime: str
    model_key: str
    dataset_path: str
    manifest_path: str


@dataclass(frozen=True)
class TabularStudyPaths:
    best_params: Path
    trials: Path
    manifest: Path


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.stem}.{os.getpid()}.tmp{path.suffix}"
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    temp_path.replace(path)
    return path


def _atomic_write_dataframe(path: Path, df: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.parent / f".{path.stem}.{os.getpid()}.tmp{path.suffix}"
    if path.suffix == ".csv":
        df.to_csv(temp_path, index=False)
    else:
        df.to_parquet(temp_path, index=False)
    temp_path.replace(path)
    return path


def _tabular_study_paths(artifacts: ArtifactManager, dataset_regime: str, model_key: str) -> TabularStudyPaths:
    study_stem = f"{dataset_regime}__{model_key}"
    return TabularStudyPaths(
        best_params=artifacts.resolve("tuning", "tabular_studies", f"{study_stem}_best_params.json"),
        trials=artifacts.resolve("tuning", "tabular_studies", f"{study_stem}_trials.parquet"),
        manifest=artifacts.resolve("manifests", f"tune_tabular_{study_stem}.json"),
    )


def _write_tabular_study_outputs(
    *,
    artifacts: ArtifactManager,
    config: dict,
    spec: TabularStudySpec,
    best_params: dict[str, Any],
    trials_df: pd.DataFrame,
    wall_time_seconds: float,
) -> TabularStudyPaths:
    paths = _tabular_study_paths(artifacts, spec.dataset_regime, spec.model_key)
    _atomic_write_json(paths.best_params, best_params)
    _atomic_write_dataframe(paths.trials, trials_df)
    stage_runtime_plan = build_stage_runtime_plan(config, "tune_tabular").as_dict()
    manifest_payload = build_manifest(
        stage=f"tune_tabular_{spec.dataset_regime}__{spec.model_key}",
        repo_root=artifacts.paths.repo_root,
        config=config,
        inputs=[artifacts.relative(Path(spec.dataset_path))],
        outputs=[
            artifacts.relative(Path(spec.manifest_path)),
            artifacts.relative(paths.best_params),
            artifacts.relative(paths.trials),
        ],
        metadata={
            "dataset_regime": spec.dataset_regime,
            "model_key": spec.model_key,
            "n_trials": int(len(trials_df)),
        },
        stage_runtime_plan=stage_runtime_plan,
        wall_time_seconds=wall_time_seconds,
    )
    _atomic_write_json(paths.manifest, manifest_payload)
    return paths


def _read_completed_tabular_study(
    *,
    artifacts: ArtifactManager,
    config: dict,
    spec: TabularStudySpec,
) -> tuple[dict[str, Any], pd.DataFrame, TabularStudyPaths] | None:
    paths = _tabular_study_paths(artifacts, spec.dataset_regime, spec.model_key)
    if not (paths.best_params.exists() and paths.trials.exists() and paths.manifest.exists()):
        return None
    with paths.manifest.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("config_hash") != config_hash(config):
        return None
    with paths.best_params.open("r", encoding="utf-8") as handle:
        best_params = json.load(handle)
    trials_df = pd.read_parquet(paths.trials)
    return best_params, trials_df, paths


def _run_tabular_study_worker(spec: TabularStudySpec, config: dict) -> dict[str, Any]:
    artifacts = ArtifactManager(config)
    progress = ProgressLogger(artifacts, ("logs", "tune_tabular_progress.jsonl"), stdout=False)
    dataset_df = pd.read_csv(spec.dataset_path)
    manifest = pd.read_parquet(spec.manifest_path)
    study_config = json.loads(json.dumps(config, default=str))
    study_config["models"]["tabular_hpo_enabled"] = [spec.model_key]
    started = perf_counter()
    results, trials_df = tune_tabular_dataset(
        dataset_df,
        spec.dataset_regime,
        manifest,
        study_config,
        progress_callback=lambda **payload: progress.emit_event(
            event_type="optuna_trial",
            stage="tune_tabular",
            status="running",
            resume_status="completed_now",
            **payload,
        ),
    )
    best_params = results.get(spec.model_key, {})
    paths = _write_tabular_study_outputs(
        artifacts=artifacts,
        config=config,
        spec=spec,
        best_params=best_params,
        trials_df=trials_df,
        wall_time_seconds=perf_counter() - started,
    )
    return {
        "dataset_regime": spec.dataset_regime,
        "model_key": spec.model_key,
        "best_params": best_params,
        "trials_df": trials_df,
        "best_params_path": str(paths.best_params),
        "trials_path": str(paths.trials),
        "manifest_path": str(paths.manifest),
        "wall_time_seconds": perf_counter() - started,
    }


def _run_tabular_study(spec: TabularStudySpec, config: dict) -> dict[str, Any]:
    return _run_tabular_study_worker(spec, config)


def run_tune_tabular(config: dict) -> dict[str, str]:
    stage_name = "tune_tabular"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    progress = ProgressLogger(artifacts, ("logs", "tune_tabular_progress.jsonl"), stdout=False)
    progress.stage_start(stage_name, message="tabular HPO started")
    best_params: dict[str, dict[str, Any]] = {}
    trial_frames: list[pd.DataFrame] = []
    per_dataset_records: dict[str, dict[str, Any]] = {}
    pending_by_model: dict[str, list[TabularStudySpec]] = {}
    enabled_models = selected_tabular_models(config, "tune")
    dataset_regimes = selected_tabular_dataset_regimes()

    for dataset_regime in dataset_regimes:
        dataset_path = artifacts.paths.artifact_path("datasets", "tabular", f"tabular_{dataset_regime}_labeled.csv")
        dataset_df = pd.read_csv(dataset_path)
        manifest = build_hpo_split_manifest(
            dataset_df,
            target=config["models"]["target"],
            dataset_regime=dataset_regime,
            population_id=dataset_regime,
            random_state=config["splits"]["random_state"],
            holdout_fraction=config["splits"]["holdout_fraction"],
            validation_fraction_within_train=config["splits"]["hpo_validation_fraction_within_train"],
        )
        manifest_path = artifacts.write_dataframe(manifest, "datasets", "splits", f"hpo_{dataset_regime}.parquet")
        per_dataset_records[dataset_regime] = {
            "dataset_path": dataset_path,
            "manifest_path": manifest_path,
            "trial_count": 0,
            "models": [],
        }
        for model_key in enabled_models:
            spec = TabularStudySpec(
                dataset_regime=dataset_regime,
                model_key=model_key,
                dataset_path=str(dataset_path),
                manifest_path=str(manifest_path),
            )
            completed = _read_completed_tabular_study(artifacts=artifacts, config=config, spec=spec)
            if completed is not None:
                params, trials_df, paths = completed
                best_params.setdefault(dataset_regime, {})[model_key] = params
                if not trials_df.empty:
                    trial_frames.append(trials_df)
                    per_dataset_records[dataset_regime]["trial_count"] += int(len(trials_df))
                per_dataset_records[dataset_regime]["models"].append(model_key)
                progress.emit_event(
                    event_type="study_status",
                    stage=stage_name,
                    status="skipped",
                    dataset_regime=dataset_regime,
                    model_key=model_key,
                    study_key=f"{dataset_regime}::{model_key}",
                    resume_status="resumed_skipped",
                    output_best_params_path=str(paths.best_params),
                    output_trials_path=str(paths.trials),
                )
                continue
            pending_by_model.setdefault(model_key, []).append(spec)

    for model_key in enabled_models:
        specs = pending_by_model.get(model_key, [])
        if not specs:
            continue
        policy = tabular_execution_policy(model_key)
        if policy.hpo_parallel_by_regime and len(specs) > 1:
            with ProcessPoolExecutor(max_workers=min(3, len(specs))) as executor:
                futures = {executor.submit(_run_tabular_study_worker, spec, config): spec for spec in specs}
                for future in as_completed(futures):
                    result = future.result()
                    dataset_regime = result["dataset_regime"]
                    best_params.setdefault(dataset_regime, {})[model_key] = result["best_params"]
                    trials_df = result["trials_df"]
                    if not trials_df.empty:
                        trial_frames.append(trials_df)
                        per_dataset_records[dataset_regime]["trial_count"] += int(len(trials_df))
                    per_dataset_records[dataset_regime]["models"].append(model_key)
                    progress.emit_event(
                        event_type="study_status",
                        stage=stage_name,
                        status="completed",
                        dataset_regime=dataset_regime,
                        model_key=model_key,
                        study_key=f"{dataset_regime}::{model_key}",
                        resume_status="completed_now",
                        elapsed_seconds=result["wall_time_seconds"],
                        output_best_params_path=result["best_params_path"],
                        output_trials_path=result["trials_path"],
                    )
        else:
            for spec in specs:
                result = _run_tabular_study(spec, config)
                best_params.setdefault(spec.dataset_regime, {})[model_key] = result["best_params"]
                trials_df = result["trials_df"]
                if not trials_df.empty:
                    trial_frames.append(trials_df)
                    per_dataset_records[spec.dataset_regime]["trial_count"] += int(len(trials_df))
                per_dataset_records[spec.dataset_regime]["models"].append(model_key)
                progress.emit_event(
                    event_type="study_status",
                    stage=stage_name,
                    status="completed",
                    dataset_regime=spec.dataset_regime,
                    model_key=model_key,
                    study_key=f"{spec.dataset_regime}::{model_key}",
                    resume_status="completed_now",
                    elapsed_seconds=result["wall_time_seconds"],
                    output_best_params_path=result["best_params_path"],
                    output_trials_path=result["trials_path"],
                )

    best_path = artifacts.write_json(best_params, "tuning", "tabular_best_params.json")
    outputs = {"best_params": str(best_path)}
    output_paths = [artifacts.relative(best_path)]
    if trial_frames:
        trials_path = artifacts.write_dataframe(pd.concat(trial_frames, ignore_index=True), "tuning", "tabular_trials.parquet")
        outputs["trials"] = str(trials_path)
        output_paths.append(artifacts.relative(trials_path))
    stage_runtime_plan = build_stage_runtime_plan(config, stage_name).as_dict()
    dataset_inputs: list[str] = []
    split_outputs: list[str] = []
    for dataset_regime in dataset_regimes:
        record = per_dataset_records[dataset_regime]
        dataset_inputs.append(artifacts.relative(record["dataset_path"]))
        split_outputs.append(artifacts.relative(record["manifest_path"]))
        manifest_outputs = [artifacts.relative(record["manifest_path"])] + output_paths
        artifacts.write_manifest(
            f"tune_tabular_{dataset_regime}",
            ["manifests", f"tune_tabular_{dataset_regime}.json"],
            inputs=[artifacts.relative(record["dataset_path"])],
            outputs=manifest_outputs,
            metadata={"n_trials": record["trial_count"], "models": record["models"]},
            stage_runtime_plan=stage_runtime_plan,
            wall_time_seconds=perf_counter() - start,
        )
    artifacts.write_manifest(
        "tune_tabular",
        ["manifests", "tune_tabular.json"],
        inputs=dataset_inputs,
        outputs=split_outputs + output_paths,
        metadata={"dataset_regimes": dataset_regimes, "models": best_params},
        stage_runtime_plan=stage_runtime_plan,
        wall_time_seconds=perf_counter() - start,
    )
    progress.stage_end(stage_name, wall_time_seconds=perf_counter() - start, message="tabular HPO finished")
    return outputs


def run_tune_sequence(config: dict) -> dict[str, str]:
    stage_name = "tune_sequence"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    progress = ProgressLogger(artifacts, ("logs", "tune_sequence_progress.jsonl"), stdout=False)
    progress.stage_start(stage_name, message="sequence HPO started")
    sequence_path = artifacts.paths.artifact_path("datasets", "sequence", "lstm_trainable.pkl")
    if not sequence_path.exists():
        progress.stage_end(stage_name, wall_time_seconds=perf_counter() - start, message="sequence HPO skipped; no sequence dataset")
        return {}
    sequence_df = artifacts.read_pickle("datasets", "sequence", "lstm_trainable.pkl")
    manifest = build_hpo_split_manifest(
        sequence_df,
        target=config["models"]["target"],
        dataset_regime="sequence",
        population_id="sequence_common",
        random_state=config["splits"]["random_state"],
        holdout_fraction=config["splits"]["holdout_fraction"],
        validation_fraction_within_train=config["splits"]["hpo_validation_fraction_within_train"],
    )
    manifest_path = artifacts.write_dataframe(manifest, "datasets", "splits", "hpo_sequence.parquet")
    params, trials_df = tune_sequence_dataset(
        sequence_df,
        manifest,
        config,
        progress_callback=lambda **payload: progress.emit_event(
            event_type="optuna_trial",
            stage=stage_name,
            status="running",
            **payload,
        ),
    )
    best_path = artifacts.write_json(params, "tuning", "sequence_best_params.json")
    outputs = {"best_params": str(best_path)}
    if not trials_df.empty:
        trials_path = artifacts.write_dataframe(trials_df, "tuning", "sequence_trials.parquet")
        outputs["trials"] = str(trials_path)
    artifacts.write_manifest(
        "tune_sequence",
        ["manifests", "tune_sequence.json"],
        inputs=[artifacts.relative(sequence_path)],
        outputs=[artifacts.relative(manifest_path), artifacts.relative(best_path)]
        + ([artifacts.relative(artifacts.paths.artifact_path("tuning", "sequence_trials.parquet"))] if "trials" in outputs else []),
        metadata={"models": list(params)},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    progress.stage_end(stage_name, wall_time_seconds=perf_counter() - start, message="sequence HPO finished")
    return outputs
