from __future__ import annotations

import pandas as pd

from inspire_aki.datasets.splits import build_hpo_split_manifest
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.models.hpo import tune_sequence_dataset, tune_tabular_dataset


def run_tune_tabular(config: dict) -> dict[str, str]:
    artifacts = ArtifactManager(config)
    best_params: dict[str, dict] = {}
    trial_frames: list[pd.DataFrame] = []
    for dataset_regime in ["preop", "intraop", "combined"]:
        dataset_df = pd.read_csv(artifacts.paths.artifact_path("datasets", "tabular", f"tabular_{dataset_regime}_labeled.csv"))
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
        params, trials_df = tune_tabular_dataset(dataset_df, dataset_regime, manifest, config)
        best_params[dataset_regime] = params
        if not trials_df.empty:
            trial_frames.append(trials_df)
        artifacts.write_manifest(
            f"tune_tabular_{dataset_regime}",
            ["manifests", f"tune_tabular_{dataset_regime}.json"],
            inputs=[artifacts.relative(artifacts.paths.artifact_path("datasets", "tabular", f"tabular_{dataset_regime}_labeled.csv"))],
            outputs=[artifacts.relative(manifest_path)],
            metadata={"n_trials": int(len(trials_df)), "models": list(params)},
        )
    best_path = artifacts.write_json(best_params, "tuning", "tabular_best_params.json")
    outputs = {"best_params": str(best_path)}
    if trial_frames:
        trials_path = artifacts.write_dataframe(pd.concat(trial_frames, ignore_index=True), "tuning", "tabular_trials.parquet")
        outputs["trials"] = str(trials_path)
    return outputs


def run_tune_sequence(config: dict) -> dict[str, str]:
    artifacts = ArtifactManager(config)
    sequence_path = artifacts.paths.artifact_path("datasets", "sequence", "lstm_trainable.pkl")
    if not sequence_path.exists():
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
    params, trials_df = tune_sequence_dataset(sequence_df, manifest, config)
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
    )
    return outputs
