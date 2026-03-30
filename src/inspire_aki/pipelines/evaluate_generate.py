from __future__ import annotations

from pathlib import Path
from time import perf_counter

import pandas as pd

from inspire_aki.evaluation.backends import build_evaluation_backend
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.models.tabular import selected_tabular_dataset_regimes
from inspire_aki.runtime import build_stage_runtime_plan


def run_evaluate_generate(config: dict) -> dict[str, str]:
    stage_name = "evaluate_generate"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    backend = build_evaluation_backend(config)
    target = config["models"]["target"]

    manifest_paths: list[str] = []
    audit_frames: list[pd.DataFrame] = []
    dataset_count = 0

    for dataset_regime in selected_tabular_dataset_regimes():
        dataset_path = artifacts.paths.artifact_path("datasets", "tabular", f"tabular_{dataset_regime}_labeled.csv")
        if not dataset_path.exists():
            continue
        dataset_df = pd.read_csv(dataset_path)
        result = backend.build(dataset_df, target=target, dataset_family=dataset_regime)
        manifest_path = artifacts.write_dataframe(result.manifest, "datasets", "splits", f"{backend.mode}_{dataset_regime}.parquet")
        audit_df = result.overlap_audit.copy()
        audit_df.insert(0, "dataset_family", dataset_regime)
        artifacts.write_dataframe(audit_df, "evaluation", f"split_audit_{dataset_regime}.csv")
        manifest_paths.append(str(manifest_path))
        audit_frames.append(audit_df)
        dataset_count += 1

    sequence_path = artifacts.paths.artifact_path("datasets", "sequence", "lstm_trainable.pkl")
    if sequence_path.exists():
        sequence_df = artifacts.read_pickle("datasets", "sequence", "lstm_trainable.pkl")
        result = backend.build(sequence_df, target=target, dataset_family="sequence")
        manifest_path = artifacts.write_dataframe(result.manifest, "datasets", "splits", f"{backend.mode}_sequence.parquet")
        audit_df = result.overlap_audit.copy()
        audit_df.insert(0, "dataset_family", "sequence")
        artifacts.write_dataframe(audit_df, "evaluation", "split_audit_sequence.csv")
        manifest_paths.append(str(manifest_path))
        audit_frames.append(audit_df)
        dataset_count += 1

    combined_audit = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()
    audit_path = artifacts.write_dataframe(combined_audit, "evaluation", "split_audit.csv")
    outputs: dict[str, str] = {"split_audit": str(audit_path)}
    for idx, path in enumerate(manifest_paths):
        outputs[f"manifest_{idx}"] = path

    artifacts.write_manifest(
        stage_name,
        ["manifests", "evaluate_generate.json"],
        outputs=[artifacts.relative(artifacts.paths.artifact_path("evaluation", "split_audit.csv"))]
        + [artifacts.relative(artifacts.paths.artifact_path("datasets", "splits", Path(path).name)) for path in manifest_paths],
        metadata={
            "evaluation_mode": backend.mode,
            "n_datasets": dataset_count,
            "n_audit_rows": int(len(combined_audit)),
        },
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return outputs
