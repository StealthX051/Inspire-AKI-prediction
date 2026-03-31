from __future__ import annotations

from time import perf_counter

import pandas as pd

from inspire_aki.cohort.labels import derive_active_labels
from inspire_aki.config import active_outcome_config, active_outcome_key, active_target_column
from inspire_aki.cohort.preop import build_preop_features
from inspire_aki.datasets.sequence import build_sequence_dataset_partitioned
from inspire_aki.datasets.tabular import build_tabular_datasets
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.csv import read_csv_optimized
from inspire_aki.features.intraop_tabular import build_intraop_features
from inspire_aki.features.timeseries import build_clean_timeseries_partitioned
from inspire_aki.runtime import build_stage_runtime_plan


def run_preop(config: dict) -> dict[str, str]:
    stage_name = "preprocess_preop"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    preop_df, audit_df = build_preop_features(config, artifacts.paths.raw_inspire_dir)
    preop_path = artifacts.write_dataframe(preop_df, "features", "preop", "preop_features.csv")
    audit_path = artifacts.write_dataframe(audit_df, "cohort", "preop_audit.csv")
    raw_ops = pd.read_csv(
        artifacts.paths.raw_inspire_dir / "operations.csv",
        usecols=["opstart_time", "opend_time"],
    )
    raw_ops = raw_ops.dropna(subset=["opstart_time", "opend_time"])
    n_excluded_nonpositive_op_len = int((raw_ops["opend_time"] - raw_ops["opstart_time"] <= 0).sum())
    artifacts.write_manifest(
        stage_name,
        ["manifests", "preprocess_preop.json"],
        inputs=[
            artifacts.relative(artifacts.paths.raw_inspire_dir / "operations.csv"),
            artifacts.relative(artifacts.paths.raw_inspire_dir / "labs.csv"),
            artifacts.relative(artifacts.paths.raw_inspire_dir / "diagnosis.csv"),
            artifacts.relative(artifacts.paths.raw_inspire_dir / "ward_vitals.csv"),
        ],
        outputs=[artifacts.relative(preop_path), artifacts.relative(audit_path)],
        metadata={
            "n_rows": len(preop_df),
            "n_excluded_nonpositive_op_len": n_excluded_nonpositive_op_len,
        },
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return {"preop": str(preop_path), "audit": str(audit_path)}


def run_intraop(config: dict) -> dict[str, str]:
    stage_name = "preprocess_intraop"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    preop_df = pd.read_csv(artifacts.paths.artifact_path("features", "preop", "preop_features.csv"))
    vitals_df = read_csv_optimized(
        artifacts.paths.raw_inspire_dir / "vitals.csv",
        config=config,
        usecols=["op_id", "chart_time", "item_name", "value"],
        large=True,
    )
    intraop_df = build_intraop_features(vitals_df, preop_df, config)
    numeric_intraop = intraop_df.select_dtypes(include=["number"])
    n_inf_values = int(pd.Series(numeric_intraop.to_numpy().ravel()).isin([float("inf"), float("-inf")]).sum())
    n_nan_values = int(numeric_intraop.isna().sum().sum())
    if n_inf_values > 0:
        raise ValueError(f"Intraoperative feature artifact contains {n_inf_values} infinite values.")
    intraop_path = artifacts.write_dataframe(intraop_df, "features", "intraop", "feature_engineered.csv")
    artifacts.write_manifest(
        stage_name,
        ["manifests", "preprocess_intraop.json"],
        inputs=[
            artifacts.relative(artifacts.paths.raw_inspire_dir / "vitals.csv"),
            artifacts.relative(artifacts.paths.artifact_path("features", "preop", "preop_features.csv")),
        ],
        outputs=[artifacts.relative(intraop_path)],
        metadata={"n_rows": len(intraop_df), "n_inf_values": n_inf_values, "n_nan_values": n_nan_values},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return {"intraop": str(intraop_path)}


def run_tabular(config: dict) -> dict[str, str]:
    stage_name = "preprocess_tabular"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    preop_df = pd.read_csv(artifacts.paths.artifact_path("features", "preop", "preop_features.csv"))
    intraop_df = pd.read_csv(artifacts.paths.artifact_path("features", "intraop", "feature_engineered.csv"))
    datasets = build_tabular_datasets(preop_df, intraop_df, config)
    outputs = {}
    for key, frame in datasets.items():
        file_name = {
            "combined": "tabular_combined.csv",
            "preop": "tabular_preop.csv",
            "intraop": "tabular_intraop.csv",
            "combined_unnormalized": "tabular_combined_unnormalized.csv",
            "normalization_stats": "normalization_stats.csv",
            "fill_rates": "fill_rates.csv",
        }[key]
        base_parts = ["features"] if key == "fill_rates" else ["datasets", "tabular"]
        if key == "normalization_stats":
            base_parts = ["datasets", "tabular"]
        path = artifacts.write_dataframe(frame, *base_parts, file_name)
        outputs[key] = str(path)
    artifacts.write_manifest(
        stage_name,
        ["manifests", "preprocess_tabular.json"],
        inputs=[
            artifacts.relative(artifacts.paths.artifact_path("features", "preop", "preop_features.csv")),
            artifacts.relative(artifacts.paths.artifact_path("features", "intraop", "feature_engineered.csv")),
        ],
        outputs=[artifacts.relative(artifacts.paths.artifact_path("datasets", "tabular", name)) for name in [
            "tabular_combined.csv",
            "tabular_preop.csv",
            "tabular_intraop.csv",
            "tabular_combined_unnormalized.csv",
            "normalization_stats.csv",
        ]] + [artifacts.relative(artifacts.paths.artifact_path("features", "fill_rates.csv"))],
        metadata={"n_rows_combined": len(datasets["combined"])},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return outputs


def run_labels(config: dict) -> dict[str, str]:
    stage_name = "preprocess_labels"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    preop_df = pd.read_csv(artifacts.paths.artifact_path("features", "preop", "preop_features.csv"))
    combined_df = pd.read_csv(artifacts.paths.artifact_path("datasets", "tabular", "tabular_combined.csv"))
    labels_df, audit_df = derive_active_labels(
        config=config,
        raw_inspire_dir=artifacts.paths.raw_inspire_dir,
        preop_df=preop_df,
        tabular_combined_df=combined_df,
    )
    outcome_cfg = active_outcome_config(config)
    target = active_target_column(config)
    labels_for_datasets = labels_df[[column for column in ["op_id", "subject_id", "patient_id", target] if column in labels_df.columns]].copy()
    label_path = artifacts.write_dataframe(labels_df, "cohort", "labels.csv")
    audit_path = artifacts.write_dataframe(audit_df, "cohort", "labels_audit.csv")

    outputs = {"labels": str(label_path), "audit": str(audit_path)}
    if active_outcome_key(config) == "aki":
        outputs["legacy_labels"] = str(artifacts.write_dataframe(labels_df, "cohort", "aki_labels.csv"))
    for dataset_name in ["combined", "preop", "intraop"]:
        dataset_path = artifacts.paths.artifact_path("datasets", "tabular", f"tabular_{dataset_name}.csv")
        dataset_df = pd.read_csv(dataset_path)
        labeled_df = dataset_df.merge(labels_for_datasets, on="op_id", how="inner")
        out_path = artifacts.write_dataframe(labeled_df, "datasets", "tabular", f"tabular_{dataset_name}_labeled.csv")
        outputs[f"{dataset_name}_labeled"] = str(out_path)

    source_paths = {
        "operations": artifacts.paths.raw_inspire_dir / "operations.csv",
        "diagnosis": artifacts.paths.raw_inspire_dir / "diagnosis.csv",
        "labs": artifacts.paths.raw_inspire_dir / "labs.csv",
        "ward_vitals": artifacts.paths.raw_inspire_dir / "ward_vitals.csv",
    }
    artifacts.write_manifest(
        stage_name,
        ["manifests", "preprocess_labels.json"],
        inputs=[
            artifacts.relative(artifacts.paths.artifact_path("features", "preop", "preop_features.csv")),
            artifacts.relative(artifacts.paths.artifact_path("datasets", "tabular", "tabular_combined.csv")),
        ]
        + [
            artifacts.relative(source_paths[source_name])
            for source_name in outcome_cfg.get("required_sources", [])
        ],
        outputs=[artifacts.relative(artifacts.paths.artifact_path("cohort", "labels.csv"))],
        metadata={"n_labels": len(labels_df), "target": target},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return outputs


def run_timeseries(config: dict) -> dict[str, str]:
    stage_name = "preprocess_timeseries"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    labels_df = pd.read_csv(artifacts.paths.artifact_path("cohort", "labels.csv"))
    path, row_count = build_clean_timeseries_partitioned(
        raw_vitals_path=artifacts.paths.raw_inspire_dir / "vitals.csv",
        op_ids=labels_df["op_id"],
        config=config,
        artifacts=artifacts,
    )
    artifacts.write_manifest(
        stage_name,
        ["manifests", "preprocess_timeseries.json"],
        inputs=[
            artifacts.relative(artifacts.paths.raw_inspire_dir / "vitals.csv"),
            artifacts.relative(artifacts.paths.artifact_path("cohort", "labels.csv")),
        ],
        outputs=[artifacts.relative(path)],
        metadata={"n_rows": row_count},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return {"timeseries": str(path)}


def run_sequence(config: dict) -> dict[str, str]:
    stage_name = "preprocess_sequence"
    start = perf_counter()
    artifacts = ArtifactManager(config)
    tabular_df = pd.read_csv(artifacts.paths.artifact_path("datasets", "tabular", "tabular_combined_labeled.csv"))
    sequence_df, partition_paths = build_sequence_dataset_partitioned(
        tabular_df=tabular_df,
        config=config,
        artifacts=artifacts,
    )
    path = artifacts.write_pickle(sequence_df, "datasets", "sequence", "lstm_trainable.pkl")
    artifacts.write_manifest(
        stage_name,
        ["manifests", "preprocess_sequence.json"],
        inputs=[
            artifacts.relative(artifacts.paths.artifact_path("datasets", "tabular", "tabular_combined_labeled.csv")),
            artifacts.relative(artifacts.paths.artifact_path("features", "timeseries", "time_series_cleaned.csv")),
        ],
        outputs=[artifacts.relative(path)],
        metadata={"n_rows": len(sequence_df), "n_partitions": len(partition_paths)},
        stage_runtime_plan=build_stage_runtime_plan(config, stage_name).as_dict(),
        wall_time_seconds=perf_counter() - start,
    )
    return {"sequence": str(path)}
