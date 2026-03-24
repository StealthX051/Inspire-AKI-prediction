from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from inspire_aki.features.timeseries import cleaned_timeseries_partition_paths
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.runtime import build_stage_runtime_plan, thread_limited_context


def _prepare_stage_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_sequence_dataset(tabular_df: pd.DataFrame, timeseries_df: pd.DataFrame, config: dict) -> pd.DataFrame:
    seq_cfg = config["sequence"]
    df_time = timeseries_df.copy()
    feature_mask = df_time.drop(columns=["op_id", "chart_time"]).notna().astype(int)
    regular_feature_count = len(config["features"]["high_frequency_labels"] + config["features"]["medium_frequency_labels"])
    df_time["presence"] = feature_mask.sum(axis=1) / regular_feature_count
    df_time = df_time[df_time["presence"] > seq_cfg["presence_threshold"]].drop(columns=["presence"])
    df_time = df_time.fillna(seq_cfg["fill_value"])

    df_tabular = tabular_df.copy()
    bool_cols = df_tabular.select_dtypes(include="bool").columns
    df_tabular[bool_cols] = df_tabular[bool_cols].astype(float)

    padded_tensors = []
    op_ids = []
    sequence_lengths = []
    pad_length = seq_cfg["pad_length"]
    for op_id, group in df_time.groupby("op_id", sort=False):
        mat = group.drop(columns=["op_id", "chart_time"]).to_numpy(dtype=float)
        if mat.shape[0] < pad_length:
            padded = np.pad(mat, pad_width=((0, pad_length - mat.shape[0]), (0, 0)), mode="constant", constant_values=0.0)
            padded_tensors.append(padded)
            op_ids.append(op_id)
            sequence_lengths.append(mat.shape[0])

    df_sequence = pd.DataFrame(
        {
            "op_id": op_ids,
            "time_tensors": padded_tensors,
            "seq_len": sequence_lengths,
        }
    )
    return df_sequence.merge(df_tabular, on="op_id", how="inner")


def _build_sequence_partition(
    cleaned_path: Path,
    tabular_partition: pd.DataFrame,
    partition_path: Path,
    config: dict,
    nested_blas_threads: int,
) -> int:
    with thread_limited_context(nested_blas_threads):
        timeseries_df = pd.read_parquet(cleaned_path)
        sequence_df = build_sequence_dataset(tabular_partition, timeseries_df, config)
        sequence_df.to_pickle(partition_path)
        return int(len(sequence_df))


def build_sequence_dataset_partitioned(
    *,
    tabular_df: pd.DataFrame,
    config: dict,
    artifacts: ArtifactManager,
) -> tuple[pd.DataFrame, list[Path]]:
    runtime_plan = build_stage_runtime_plan(config, "preprocess_sequence")
    partition_dir = _prepare_stage_dir(artifacts.paths.artifact_path("staging", "sequence"))
    cleaned_paths = cleaned_timeseries_partition_paths(artifacts)
    if not cleaned_paths:
        return pd.DataFrame(columns=["op_id", "time_tensors", "seq_len"]), []

    tabular_shards: dict[int, pd.DataFrame] = {}
    working_tabular = tabular_df.copy()
    bool_cols = working_tabular.select_dtypes(include="bool").columns
    working_tabular[bool_cols] = working_tabular[bool_cols].astype(float)
    shard_ids = working_tabular["op_id"].astype("int64") % runtime_plan.sequence_partitions
    for partition_id, shard in working_tabular.groupby(shard_ids, sort=False):
        tabular_shards[int(partition_id)] = shard.copy()

    jobs = []
    for cleaned_path in cleaned_paths:
        partition_id = int(cleaned_path.stem.split("-")[-1])
        tabular_partition = tabular_shards.get(partition_id)
        if tabular_partition is None or tabular_partition.empty:
            continue
        jobs.append((cleaned_path, tabular_partition, partition_dir / f"part-{partition_id:05d}.pkl"))

    if jobs:
        Parallel(n_jobs=max(1, runtime_plan.sequence_workers), backend="loky")(
            delayed(_build_sequence_partition)(
                cleaned_path,
                tabular_partition,
                partition_path,
                config,
                runtime_plan.nested_blas_threads,
            )
            for cleaned_path, tabular_partition, partition_path in jobs
        )

    partition_paths = sorted(partition_dir.glob("part-*.pkl"))
    frames = [pd.read_pickle(path) for path in partition_paths]
    if not frames:
        return pd.DataFrame(columns=["op_id", "time_tensors", "seq_len"]), partition_paths
    return pd.concat(frames, ignore_index=True), partition_paths
