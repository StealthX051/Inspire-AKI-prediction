from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from inspire_aki.config import load_config
from inspire_aki.datasets.sequence import build_sequence_dataset, build_sequence_dataset_partitioned
from inspire_aki.features.timeseries import build_clean_timeseries, build_clean_timeseries_partitioned
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.pipelines.preprocess import run_intraop, run_labels, run_preop, run_tabular


def _prepare_labeled_inputs(config_path: Path) -> tuple[dict, ArtifactManager, pd.DataFrame]:
    config = load_config(config_path)
    run_preop(config)
    run_intraop(config)
    run_tabular(config)
    run_labels(config)
    artifacts = ArtifactManager(config)
    labels_df = pd.read_csv(artifacts.paths.artifact_path("cohort", "labels.csv"))
    return config, artifacts, labels_df


def test_partitioned_timeseries_matches_sequential_output(synthetic_config: Path) -> None:
    config, artifacts, labels_df = _prepare_labeled_inputs(synthetic_config)
    vitals_df = pd.read_csv(
        artifacts.paths.raw_inspire_dir / "vitals.csv",
        usecols=["op_id", "chart_time", "item_name", "value"],
    )

    sequential_df = (
        build_clean_timeseries(vitals_df, labels_df["op_id"], config)
        .sort_values(["op_id", "chart_time"])
        .reset_index(drop=True)
    )
    partitioned_path, row_count = build_clean_timeseries_partitioned(
        raw_vitals_path=artifacts.paths.raw_inspire_dir / "vitals.csv",
        op_ids=labels_df["op_id"],
        config=config,
        artifacts=artifacts,
    )
    partitioned_df = pd.read_csv(partitioned_path).sort_values(["op_id", "chart_time"]).reset_index(drop=True)

    assert row_count == len(partitioned_df)
    pd.testing.assert_frame_equal(
        sequential_df,
        partitioned_df,
        check_dtype=False,
        check_exact=False,
        rtol=1e-9,
        atol=1e-9,
    )


def test_partitioned_sequence_matches_sequential_output(synthetic_config: Path) -> None:
    config, artifacts, labels_df = _prepare_labeled_inputs(synthetic_config)
    vitals_df = pd.read_csv(
        artifacts.paths.raw_inspire_dir / "vitals.csv",
        usecols=["op_id", "chart_time", "item_name", "value"],
    )
    sequential_timeseries = build_clean_timeseries(vitals_df, labels_df["op_id"], config)
    build_clean_timeseries_partitioned(
        raw_vitals_path=artifacts.paths.raw_inspire_dir / "vitals.csv",
        op_ids=labels_df["op_id"],
        config=config,
        artifacts=artifacts,
    )

    tabular_df = pd.read_csv(artifacts.paths.artifact_path("datasets", "tabular", "tabular_combined_labeled.csv"))
    sequential_df = build_sequence_dataset(tabular_df, sequential_timeseries, config).sort_values("op_id").reset_index(drop=True)
    partitioned_df, partition_paths = build_sequence_dataset_partitioned(
        tabular_df=tabular_df,
        config=config,
        artifacts=artifacts,
    )
    partitioned_df = partitioned_df.sort_values("op_id").reset_index(drop=True)

    assert partition_paths
    assert sequential_df["op_id"].tolist() == partitioned_df["op_id"].tolist()
    assert sequential_df["seq_len"].tolist() == partitioned_df["seq_len"].tolist()
    for seq_tensor, part_tensor in zip(sequential_df["time_tensors"], partitioned_df["time_tensors"], strict=True):
        np.testing.assert_allclose(seq_tensor, part_tensor)


def test_partitioned_timeseries_uses_staging_partitions(synthetic_config: Path) -> None:
    config, artifacts, labels_df = _prepare_labeled_inputs(synthetic_config)
    build_clean_timeseries_partitioned(
        raw_vitals_path=artifacts.paths.raw_inspire_dir / "vitals.csv",
        op_ids=labels_df["op_id"],
        config=config,
        artifacts=artifacts,
    )

    filtered_parts = sorted(artifacts.paths.artifact_path("staging", "timeseries_filtered").glob("part-*.parquet"))
    cleaned_parts = sorted(artifacts.paths.artifact_path("staging", "timeseries_cleaned").glob("part-*.parquet"))

    assert filtered_parts
    assert cleaned_parts
    assert {path.name for path in filtered_parts} == {path.name for path in cleaned_parts}
