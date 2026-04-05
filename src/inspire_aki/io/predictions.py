from __future__ import annotations

from pathlib import Path

import pandas as pd

from inspire_aki.io.artifacts import ArtifactManager

PREDICTION_PARTITIONS = ("tabular", "sequence")
PREDICTION_PRIMARY_KEY = [
    "op_id",
    "dataset_regime",
    "population_id",
    "repeat_id",
    "fold_id",
    "split_name",
    "model_key",
]
PREDICTION_SORT_COLUMNS = ["dataset_regime", "model_key", "repeat_id", "fold_id", "op_id"]
PREDICTION_COLUMNS = [
    "op_id",
    "patient_id",
    "dataset_regime",
    "population_id",
    "repeat_id",
    "fold_id",
    "split_name",
    "model_key",
    "target",
    "y_true",
    "y_prob_raw",
    "y_prob_calibrated",
    "y_pred",
    "threshold",
    "calibration_method",
    "run_id",
    "source_index",
]


def _normalize_prediction_frame(predictions_df: pd.DataFrame) -> pd.DataFrame:
    if predictions_df.empty:
        return pd.DataFrame(columns=PREDICTION_COLUMNS)
    frame = predictions_df.copy()
    for column in PREDICTION_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[PREDICTION_COLUMNS]
    frame = frame.drop_duplicates(subset=PREDICTION_PRIMARY_KEY, keep="last")
    return frame.sort_values(PREDICTION_SORT_COLUMNS + ["population_id", "split_name"], kind="stable").reset_index(drop=True)


def prediction_partition_path(artifacts: ArtifactManager, partition_name: str) -> Path:
    return artifacts.paths.artifact_path("predictions", "raw", f"{partition_name}.parquet")


def write_prediction_partition(partition_name: str, predictions_df: pd.DataFrame, artifacts: ArtifactManager) -> Path:
    if partition_name not in PREDICTION_PARTITIONS:
        raise ValueError(f"Unknown prediction partition '{partition_name}'.")
    normalized = _normalize_prediction_frame(predictions_df)
    return artifacts.write_dataframe(normalized, "predictions", "raw", f"{partition_name}.parquet")


def read_prediction_partitions(artifacts: ArtifactManager) -> dict[str, pd.DataFrame]:
    partitions: dict[str, pd.DataFrame] = {}
    for partition_name in PREDICTION_PARTITIONS:
        path = prediction_partition_path(artifacts, partition_name)
        if path.exists():
            partitions[partition_name] = pd.read_parquet(path)
    return partitions


def materialize_raw_predictions(artifacts: ArtifactManager) -> Path:
    partitions = read_prediction_partitions(artifacts)
    frames = [frame for frame in partitions.values() if not frame.empty]
    if frames:
        combined = _normalize_prediction_frame(pd.concat(frames, ignore_index=True))
    else:
        combined = pd.DataFrame(columns=PREDICTION_COLUMNS)
    return artifacts.write_dataframe(combined, "predictions", "raw_predictions.parquet")


def read_raw_predictions(artifacts: ArtifactManager) -> pd.DataFrame:
    raw_path = artifacts.paths.artifact_path("predictions", "raw_predictions.parquet")
    if not raw_path.exists():
        raise FileNotFoundError(f"Materialized raw prediction artifact not found: {raw_path}")
    return pd.read_parquet(raw_path)
