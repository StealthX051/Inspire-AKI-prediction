from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler

from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.io.csv import read_csv_optimized
from inspire_aki.runtime import build_stage_runtime_plan, thread_limited_context


TIMESERIES_FILTER_COLUMNS = ["op_id", "chart_time", "item_name", "value"]


def _empty_timeseries_frame(regular_labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=["op_id", "chart_time", *regular_labels])


def _prepare_stage_dir(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _label_seed(label: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(label))


def _compute_label_outlier_stats(values: pd.Series, config: dict) -> dict[str, float] | None:
    outlier_cfg = config["features"]["outlier_quantiles"]
    if values.empty:
        return None
    lower_1 = values.quantile(outlier_cfg["lower_extreme"])
    upper_1 = values.quantile(outlier_cfg["upper_extreme"])
    if pd.isna(lower_1) or pd.isna(upper_1):
        return None
    return {
        "lower_1": float(lower_1),
        "upper_1": float(upper_1),
        "lower_05": float(values.quantile(outlier_cfg["lower_fill_low"])),
        "lower_5": float(values.quantile(outlier_cfg["lower_fill_high"])),
        "upper_95": float(values.quantile(outlier_cfg["upper_fill_low"])),
        "upper_995": float(values.quantile(outlier_cfg["upper_fill_high"])),
    }


def _deterministic_uniform(df: pd.DataFrame, label: str) -> np.ndarray:
    op_id = pd.to_numeric(df["op_id"], errors="coerce").fillna(0).astype("int64").to_numpy()
    chart_time = pd.to_numeric(df["chart_time"], errors="coerce").fillna(0).astype("int64").to_numpy()
    hashed = (op_id * 73_856_093 + chart_time * 19_349_663 + _label_seed(label) * 83_492_791) % 10_000_019
    return hashed.astype(float) / 10_000_019.0


def _apply_label_outlier_stats(label_df: pd.DataFrame, label: str, stats: dict[str, float] | None) -> pd.DataFrame:
    if label_df.empty or stats is None:
        return label_df
    output = label_df.copy()
    fractions = _deterministic_uniform(output, label)
    lower_mask = output["value"] < stats["lower_1"]
    upper_mask = output["value"] > stats["upper_1"]
    if int(lower_mask.sum()) > 0:
        replacement = stats["lower_05"] + fractions[lower_mask.to_numpy()] * (stats["lower_5"] - stats["lower_05"])
        output.loc[lower_mask, "value"] = replacement
    if int(upper_mask.sum()) > 0:
        replacement = stats["upper_95"] + fractions[upper_mask.to_numpy()] * (stats["upper_995"] - stats["upper_95"])
        output.loc[upper_mask, "value"] = replacement
    return output


def _compute_regular_label_stats(df_regular: pd.DataFrame, regular_labels: list[str], config: dict) -> dict[str, dict[str, float] | None]:
    stats: dict[str, dict[str, float] | None] = {}
    for label in regular_labels:
        label_values = df_regular.loc[df_regular["item_name"] == label, "value"]
        stats[label] = _compute_label_outlier_stats(label_values, config)
    return stats


def _compute_partitioned_label_stats(
    filtered_paths: list[Path],
    regular_labels: list[str],
    config: dict,
) -> dict[str, dict[str, float] | None]:
    value_buffers: dict[str, list[np.ndarray]] = {label: [] for label in regular_labels}
    for path in filtered_paths:
        frame = pd.read_parquet(path, columns=["item_name", "value"])
        if frame.empty:
            continue
        for label, group in frame.groupby("item_name", sort=False):
            if label in value_buffers:
                value_buffers[label].append(group["value"].to_numpy())
    stats: dict[str, dict[str, float] | None] = {}
    for label in regular_labels:
        if value_buffers[label]:
            values = pd.Series(np.concatenate(value_buffers[label]))
            stats[label] = _compute_label_outlier_stats(values, config)
        else:
            stats[label] = None
    return stats


def build_clean_timeseries(vitals_df: pd.DataFrame, op_ids: pd.Series, config: dict) -> pd.DataFrame:
    feature_cfg = config["features"]
    seq_cfg = config["sequence"]
    regular_labels = feature_cfg["high_frequency_labels"] + feature_cfg["medium_frequency_labels"]

    df_vitals = vitals_df[vitals_df["op_id"].isin(op_ids.unique())].copy()
    df_vitals = df_vitals.drop_duplicates(subset=["op_id", "chart_time", "item_name"], keep="first")
    df_regular = df_vitals.loc[df_vitals["item_name"].isin(regular_labels), TIMESERIES_FILTER_COLUMNS]
    label_stats = _compute_regular_label_stats(df_regular, regular_labels, config)

    contained = []
    for label in regular_labels:
        label_df = df_regular.loc[df_regular["item_name"] == label].copy()
        if label_df.empty:
            continue
        label_df = _apply_label_outlier_stats(label_df, label, label_stats[label])
        contained.append(label_df)
    df_regular = pd.concat(contained, ignore_index=True) if contained else df_regular

    interpolated = []
    step = seq_cfg["interpolation_step_minutes"]
    for op_id, group in df_regular.groupby("op_id", sort=False):
        op_frame = group[["item_name", "value", "chart_time"]].copy()
        df_complete = pd.DataFrame({"chart_time": np.arange(op_frame["chart_time"].min(), op_frame["chart_time"].max() + step, step)})
        op_frame = op_frame.pivot(index="chart_time", columns="item_name", values="value").reindex(columns=regular_labels)
        df_complete = df_complete.merge(op_frame, on="chart_time", how="left")
        df_complete.fillna(df_complete.mean(numeric_only=True), inplace=True)
        df_complete["op_id"] = op_id
        interpolated.append(df_complete[["op_id", "chart_time", *regular_labels]])

    df_final = pd.concat(interpolated, ignore_index=True) if interpolated else _empty_timeseries_frame(regular_labels)
    if seq_cfg["normalize_timeseries"] and not df_final.empty:
        ignore = {"op_id", "chart_time", "aki"}
        cols_to_norm = [col for col in df_final.columns if col not in ignore]
        scaler = StandardScaler()
        df_final[cols_to_norm] = scaler.fit_transform(df_final[cols_to_norm])
    return df_final


def cleaned_timeseries_partition_paths(artifacts: ArtifactManager) -> list[Path]:
    stage_dir = artifacts.paths.artifact_path("staging", "timeseries_cleaned")
    if not stage_dir.exists():
        return []
    return sorted(stage_dir.glob("part-*.parquet"))


def _write_partitioned_vitals(
    *,
    raw_vitals_path: Path,
    op_ids: pd.Series,
    regular_labels: list[str],
    config: dict,
    output_dir: Path,
    partition_count: int,
) -> list[Path]:
    op_id_set = set(pd.Series(op_ids).astype(int).tolist())
    label_set = set(regular_labels)
    writers: dict[int, pq.ParquetWriter] = {}
    output_paths: dict[int, Path] = {}
    try:
        reader = read_csv_optimized(
            raw_vitals_path,
            config=config,
            usecols=TIMESERIES_FILTER_COLUMNS,
            chunksize=1_000_000,
            large=True,
        )
        for chunk in reader:
            filtered = chunk[chunk["op_id"].isin(op_id_set) & chunk["item_name"].isin(label_set)].copy()
            if filtered.empty:
                continue
            filtered["partition_id"] = filtered["op_id"].astype("int64") % partition_count
            for partition_id, part_df in filtered.groupby("partition_id", sort=False):
                payload = part_df[TIMESERIES_FILTER_COLUMNS]
                output_path = output_dir / f"part-{int(partition_id):05d}.parquet"
                table = pa.Table.from_pandas(payload, preserve_index=False)
                if partition_id not in writers:
                    writers[partition_id] = pq.ParquetWriter(output_path, table.schema)
                    output_paths[partition_id] = output_path
                writers[partition_id].write_table(table)
    finally:
        for writer in writers.values():
            writer.close()
    return [output_paths[key] for key in sorted(output_paths)]


def _clean_partition_frame(
    df_regular: pd.DataFrame,
    regular_labels: list[str],
    config: dict,
    label_stats: dict[str, dict[str, float] | None],
) -> pd.DataFrame:
    seq_cfg = config["sequence"]
    df_regular = df_regular.drop_duplicates(subset=["op_id", "chart_time", "item_name"], keep="first")
    contained = []
    for label in regular_labels:
        label_df = df_regular.loc[df_regular["item_name"] == label].copy()
        if label_df.empty:
            continue
        label_df = _apply_label_outlier_stats(label_df, label, label_stats[label])
        contained.append(label_df)
    df_regular = pd.concat(contained, ignore_index=True) if contained else df_regular
    if df_regular.empty:
        return _empty_timeseries_frame(regular_labels)

    interpolated: list[pd.DataFrame] = []
    step = seq_cfg["interpolation_step_minutes"]
    for op_id, group in df_regular.groupby("op_id", sort=False):
        op_frame = group[["item_name", "value", "chart_time"]].copy()
        full_times = pd.DataFrame({"chart_time": np.arange(op_frame["chart_time"].min(), op_frame["chart_time"].max() + step, step)})
        wide = op_frame.pivot(index="chart_time", columns="item_name", values="value").reindex(columns=regular_labels)
        df_complete = full_times.merge(wide, on="chart_time", how="left")
        df_complete.fillna(df_complete.mean(numeric_only=True), inplace=True)
        df_complete["op_id"] = op_id
        interpolated.append(df_complete[["op_id", "chart_time", *regular_labels]])
    return pd.concat(interpolated, ignore_index=True) if interpolated else _empty_timeseries_frame(regular_labels)


def _clean_partition_file(
    filtered_path: Path,
    cleaned_path: Path,
    regular_labels: list[str],
    config: dict,
    label_stats: dict[str, dict[str, float] | None],
    nested_blas_threads: int,
) -> int:
    with thread_limited_context(nested_blas_threads):
        filtered_df = pd.read_parquet(filtered_path)
        cleaned_df = _clean_partition_frame(filtered_df, regular_labels, config, label_stats)
        cleaned_df.to_parquet(cleaned_path, index=False)
        return int(len(cleaned_df))


def _fit_partitioned_scaler(cleaned_paths: list[Path], regular_labels: list[str], nested_blas_threads: int) -> StandardScaler | None:
    scaler = StandardScaler()
    fitted = False
    with thread_limited_context(nested_blas_threads):
        for path in cleaned_paths:
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            scaler.partial_fit(frame[regular_labels])
            fitted = True
    return scaler if fitted else None


def _apply_partitioned_scaler(cleaned_paths: list[Path], regular_labels: list[str], scaler: StandardScaler, nested_blas_threads: int) -> None:
    with thread_limited_context(nested_blas_threads):
        for path in cleaned_paths:
            frame = pd.read_parquet(path)
            if frame.empty:
                continue
            frame.loc[:, regular_labels] = scaler.transform(frame[regular_labels])
            frame.to_parquet(path, index=False)


def _combine_partitioned_csv(cleaned_paths: list[Path], final_path: Path, regular_labels: list[str]) -> int:
    row_count = 0
    header_written = False
    if final_path.exists():
        final_path.unlink()
    for path in cleaned_paths:
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        frame = frame[["op_id", "chart_time", *regular_labels]]
        frame.to_csv(final_path, mode="a", header=not header_written, index=False)
        header_written = True
        row_count += len(frame)
    if not header_written:
        _empty_timeseries_frame(regular_labels).to_csv(final_path, index=False)
    return row_count


def build_clean_timeseries_partitioned(
    *,
    raw_vitals_path: Path,
    op_ids: pd.Series,
    config: dict,
    artifacts: ArtifactManager,
) -> tuple[Path, int]:
    feature_cfg = config["features"]
    seq_cfg = config["sequence"]
    regular_labels = feature_cfg["high_frequency_labels"] + feature_cfg["medium_frequency_labels"]
    runtime_plan = build_stage_runtime_plan(config, "preprocess_timeseries")

    filtered_dir = _prepare_stage_dir(artifacts.paths.artifact_path("staging", "timeseries_filtered"))
    cleaned_dir = _prepare_stage_dir(artifacts.paths.artifact_path("staging", "timeseries_cleaned"))
    filtered_paths = _write_partitioned_vitals(
        raw_vitals_path=raw_vitals_path,
        op_ids=op_ids,
        regular_labels=regular_labels,
        config=config,
        output_dir=filtered_dir,
        partition_count=runtime_plan.timeseries_partitions,
    )

    cleaned_paths = [cleaned_dir / path.name for path in filtered_paths]
    if filtered_paths:
        label_stats = _compute_partitioned_label_stats(filtered_paths, regular_labels, config)
        Parallel(n_jobs=max(1, runtime_plan.timeseries_workers), backend="loky")(
            delayed(_clean_partition_file)(
                filtered_path,
                cleaned_path,
                regular_labels,
                config,
                label_stats,
                runtime_plan.nested_blas_threads,
            )
            for filtered_path, cleaned_path in zip(filtered_paths, cleaned_paths, strict=True)
        )
    else:
        cleaned_paths = []

    if seq_cfg["normalize_timeseries"] and cleaned_paths:
        scaler = _fit_partitioned_scaler(cleaned_paths, regular_labels, runtime_plan.nested_blas_threads)
        if scaler is not None:
            _apply_partitioned_scaler(cleaned_paths, regular_labels, scaler, runtime_plan.nested_blas_threads)

    final_path = artifacts.resolve("features", "timeseries", "time_series_cleaned.csv")
    row_count = _combine_partitioned_csv(cleaned_paths, final_path, regular_labels)
    return final_path, row_count
