from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml
from sklearn.preprocessing import StandardScaler

from inspire_aki.config import REPO_ROOT, load_config
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.models.sequence import (
    HybridModel,
    _df_to_tensors,
    _enable_sequence_cuda_benchmark,
    _sequence_loader_kwargs,
    sequence_feature_columns,
)
from inspire_aki.runtime import build_stage_runtime_plan


_TARGET_COMMANDS: dict[str, tuple[str, ...]] = {
    "run_all": ("run", "all"),
    "tune_tabular": ("tune", "tabular"),
    "tune_sequence": ("tune", "sequence"),
    "train_tabular": ("train", "tabular"),
    "train_sequence": ("train", "sequence"),
}

_TARGET_MANIFESTS = {
    "tune_tabular": ("manifests", "tune_tabular.json"),
    "tune_sequence": ("manifests", "tune_sequence.json"),
    "train_tabular": ("manifests", "train_tabular.json"),
    "train_sequence": ("manifests", "train_sequence.json"),
}

_MAX_RSS_PATTERN = re.compile(r"Maximum resident set size \(kbytes\):\s+(\d+)")


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_benchmark_config(base_config: dict[str, Any], profile: str, path: Path) -> Path:
    payload = json.loads(json.dumps(base_config, default=str))
    payload.setdefault("runtime", {})["profile"] = profile
    payload.setdefault("paths", {})["artifacts_dir"] = str(path.parent / "artifacts")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _stage_command(target: str, config_path: Path) -> list[str]:
    if target not in _TARGET_COMMANDS:
        raise ValueError(f"Unsupported benchmark target '{target}'.")
    return [sys.executable, "-m", "inspire_aki", *_TARGET_COMMANDS[target], "--config", str(config_path)]


def _manifest_wall_time(config: dict[str, Any], target: str) -> float | None:
    manifest_parts = _TARGET_MANIFESTS.get(target)
    if manifest_parts is None:
        return None
    artifacts = ArtifactManager(config)
    manifest_path = artifacts.paths.artifact_path(*manifest_parts)
    if not manifest_path.exists():
        return None
    payload = artifacts.read_json(*manifest_parts)
    wall_time = payload.get("wall_time_seconds")
    return None if wall_time is None else float(wall_time)


def _max_rss_kb(raw_output: str) -> int | None:
    match = _MAX_RSS_PATTERN.search(raw_output)
    if match is None:
        return None
    return int(match.group(1))


def _run_subprocess_target(
    target: str,
    config_path: Path,
    log_path: Path,
    *,
    env_overrides: dict[str, str] | None = None,
) -> tuple[int, float, str]:
    command = _stage_command(target, config_path)
    time_cmd = shutil.which("time")
    started = perf_counter()
    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)
    if time_cmd is not None and Path(time_cmd).resolve() == Path("/usr/bin/time"):
        completed = subprocess.run(
            [time_cmd, "-v", *command],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        raw_output = (completed.stdout or "") + (completed.stderr or "")
    else:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            env=env,
        )
        raw_output = (completed.stdout or "") + (completed.stderr or "")
    elapsed = perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(raw_output, encoding="utf-8")
    return completed.returncode, elapsed, raw_output


def _invoke_subprocess_target(
    target: str,
    config_path: Path,
    log_path: Path,
    *,
    env_overrides: dict[str, str] | None = None,
) -> tuple[int, float, str]:
    try:
        return _run_subprocess_target(target, config_path, log_path, env_overrides=env_overrides)
    except TypeError as exc:
        if "env_overrides" not in str(exc):
            raise
        return _run_subprocess_target(target, config_path, log_path)


def benchmark_sequence_loader(
    config: dict[str, Any],
    *,
    output_dir: Path,
    sample_size: int = 4096,
    epochs: int = 2,
    model_key: str = "lstm_only",
) -> list[dict[str, Any]]:
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("torch is required for the sequence_loader benchmark target.") from exc

    artifacts = ArtifactManager(config)
    sequence_path = artifacts.paths.artifact_path("datasets", "sequence", "lstm_trainable.pkl")
    if not sequence_path.exists():
        raise FileNotFoundError(sequence_path)
    sequence_df = artifacts.read_pickle("datasets", "sequence", "lstm_trainable.pkl")
    target = config["models"]["target"]
    feature_cols = sequence_feature_columns(sequence_df, target)
    sample_df = sequence_df.head(sample_size).copy()
    scaler = StandardScaler()
    sample_df.loc[:, feature_cols] = scaler.fit_transform(sample_df[feature_cols].astype(float))
    dataset = TensorDataset(*_df_to_tensors(sample_df, feature_cols, target=target))
    plan = build_stage_runtime_plan(config, "tune_sequence")
    device = torch.device("cuda" if plan.sequence_use_gpu and torch.cuda.is_available() else "cpu")
    _enable_sequence_cuda_benchmark(device)

    defaults = config["models"]["sequence_defaults"]
    lstm_input_size = dataset.tensors[1].shape[2] if dataset.tensors[1].numel() > 1 else 0
    combos = [
        {"batch_size": 512, "workers": 0},
        {"batch_size": 1024, "workers": 0},
        {"batch_size": 1024, "workers": 4},
        {"batch_size": 1024, "workers": 8},
    ]
    rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for combo in combos:
        model = HybridModel(
            tabular_input_size=len(feature_cols),
            lstm_input_size=lstm_input_size,
            lstm_hidden_size=int(defaults["lstm_hidden_size"]),
            lstm_num_layers=int(defaults["lstm_num_layers"]),
            mlp_dims=list(defaults.get("mlp_dims", [])),
            dropout_rate=float(defaults["dropout_rate"]),
            mode=model_key,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(defaults["learning_rate"]))
        criterion = torch.nn.BCEWithLogitsLoss()
        loader = DataLoader(
            dataset,
            batch_size=int(combo["batch_size"]),
            **_sequence_loader_kwargs(loader_workers=int(combo["workers"]), use_gpu=device.type == "cuda", shuffle=True),
        )
        started = perf_counter()
        for _ in range(epochs):
            model.train()
            for batch in loader:
                x_tab_batch, x_time_batch, seq_len_batch, y_batch = [tensor.to(device) for tensor in batch]
                optimizer.zero_grad()
                outputs = model(x_tab_batch, x_time_batch, seq_len_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()
        elapsed = perf_counter() - started
        row = {
            "target": "sequence_loader",
            "batch_size": int(combo["batch_size"]),
            "workers": int(combo["workers"]),
            "persistent_workers": bool(combo["workers"] > 0),
            "epochs": int(epochs),
            "sample_size": int(len(sample_df)),
            "wall_time_seconds": elapsed,
            "device": device.type,
        }
        rows.append(row)
    (output_dir / "sequence_loader_results.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    return rows


def run_runtime_benchmarks(
    *,
    config_path: str | None,
    profiles: list[str],
    targets: list[str],
    repeats: int,
    output_dir: Path,
    model_keys: list[str] | None = None,
    dataset_regimes: list[str] | None = None,
    execution_policy: str = "optimized_low_cpu",
) -> dict[str, Any]:
    base_config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    env_overrides: dict[str, str] = {"INSPIRE_AKI_EXECUTION_POLICY": execution_policy}
    if model_keys:
        env_overrides["INSPIRE_AKI_MODEL_KEYS"] = ",".join(model_keys)
    if dataset_regimes:
        env_overrides["INSPIRE_AKI_DATASET_REGIMES"] = ",".join(dataset_regimes)

    for profile in profiles:
        profile_dir = output_dir / profile
        for target in targets:
            for repeat_idx in range(1, repeats + 1):
                run_dir = profile_dir / target / f"repeat_{repeat_idx}"
                run_dir.mkdir(parents=True, exist_ok=True)
                run_config = json.loads(json.dumps(base_config, default=str))
                run_config.setdefault("runtime", {})["profile"] = profile
                temp_config_path = _write_benchmark_config(run_config, profile, run_dir / "benchmark_config.yaml")
                loaded_run_config = load_config(temp_config_path)
                started_utc = _timestamp_utc()
                if target == "sequence_loader":
                    rows = benchmark_sequence_loader(loaded_run_config, output_dir=run_dir)
                    for row in rows:
                        summary_rows.append(
                            {
                                "profile": profile,
                                "target": target,
                                "repeat": repeat_idx,
                                "execution_policy": execution_policy,
                                "model_keys": ",".join(model_keys or []),
                                "dataset_regimes": ",".join(dataset_regimes or []),
                                "started_utc": started_utc,
                                "finished_utc": _timestamp_utc(),
                                "returncode": 0,
                                "manifest_wall_time_seconds": None,
                                "max_rss_kb": None,
                                "log_path": str(run_dir / "sequence_loader_results.json"),
                                **row,
                            }
                        )
                    continue

                log_path = run_dir / "raw.log"
                returncode, elapsed, raw_output = _invoke_subprocess_target(
                    target,
                    temp_config_path,
                    log_path,
                    env_overrides=env_overrides,
                )
                summary_rows.append(
                    {
                        "profile": profile,
                        "target": target,
                        "repeat": repeat_idx,
                        "execution_policy": execution_policy,
                        "model_keys": ",".join(model_keys or []),
                        "dataset_regimes": ",".join(dataset_regimes or []),
                        "started_utc": started_utc,
                        "finished_utc": _timestamp_utc(),
                        "wall_time_seconds": elapsed,
                        "returncode": returncode,
                        "manifest_wall_time_seconds": _manifest_wall_time(loaded_run_config, target),
                        "max_rss_kb": _max_rss_kb(raw_output),
                        "log_path": str(log_path),
                    }
                )

    summary_json_path = output_dir / "summary.json"
    summary_csv_path = output_dir / "summary.csv"
    summary_json_path.write_text(json.dumps(summary_rows, indent=2, sort_keys=True), encoding="utf-8")
    if summary_rows:
        fieldnames = sorted({key for row in summary_rows for key in row})
        with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    else:
        summary_csv_path.write_text("", encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "profiles": profiles,
        "targets": targets,
        "repeats": repeats,
        "execution_policy": execution_policy,
        "rows": len(summary_rows),
        "summary_json": str(summary_json_path),
        "summary_csv": str(summary_csv_path),
    }
