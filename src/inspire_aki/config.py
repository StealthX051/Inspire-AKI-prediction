from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from inspire_aki.registry import MANUSCRIPT_SECTIONS, SUPPORTED_SHAP_MODELS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "aki" / "default.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config at {path} must load to a mapping.")
    return data


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    reports_cfg = normalized.setdefault("reports", {})
    if "batch_shap_jobs" in reports_cfg:
        reports_cfg["shap_jobs"] = copy.deepcopy(reports_cfg.pop("batch_shap_jobs"))
    else:
        reports_cfg.pop("batch_shap_jobs", None)
    reports_cfg.setdefault("shap_jobs", [])
    reports_cfg.setdefault("manuscript_sections", list(MANUSCRIPT_SECTIONS))

    compat_cfg = normalized.get("compat")
    if isinstance(compat_cfg, dict):
        compat_cfg.pop("export_legacy_aliases", None)
        if not compat_cfg:
            normalized.pop("compat", None)

    return normalized


def validate_config(config: dict[str, Any]) -> None:
    splits_cfg = config["splits"]
    if splits_cfg["use_bootstrapping"] and (splits_cfg["n_bootstrap_iterations"] % splits_cfg["n_cv_folds"]) != 0:
        raise ValueError("When bootstrapping is enabled, splits.n_bootstrap_iterations must be divisible by splits.n_cv_folds.")

    runtime_cfg = config.get("runtime", {})
    if runtime_cfg.get("profile", "balanced") not in {"balanced", "aggressive", "conservative", "throughput"}:
        raise ValueError("runtime.profile must be one of: balanced, aggressive, conservative, throughput.")
    if float(runtime_cfg.get("cpu_reserve_fraction", 0.125)) < 0:
        raise ValueError("runtime.cpu_reserve_fraction must be non-negative.")
    if float(runtime_cfg.get("ram_reserve_fraction", 0.15)) < 0:
        raise ValueError("runtime.ram_reserve_fraction must be non-negative.")
    if int(runtime_cfg.get("cpu_reserve_min", 4)) < 0:
        raise ValueError("runtime.cpu_reserve_min must be non-negative.")
    if int(runtime_cfg.get("ram_reserve_gb_min", 16)) < 0:
        raise ValueError("runtime.ram_reserve_gb_min must be non-negative.")
    if int(runtime_cfg.get("nested_blas_threads", 1)) < 1:
        raise ValueError("runtime.nested_blas_threads must be at least 1.")
    orchestration_cfg = runtime_cfg.get("orchestration", {})
    if not isinstance(orchestration_cfg, dict):
        raise ValueError("runtime.orchestration must be a mapping.")
    if orchestration_cfg.get("mode", "serial") not in {"serial", "overlap"}:
        raise ValueError("runtime.orchestration.mode must be one of: serial, overlap.")
    if int(runtime_cfg.get("progress_interval_seconds", 60)) < 1:
        raise ValueError("runtime.progress_interval_seconds must be at least 1.")

    hpo_cfg = config.get("models", {}).get("hpo", {})
    if int(hpo_cfg.get("n_trials", 50)) < 1:
        raise ValueError("models.hpo.n_trials must be at least 1.")
    if int(hpo_cfg.get("tabular_mlp_epochs", 100)) < 1:
        raise ValueError("models.hpo.tabular_mlp_epochs must be at least 1.")
    if int(hpo_cfg.get("sequence_epochs", 150)) < 1:
        raise ValueError("models.hpo.sequence_epochs must be at least 1.")
    if int(hpo_cfg.get("sequence_patience", 15)) < 1:
        raise ValueError("models.hpo.sequence_patience must be at least 1.")
    if int(hpo_cfg.get("sequence_batch_size", 1024)) < 1:
        raise ValueError("models.hpo.sequence_batch_size must be at least 1.")

    ag_cfg = config.get("models", {}).get("autogluon", {})
    if ag_cfg:
        num_gpus = ag_cfg.get("num_gpus", "auto")
        if num_gpus != "auto":
            try:
                numeric_num_gpus = float(num_gpus)
            except (TypeError, ValueError) as exc:
                raise ValueError("models.autogluon.num_gpus must be 'auto' or a non-negative integer.") from exc
            if numeric_num_gpus < 0 or not numeric_num_gpus.is_integer():
                raise ValueError("models.autogluon.num_gpus must be 'auto' or a non-negative integer.")

    report_sections = config.get("reports", {}).get("manuscript_sections", [])
    unknown_sections = sorted(set(report_sections) - set(MANUSCRIPT_SECTIONS))
    if unknown_sections:
        raise ValueError(f"Unknown reports.manuscript_sections values: {unknown_sections}")

    shap_jobs = config.get("reports", {}).get("shap_jobs", [])
    for job in shap_jobs:
        model_key = job.get("model_key")
        if model_key not in SUPPORTED_SHAP_MODELS:
            raise ValueError(
                f"Unsupported SHAP model_key '{model_key}'. Supported SHAP models: {list(SUPPORTED_SHAP_MODELS)}."
            )
        if "dataset_regime" not in job:
            raise ValueError(f"SHAP job is missing dataset_regime: {job}")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    base_cfg = load_yaml(DEFAULT_CONFIG_PATH)
    merged_cfg = base_cfg
    if config_path is not None:
        override_path = Path(config_path)
        if not override_path.is_absolute():
            override_path = REPO_ROOT / override_path
        override_cfg = load_yaml(override_path)
        merged_cfg = _deep_merge(base_cfg, override_cfg)
    normalized_cfg = _normalize_config(merged_cfg)
    validate_config(normalized_cfg)
    return normalized_cfg


def config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]
