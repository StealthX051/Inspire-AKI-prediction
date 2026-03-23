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
