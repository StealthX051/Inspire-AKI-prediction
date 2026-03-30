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
DEFAULT_COHORT_KEY = "default_noncardiac_adult"
DEFAULT_OUTCOME_KEY = "aki"
KNOWN_RAW_SOURCES = {"operations", "diagnosis", "labs", "ward_vitals"}
KNOWN_OUTCOME_KINDS = {"aki", "diagnosis_window", "time_comparison", "composite"}


def _default_cohort_profiles() -> dict[str, Any]:
    return {
        DEFAULT_COHORT_KEY: {
            "preop_window_days": 90,
            "postop_windows_days": [2, 7],
            "max_preop_creatinine": 4.5,
            "min_age": 18,
            "max_asa_exclusive": 6,
            "exclude_antype": ["Regional"],
            "department_include": [],
            "department_exclude": ["PED"],
            "include_icd10_prefixes": [],
            "exclude_icd10_prefixes": ["10", "0TY", "B50", "B51"],
            "require_height_weight": True,
            "require_positive_op_len": True,
            "cardiovascular_prefix": "I",
            "creatinine_item_name": "creatinine",
            "dialysis_item_name": "crrt",
        }
    }


def _default_outcome_catalog() -> dict[str, Any]:
    return {
        "aki": {
            "kind": "aki",
            "target_column": "aki_boolean",
            "display_name": "Postoperative AKI",
            "positive_label": "Postoperative AKI",
            "negative_label": "No postoperative AKI",
            "required_sources": ["labs", "ward_vitals"],
        },
        "macce": {
            "kind": "composite",
            "target_column": "macce",
            "display_name": "30-day MACCE",
            "positive_label": "30-day MACCE",
            "negative_label": "No 30-day MACCE",
            "required_sources": ["diagnosis"],
            "window_days": 30,
            "component_keys": [
                "macce_mi",
                "macce_stroke",
                "macce_angina",
                "macce_hf",
                "macce_cardiac_arrest",
            ],
            "component_diagnosis_prefixes": {
                "macce_mi": ["I21"],
                "macce_stroke": ["I63"],
                "macce_angina": ["I20"],
                "macce_hf": ["I50"],
                "macce_cardiac_arrest": ["I46"],
            },
        },
        "pna": {
            "kind": "diagnosis_window",
            "target_column": "pna",
            "display_name": "30-day PNA",
            "positive_label": "30-day PNA",
            "negative_label": "No 30-day PNA",
            "required_sources": ["diagnosis"],
            "window_days": 30,
            "diagnosis_prefixes": ["J11", "J12", "J13", "J14", "J15", "J16", "J17", "J18"],
        },
        "pe": {
            "kind": "diagnosis_window",
            "target_column": "pe",
            "display_name": "30-day PE",
            "positive_label": "30-day PE",
            "negative_label": "No 30-day PE",
            "required_sources": ["diagnosis"],
            "window_days": 30,
            "diagnosis_prefixes": ["I26"],
        },
        "postop_icu_admission": {
            "kind": "time_comparison",
            "target_column": "postop_icu_admission",
            "display_name": "Postoperative ICU Admission",
            "positive_label": "Postoperative ICU Admission",
            "negative_label": "No postoperative ICU Admission",
            "required_sources": ["operations"],
            "source_column": "icuin_time",
            "reference_column": "opend_time",
            "comparison_rule": "strictly_after",
        },
        "postop_mortality_30d": {
            "kind": "time_comparison",
            "target_column": "postop_mortality_30d",
            "display_name": "30-day Mortality",
            "positive_label": "30-day Mortality",
            "negative_label": "No 30-day Mortality",
            "required_sources": ["operations"],
            "source_column": "allcause_death_time",
            "reference_column": "opend_time",
            "comparison_rule": "strictly_after_within_window",
            "window_days": 30,
        },
    }


def _infer_outcome_key_from_target(target: Any, catalog: dict[str, Any]) -> str | None:
    if not target:
        return None
    target_name = str(target)
    for outcome_key, outcome_cfg in catalog.items():
        if str(outcome_cfg.get("target_column")) == target_name:
            return outcome_key
    return None


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
    normalized.setdefault("evaluation_mode", "legacy_repeated_cv")
    normalized.setdefault("study", {})

    cohorts_cfg = normalized.setdefault("cohorts", {})
    legacy_cohort = normalized.get("cohort", {})
    default_profiles = _default_cohort_profiles()
    if isinstance(legacy_cohort, dict) and legacy_cohort:
        default_profiles[DEFAULT_COHORT_KEY] = _deep_merge(default_profiles[DEFAULT_COHORT_KEY], legacy_cohort)
    existing_profiles = cohorts_cfg.get("profiles", {})
    if not isinstance(existing_profiles, dict):
        existing_profiles = {}
    cohorts_cfg["profiles"] = _deep_merge(default_profiles, existing_profiles)

    outcomes_cfg = normalized.setdefault("outcomes", {})
    existing_catalog = outcomes_cfg.get("catalog", {})
    if not isinstance(existing_catalog, dict):
        existing_catalog = {}
    outcomes_cfg["catalog"] = _deep_merge(_default_outcome_catalog(), existing_catalog)

    study_cfg = normalized.setdefault("study", {})
    if not study_cfg.get("cohort_key"):
        study_cfg["cohort_key"] = DEFAULT_COHORT_KEY

    models_cfg = normalized.setdefault("models", {})
    resolved_outcome_key = study_cfg.get("outcome_key")
    if not resolved_outcome_key:
        resolved_outcome_key = _infer_outcome_key_from_target(models_cfg.get("target"), outcomes_cfg["catalog"]) or DEFAULT_OUTCOME_KEY
        study_cfg["outcome_key"] = resolved_outcome_key

    normalized["cohort"] = copy.deepcopy(cohorts_cfg["profiles"].get(study_cfg["cohort_key"], {}))
    normalized["outcome"] = copy.deepcopy(outcomes_cfg["catalog"].get(study_cfg["outcome_key"], {}))
    if "target" not in models_cfg or models_cfg.get("target") in {None, ""}:
        models_cfg["target"] = normalized["outcome"].get("target_column")

    reports_cfg = normalized.setdefault("reports", {})
    if "batch_shap_jobs" in reports_cfg:
        reports_cfg["shap_jobs"] = copy.deepcopy(reports_cfg.pop("batch_shap_jobs"))
    else:
        reports_cfg.pop("batch_shap_jobs", None)
    legacy_figure_dpi = reports_cfg.get("figure_dpi")
    if "figure_png_dpi" not in reports_cfg and legacy_figure_dpi is not None:
        reports_cfg["figure_png_dpi"] = legacy_figure_dpi
    reports_cfg.setdefault("figure_png_dpi", 600)
    reports_cfg["figure_dpi"] = reports_cfg["figure_png_dpi"]
    reports_cfg.setdefault("table_formats", ["html", "md", "csv"])
    reports_cfg.setdefault("figure_formats", ["png", "svg"])
    reports_cfg.setdefault("style_variant", "legacy_manuscript")
    reports_cfg.setdefault("highlight_best_values", True)
    reports_cfg.setdefault("generate_supplemental_outputs", True)
    reports_cfg.setdefault("shap_jobs", [])
    reports_cfg.setdefault("manuscript_sections", list(MANUSCRIPT_SECTIONS))

    compat_cfg = normalized.get("compat")
    if isinstance(compat_cfg, dict):
        compat_cfg.pop("export_legacy_aliases", None)
        if not compat_cfg:
            normalized.pop("compat", None)

    return normalized


def validate_config(config: dict[str, Any]) -> None:
    evaluation_mode = config.get("evaluation_mode", "legacy_repeated_cv")
    if evaluation_mode not in {"legacy_repeated_cv", "grouped_holdout", "grouped_nested_cv"}:
        raise ValueError("evaluation_mode must be one of: legacy_repeated_cv, grouped_holdout, grouped_nested_cv.")

    study_cfg = config.get("study", {})
    cohort_key = study_cfg.get("cohort_key", DEFAULT_COHORT_KEY)
    outcome_key = study_cfg.get("outcome_key", DEFAULT_OUTCOME_KEY)
    cohort_profiles = config.get("cohorts", {}).get("profiles", {})
    if cohort_key not in cohort_profiles:
        raise ValueError(f"Unknown study.cohort_key '{cohort_key}'.")
    outcome_catalog = config.get("outcomes", {}).get("catalog", {})
    if outcome_key not in outcome_catalog:
        raise ValueError(f"Unknown study.outcome_key '{outcome_key}'.")
    outcome_cfg = config.get("outcome", {})
    if config.get("models", {}).get("target") != outcome_cfg.get("target_column"):
        raise ValueError(
            "models.target must match the active outcome target_column derived from study.outcome_key."
        )
    required_sources = outcome_cfg.get("required_sources", [])
    unknown_sources = sorted(set(required_sources) - KNOWN_RAW_SOURCES)
    if unknown_sources:
        raise ValueError(f"Unknown outcomes.catalog required_sources values: {unknown_sources}")
    outcome_kind = outcome_cfg.get("kind")
    if outcome_kind not in KNOWN_OUTCOME_KINDS:
        raise ValueError(f"Unsupported outcomes.catalog kind '{outcome_kind}'.")
    if outcome_kind == "diagnosis_window":
        if not outcome_cfg.get("diagnosis_prefixes"):
            raise ValueError("diagnosis_window outcomes require diagnosis_prefixes.")
        if int(outcome_cfg.get("window_days", 0)) < 1:
            raise ValueError("diagnosis_window outcomes require window_days >= 1.")
    elif outcome_kind == "time_comparison":
        comparison_rule = outcome_cfg.get("comparison_rule")
        if comparison_rule not in {"strictly_after", "strictly_after_within_window"}:
            raise ValueError("time_comparison outcomes require a supported comparison_rule.")
        if not outcome_cfg.get("source_column") or not outcome_cfg.get("reference_column"):
            raise ValueError("time_comparison outcomes require source_column and reference_column.")
        if comparison_rule == "strictly_after_within_window" and int(outcome_cfg.get("window_days", 0)) < 1:
            raise ValueError("strictly_after_within_window outcomes require window_days >= 1.")
    elif outcome_kind == "composite":
        component_keys = outcome_cfg.get("component_keys", [])
        component_prefixes = outcome_cfg.get("component_diagnosis_prefixes", {})
        if not component_keys:
            raise ValueError("composite outcomes require component_keys.")
        if sorted(component_keys) != sorted(component_prefixes.keys()):
            raise ValueError("composite outcomes require component_diagnosis_prefixes for every component_key.")
        if int(outcome_cfg.get("window_days", 0)) < 1:
            raise ValueError("composite outcomes require window_days >= 1.")

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
    table_formats = config.get("reports", {}).get("table_formats", [])
    unknown_table_formats = sorted(set(table_formats) - {"html", "md", "csv"})
    if unknown_table_formats:
        raise ValueError(f"Unknown reports.table_formats values: {unknown_table_formats}")
    figure_formats = config.get("reports", {}).get("figure_formats", [])
    unknown_figure_formats = sorted(set(figure_formats) - {"png", "svg"})
    if unknown_figure_formats:
        raise ValueError(f"Unknown reports.figure_formats values: {unknown_figure_formats}")
    if int(config.get("reports", {}).get("figure_png_dpi", 600)) < 72:
        raise ValueError("reports.figure_png_dpi must be at least 72.")

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


def active_cohort_key(config: dict[str, Any]) -> str:
    return str(config["study"]["cohort_key"])


def active_outcome_key(config: dict[str, Any]) -> str:
    return str(config["study"]["outcome_key"])


def active_outcome_config(config: dict[str, Any]) -> dict[str, Any]:
    return config["outcome"]


def active_target_column(config: dict[str, Any]) -> str:
    return str(config["models"]["target"])
