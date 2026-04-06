from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from inspire_aki.registry import MANUSCRIPT_SECTIONS, SUPPORTED_SHAP_MODELS, SUPPORTED_SHAP_PLOT_FAMILIES


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "aki" / "default.yaml"
LEGACY_NONCARDIAC_COHORT_KEY = "default_noncardiac_adult"
STRICT_NONCARDIAC_AUDIT_COHORT_KEY = "strict_noncardiac_adult_procedure_audit"
DEFAULT_COHORT_KEY = STRICT_NONCARDIAC_AUDIT_COHORT_KEY
DEFAULT_OUTCOME_KEY = "aki"
KNOWN_RAW_SOURCES = {"operations", "diagnosis", "labs", "ward_vitals"}
KNOWN_OUTCOME_KINDS = {"aki", "diagnosis_window", "time_comparison", "composite"}


def _default_procedure_audit_resolution_config(*, enabled: bool) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "exclude_audit_classes": ["cardiac_exclude"],
        "exclude_manual_review_buckets": [
            "cpb_positive_aortic_or_vascular",
            "respiratory_plus_cpb",
            "other_cpb_discordant_nonvascular_nonrespiratory",
            "other_prefix_level_review",
        ],
        "retain_unresolved_cpb_negative_with_benign_neighbor": True,
        "benign_neighbor_keywords": [
            "scalp skin",
            "sternum",
            "chest wall",
            "external ear",
            "conjunctiva",
            "breast",
        ],
    }


def _legacy_noncardiac_adult_profile() -> dict[str, Any]:
    return {
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
        "procedure_audit_resolution": _default_procedure_audit_resolution_config(enabled=False),
    }


def _strict_noncardiac_adult_profile() -> dict[str, Any]:
    strict_profile = copy.deepcopy(_legacy_noncardiac_adult_profile())
    strict_profile["exclude_icd10_prefixes"] = [*strict_profile["exclude_icd10_prefixes"], "02"]
    strict_profile["procedure_audit_resolution"] = _default_procedure_audit_resolution_config(enabled=True)
    return strict_profile


def _default_cohort_profiles() -> dict[str, Any]:
    return {
        LEGACY_NONCARDIAC_COHORT_KEY: _legacy_noncardiac_adult_profile(),
        STRICT_NONCARDIAC_AUDIT_COHORT_KEY: _strict_noncardiac_adult_profile(),
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


def _default_gs_aki_config() -> dict[str, Any]:
    return {
        "intraperitoneal_map_path": "configs/clinical_baselines/intraperitoneal_proxy_map_5char.csv",
        "diabetes_prefixes": ["E08", "E09", "E10", "E11", "E13"],
        "hypertension_prefixes": ["I10", "I11", "I12", "I13", "I15", "I16", "I1A"],
        "chf_prefixes": ["I50"],
        "ascites_prefixes": ["R18"],
        "recent_window_days": 30,
        "class_cutpoints": {
            "I": [0, 2],
            "II": [3, 3],
            "III": [4, 4],
            "IV": [5, 5],
            "V": [6, 9],
        },
        "score_max": 9,
    }


def _default_procedure_audit_config() -> dict[str, Any]:
    return {
        "cms_order_zip_path": "external/cms_icd10pcs/april-1-2026-icd10pcs-order.zip",
        "ct_department_code": "CTS",
        "definite_cardiac_prefixes": ["02"],
        "definite_thoracic_prefixes": ["0B"],
        "manual_review_if_cpb_discordant": True,
    }


def _normalize_procedure_prefix_list(values: Any) -> list[str]:
    if not isinstance(values, list | tuple):
        return []
    normalized: list[str] = []
    for value in values:
        prefix = str(value).strip().upper()
        if not prefix:
            continue
        if prefix.isdigit() and len(prefix) == 1:
            prefix = prefix.zfill(2)
        normalized.append(prefix)
    return normalized


def _normalize_string_list(values: Any) -> list[str]:
    if not isinstance(values, list | tuple):
        return []
    normalized: list[str] = []
    for value in values:
        item = str(value).strip()
        if item:
            normalized.append(item)
    return normalized


def _normalize_shap_dependence_pairs(values: Any) -> list[Any]:
    if values is None:
        return []
    if not isinstance(values, list):
        return copy.deepcopy(values)
    normalized: list[Any] = []
    for value in values:
        if not isinstance(value, dict):
            normalized.append(copy.deepcopy(value))
            continue
        normalized.append(
            {
                **copy.deepcopy(value),
                "main_feature": str(value.get("main_feature", "")).strip(),
                "interaction_feature": str(value.get("interaction_feature", "")).strip(),
            }
        )
    return normalized


def _normalize_shap_job(job: Any) -> Any:
    if not isinstance(job, dict):
        return copy.deepcopy(job)
    normalized = copy.deepcopy(job)
    normalized["plots"] = _normalize_string_list(normalized.get("plots", ["beeswarm"])) or ["beeswarm"]
    normalized["scatter_features"] = _normalize_string_list(normalized.get("scatter_features", []))
    normalized["dependence_pairs"] = _normalize_shap_dependence_pairs(normalized.get("dependence_pairs", []))
    return normalized


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

    clinical_baselines_cfg = normalized.setdefault("clinical_baselines", {})
    existing_gs_aki = clinical_baselines_cfg.get("gs_aki", {})
    if not isinstance(existing_gs_aki, dict):
        existing_gs_aki = {}
    clinical_baselines_cfg["gs_aki"] = _deep_merge(_default_gs_aki_config(), existing_gs_aki)

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

    cohort_cfg = normalized["cohort"]
    procedure_audit_resolution = cohort_cfg.get("procedure_audit_resolution", {})
    if not isinstance(procedure_audit_resolution, dict):
        procedure_audit_resolution = {}
    enabled = bool(procedure_audit_resolution.get("enabled", False))
    cohort_cfg["procedure_audit_resolution"] = _deep_merge(
        _default_procedure_audit_resolution_config(enabled=enabled),
        procedure_audit_resolution,
    )
    cohort_cfg["procedure_audit_resolution"]["enabled"] = bool(
        cohort_cfg["procedure_audit_resolution"].get("enabled", enabled)
    )
    cohort_cfg["procedure_audit_resolution"]["exclude_audit_classes"] = _normalize_string_list(
        cohort_cfg["procedure_audit_resolution"].get("exclude_audit_classes", ["cardiac_exclude"])
    )
    cohort_cfg["procedure_audit_resolution"]["exclude_manual_review_buckets"] = _normalize_string_list(
        cohort_cfg["procedure_audit_resolution"].get(
            "exclude_manual_review_buckets",
            [
                "cpb_positive_aortic_or_vascular",
                "respiratory_plus_cpb",
                "other_cpb_discordant_nonvascular_nonrespiratory",
                "other_prefix_level_review",
            ],
        )
    )
    cohort_cfg["procedure_audit_resolution"]["retain_unresolved_cpb_negative_with_benign_neighbor"] = bool(
        cohort_cfg["procedure_audit_resolution"].get("retain_unresolved_cpb_negative_with_benign_neighbor", True)
    )
    cohort_cfg["procedure_audit_resolution"]["benign_neighbor_keywords"] = _normalize_string_list(
        cohort_cfg["procedure_audit_resolution"].get(
            "benign_neighbor_keywords",
            ["scalp skin", "sternum", "chest wall", "external ear", "conjunctiva", "breast"],
        )
    )

    reports_cfg = normalized.setdefault("reports", {})
    if "batch_shap_jobs" in reports_cfg:
        reports_cfg["shap_jobs"] = copy.deepcopy(reports_cfg.pop("batch_shap_jobs"))
    else:
        reports_cfg.pop("batch_shap_jobs", None)
    if "mirror_top_level_figures_to_primary_figures" in reports_cfg and "route_top_level_figures_to_primary_figures" not in reports_cfg:
        reports_cfg["route_top_level_figures_to_primary_figures"] = bool(
            reports_cfg.pop("mirror_top_level_figures_to_primary_figures")
        )
    else:
        reports_cfg.pop("mirror_top_level_figures_to_primary_figures", None)
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
    reports_cfg.setdefault("route_top_level_figures_to_primary_figures", True)
    reports_cfg["primary_figure_subdir"] = str(reports_cfg.get("primary_figure_subdir", "primary_figures")).strip() or "primary_figures"
    reports_cfg["shap_jobs"] = [_normalize_shap_job(job) for job in reports_cfg.get("shap_jobs", [])]
    reports_cfg["featured_shap_scatter_features"] = _normalize_string_list(
        reports_cfg.get("featured_shap_scatter_features", [])
    )
    reports_cfg.setdefault("manuscript_sections", list(MANUSCRIPT_SECTIONS))
    existing_procedure_audit = reports_cfg.get("procedure_audit", {})
    if not isinstance(existing_procedure_audit, dict):
        existing_procedure_audit = {}
    reports_cfg["procedure_audit"] = _deep_merge(_default_procedure_audit_config(), existing_procedure_audit)
    reports_cfg["procedure_audit"]["ct_department_code"] = str(
        reports_cfg["procedure_audit"].get("ct_department_code", "CTS")
    ).strip().upper()
    reports_cfg["procedure_audit"]["definite_cardiac_prefixes"] = _normalize_procedure_prefix_list(
        reports_cfg["procedure_audit"].get("definite_cardiac_prefixes", ["02"])
    )
    reports_cfg["procedure_audit"]["definite_thoracic_prefixes"] = _normalize_procedure_prefix_list(
        reports_cfg["procedure_audit"].get("definite_thoracic_prefixes", ["0B"])
    )
    reports_cfg["procedure_audit"]["manual_review_if_cpb_discordant"] = bool(
        reports_cfg["procedure_audit"].get("manual_review_if_cpb_discordant", True)
    )

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

    tabular_enabled = list(config.get("models", {}).get("tabular_enabled", []))
    tabular_hpo_enabled = list(config.get("models", {}).get("tabular_hpo_enabled", []))
    if "gs_aki_rule" in tabular_hpo_enabled:
        raise ValueError("gs_aki_rule is a deterministic clinical baseline and cannot be added to models.tabular_hpo_enabled.")
    if outcome_key != DEFAULT_OUTCOME_KEY and "gs_aki_rule" in tabular_enabled:
        raise ValueError("gs_aki_rule is only supported for the AKI outcome.")
    if "gs_aki_rule" in tabular_enabled:
        gs_aki_cfg = config.get("clinical_baselines", {}).get("gs_aki", {})
        required_keys = {
            "intraperitoneal_map_path",
            "diabetes_prefixes",
            "hypertension_prefixes",
            "chf_prefixes",
            "ascites_prefixes",
            "recent_window_days",
            "class_cutpoints",
            "score_max",
        }
        missing_keys = sorted(required_keys - set(gs_aki_cfg))
        if missing_keys:
            raise ValueError(f"clinical_baselines.gs_aki is missing required keys: {missing_keys}")
        if int(gs_aki_cfg.get("recent_window_days", 0)) < 1:
            raise ValueError("clinical_baselines.gs_aki.recent_window_days must be at least 1.")
        if int(gs_aki_cfg.get("score_max", 0)) < 1:
            raise ValueError("clinical_baselines.gs_aki.score_max must be at least 1.")
        class_cutpoints = gs_aki_cfg.get("class_cutpoints", {})
        expected_classes = ["I", "II", "III", "IV", "V"]
        if list(class_cutpoints.keys()) != expected_classes:
            raise ValueError("clinical_baselines.gs_aki.class_cutpoints must contain ordered keys I, II, III, IV, V.")
        for class_name, bounds in class_cutpoints.items():
            if not isinstance(bounds, list | tuple) or len(bounds) != 2:
                raise ValueError(
                    f"clinical_baselines.gs_aki.class_cutpoints['{class_name}'] must be a [min, max] pair."
                )
            lower, upper = int(bounds[0]), int(bounds[1])
            if lower > upper:
                raise ValueError(
                    f"clinical_baselines.gs_aki.class_cutpoints['{class_name}'] must have min <= max."
                )
        map_path = Path(str(gs_aki_cfg["intraperitoneal_map_path"]))
        if not map_path.is_absolute():
            map_path = REPO_ROOT / map_path
        if not map_path.exists():
            raise ValueError(
                "clinical_baselines.gs_aki.intraperitoneal_map_path must point to an existing committed CSV when gs_aki_rule is enabled."
            )

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
    route_primary = config.get("reports", {}).get("route_top_level_figures_to_primary_figures", True)
    if not isinstance(route_primary, bool):
        raise ValueError("reports.route_top_level_figures_to_primary_figures must be a boolean.")
    primary_figure_subdir = str(config.get("reports", {}).get("primary_figure_subdir", "primary_figures")).strip()
    if not primary_figure_subdir or primary_figure_subdir in {".", ".."} or "/" in primary_figure_subdir or "\\" in primary_figure_subdir:
        raise ValueError("reports.primary_figure_subdir must be a simple directory name.")
    featured_scatter_features = config.get("reports", {}).get("featured_shap_scatter_features", [])
    if not isinstance(featured_scatter_features, list):
        raise ValueError("reports.featured_shap_scatter_features must be a list of feature names.")

    shap_jobs = config.get("reports", {}).get("shap_jobs", [])
    for job in shap_jobs:
        if not isinstance(job, dict):
            raise ValueError(f"SHAP job must be a mapping: {job!r}")
        model_key = job.get("model_key")
        if model_key not in SUPPORTED_SHAP_MODELS:
            raise ValueError(
                f"Unsupported SHAP model_key '{model_key}'. Supported SHAP models: {list(SUPPORTED_SHAP_MODELS)}."
            )
        if "dataset_regime" not in job:
            raise ValueError(f"SHAP job is missing dataset_regime: {job}")
        plots = job.get("plots", ["beeswarm"])
        if not isinstance(plots, list) or not plots:
            raise ValueError(f"SHAP job plots must be a non-empty list: {job}")
        unknown_plot_families = sorted(set(plots) - set(SUPPORTED_SHAP_PLOT_FAMILIES))
        if unknown_plot_families:
            raise ValueError(
                f"Unsupported SHAP plot families {unknown_plot_families}. "
                f"Supported plot families: {list(SUPPORTED_SHAP_PLOT_FAMILIES)}."
            )
        scatter_features = job.get("scatter_features", [])
        if not isinstance(scatter_features, list):
            raise ValueError(f"SHAP job scatter_features must be a list of feature names: {job}")
        dependence_pairs = job.get("dependence_pairs", [])
        if not isinstance(dependence_pairs, list):
            raise ValueError(f"SHAP job dependence_pairs must be a list of mappings: {job}")
        for pair in dependence_pairs:
            if not isinstance(pair, dict):
                raise ValueError(f"SHAP dependence pair must be a mapping: {pair!r}")
            if not str(pair.get("main_feature", "")).strip() or not str(pair.get("interaction_feature", "")).strip():
                raise ValueError(f"SHAP dependence pair must include main_feature and interaction_feature: {pair!r}")


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
