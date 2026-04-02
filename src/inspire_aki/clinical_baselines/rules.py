from __future__ import annotations

from typing import Any

from inspire_aki.clinical_baselines.gs_aki import gs_aki_high_risk_probability_threshold


RULE_BASELINE_THRESHOLD_METRICS = (
    "sensitivity",
    "specificity",
    "precision",
    "f_score",
    "balanced_accuracy",
)


def is_clinical_rule_baseline(model_key: str) -> bool:
    return str(model_key) in {"asa_rule", "gs_aki_rule"}


def clinical_rule_probability_threshold(model_key: str, config: dict[str, Any]) -> float | None:
    normalized = str(model_key)
    if normalized == "asa_rule":
        return 0.5
    if normalized == "gs_aki_rule":
        return gs_aki_high_risk_probability_threshold(config)
    return None


def clinical_rule_calibration_method(model_key: str) -> str | None:
    normalized = str(model_key)
    if normalized == "asa_rule":
        return "identity_prespecified_asa_ge_3"
    if normalized == "gs_aki_rule":
        return "identity_prespecified_class_iii_plus"
    return None


def main_performance_table_hidden_metrics(model_key: str) -> tuple[str, ...]:
    if str(model_key) == "gs_aki_rule":
        return RULE_BASELINE_THRESHOLD_METRICS
    return ()
