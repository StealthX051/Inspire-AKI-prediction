from __future__ import annotations

from inspire_aki.evaluation.backends.base import EvaluationBackend, EvaluationBackendResult
from inspire_aki.evaluation.backends.grouped_holdout import GroupedHoldoutBackend
from inspire_aki.evaluation.backends.grouped_nested_cv import GroupedNestedCVBackend
from inspire_aki.evaluation.backends.legacy_repeated_cv import LegacyRepeatedCVBackend

def build_evaluation_backend(config: dict) -> EvaluationBackend:
    mode = config.get("evaluation_mode", "legacy_repeated_cv")
    if mode == "legacy_repeated_cv":
        return LegacyRepeatedCVBackend(config)
    if mode == "grouped_holdout":
        return GroupedHoldoutBackend(config)
    if mode == "grouped_nested_cv":
        return GroupedNestedCVBackend(config)
    raise ValueError(f"Unsupported evaluation_mode '{mode}'.")


__all__ = ["EvaluationBackend", "EvaluationBackendResult", "build_evaluation_backend"]
