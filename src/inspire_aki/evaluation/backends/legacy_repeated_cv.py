from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from inspire_aki.datasets.splits import build_bootstrap_split_manifest
from inspire_aki.evaluation.backends.base import EvaluationBackendResult
from inspire_aki.evaluation.split_manager import adapt_legacy_manifest


@dataclass
class LegacyRepeatedCVBackend:
    config: dict
    mode: str = "legacy_repeated_cv"

    def build(self, df: pd.DataFrame, *, target: str, dataset_family: str) -> EvaluationBackendResult:
        manifest = build_bootstrap_split_manifest(
            df,
            target=target,
            dataset_regime=dataset_family,
            population_id=dataset_family,
            random_state=int(self.config["splits"]["random_state"]),
            n_iterations=int(self.config["splits"]["n_bootstrap_iterations"]),
            n_cv_folds=int(self.config["splits"]["n_cv_folds"]),
            use_bootstrapping=bool(self.config["splits"]["use_bootstrapping"]),
        )
        adapted = adapt_legacy_manifest(manifest, df, target=target, dataset_family=dataset_family)
        overlap_audit = (
            adapted.groupby(["split_scope", "outer_repeat_id", "outer_fold_id", "split_name"], dropna=False)
            .agg(
                n_operations=("op_id", "nunique"),
                n_patients=("patient_id", "nunique"),
                prevalence=("y_true", "mean"),
            )
            .reset_index()
        )
        empty = pd.DataFrame()
        return EvaluationBackendResult(
            manifest=adapted,
            overlap_audit=overlap_audit,
            raw_predictions=empty,
            thresholds=empty,
            runtime_breakdown=empty,
        )
