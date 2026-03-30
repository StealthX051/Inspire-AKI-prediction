from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from inspire_aki.evaluation.backends.base import EvaluationBackendResult
from inspire_aki.evaluation.split_manager import build_grouped_holdout_manifest


@dataclass
class GroupedHoldoutBackend:
    config: dict
    mode: str = "grouped_holdout"

    def build(self, df: pd.DataFrame, *, target: str, dataset_family: str) -> EvaluationBackendResult:
        bundle = build_grouped_holdout_manifest(
            df,
            target=target,
            dataset_family=dataset_family,
            holdout_fraction=float(self.config["splits"]["holdout_fraction"]),
            inner_n_splits=int(self.config["splits"]["n_cv_folds"]),
            random_state=int(self.config["splits"]["random_state"]),
        )
        empty = pd.DataFrame()
        return EvaluationBackendResult(
            manifest=bundle.manifest,
            overlap_audit=bundle.overlap_audit,
            raw_predictions=empty,
            thresholds=empty,
            runtime_breakdown=empty,
        )
