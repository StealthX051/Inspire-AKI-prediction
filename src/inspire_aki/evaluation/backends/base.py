from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class EvaluationBackendResult:
    manifest: pd.DataFrame
    overlap_audit: pd.DataFrame
    raw_predictions: pd.DataFrame
    thresholds: pd.DataFrame
    runtime_breakdown: pd.DataFrame


class EvaluationBackend(Protocol):
    mode: str

    def build(self, df: pd.DataFrame, *, target: str, dataset_family: str) -> EvaluationBackendResult:
        ...
