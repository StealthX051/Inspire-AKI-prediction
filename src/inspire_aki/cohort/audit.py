from __future__ import annotations

from typing import Any

import pandas as pd


def record_count(audit: list[dict[str, Any]], step: str, df: pd.DataFrame, note: str | None = None) -> list[dict[str, Any]]:
    audit.append({"step": step, "count": int(len(df)), "note": note or ""})
    return audit


def audit_frame(audit: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(audit)
