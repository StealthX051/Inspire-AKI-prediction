from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def read_csv_optimized(
    path: str | Path,
    *,
    config: dict[str, Any] | None = None,
    usecols: list[str] | None = None,
    dtype_map: dict[str, Any] | None = None,
    chunksize: int | None = None,
    large: bool = False,
) -> pd.DataFrame | pd.io.parsers.TextFileReader:
    path = Path(path)
    runtime_cfg = config.get("runtime", {}) if isinstance(config, dict) else {}
    preferred_engine = runtime_cfg.get("csv_engine", "c")

    kwargs: dict[str, Any] = {}
    if usecols is not None:
        kwargs["usecols"] = usecols
    if dtype_map:
        kwargs["dtype"] = dtype_map
    if chunksize is not None:
        kwargs["chunksize"] = chunksize
        return pd.read_csv(path, **kwargs)

    if preferred_engine == "pyarrow" and large:
        try:
            return pd.read_csv(path, engine="pyarrow", **kwargs)
        except Exception:
            pass
    return pd.read_csv(path, **kwargs)
