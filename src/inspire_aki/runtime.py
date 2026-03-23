from __future__ import annotations

import os
from typing import Any


def worker_count(config: dict[str, Any] | None = None) -> int:
    runtime_cfg = config.get("runtime", {}) if isinstance(config, dict) else {}
    reserve = int(runtime_cfg.get("cpu_reserve", 2))
    min_workers = int(runtime_cfg.get("min_worker_count", 1))
    detected = os.cpu_count() or 1
    return max(min_workers, detected - reserve)


def configure_torch_threads(config: dict[str, Any] | None = None) -> int | None:
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency guard
        return None

    count = worker_count(config)
    torch.set_num_threads(count)
    return count
