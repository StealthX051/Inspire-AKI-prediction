from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspire_aki.config import config_hash


def _git_sha(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def build_manifest(
    *,
    stage: str,
    repo_root: Path,
    config: dict[str, Any],
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(repo_root),
        "config_hash": config_hash(config),
        "inputs": inputs or [],
        "outputs": outputs or [],
        "metadata": metadata or {},
    }
