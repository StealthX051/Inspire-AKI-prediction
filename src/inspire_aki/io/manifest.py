from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inspire_aki.config import config_hash
from inspire_aki.runtime import build_stage_runtime_plan, detect_system_resources


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
    stage_runtime_plan: dict[str, Any] | None = None,
    wall_time_seconds: float | None = None,
) -> dict[str, Any]:
    system_resources = detect_system_resources().as_dict()
    runtime_plan = stage_runtime_plan or build_stage_runtime_plan(config, stage).as_dict()
    return {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(repo_root),
        "config_hash": config_hash(config),
        "runtime_profile": config.get("runtime", {}).get("profile", "balanced"),
        "system_resources": system_resources,
        "stage_runtime_plan": runtime_plan,
        "wall_time_seconds": wall_time_seconds,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "metadata": metadata or {},
    }
