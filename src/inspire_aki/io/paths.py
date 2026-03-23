from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from inspire_aki.config import REPO_ROOT


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


@dataclass(frozen=True)
class ProjectPaths:
    repo_root: Path
    artifacts_root: Path
    raw_inspire_dir: Path
    compat_aki_dir: Path
    compat_base_dir: Path
    compat_results_dir: Path

    @classmethod
    def from_config(cls, config: dict) -> "ProjectPaths":
        path_cfg = config["paths"]
        return cls(
            repo_root=REPO_ROOT,
            artifacts_root=_resolve_path(path_cfg["artifacts_dir"]),
            raw_inspire_dir=_resolve_path(path_cfg["raw_inspire_dir"]),
            compat_aki_dir=_resolve_path(path_cfg["compat_aki_dir"]),
            compat_base_dir=_resolve_path(path_cfg["compat_base_dir"]),
            compat_results_dir=_resolve_path(path_cfg["compat_results_dir"]),
        )

    def artifact_path(self, *parts: str) -> Path:
        return self.artifacts_root.joinpath(*parts)
