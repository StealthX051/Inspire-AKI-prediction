from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from inspire_aki.io.manifest import build_manifest
from inspire_aki.io.paths import ProjectPaths


class ArtifactManager:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.paths = ProjectPaths.from_config(config)
        self.paths.artifacts_root.mkdir(parents=True, exist_ok=True)

    def resolve(self, *parts: str) -> Path:
        path = self.paths.artifact_path(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.paths.repo_root))
        except ValueError:
            return str(path)

    def write_dataframe(self, df: pd.DataFrame, *parts: str, index: bool = False) -> Path:
        path = self.resolve(*parts)
        if path.suffix == ".csv":
            df.to_csv(path, index=index)
        else:
            df.to_parquet(path, index=index)
        return path

    def read_dataframe(self, *parts: str) -> pd.DataFrame:
        path = self.paths.artifact_path(*parts)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        return pd.read_parquet(path)

    def write_json(self, payload: dict[str, Any], *parts: str) -> Path:
        path = self.resolve(*parts)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        return path

    def read_json(self, *parts: str) -> dict[str, Any]:
        path = self.paths.artifact_path(*parts)
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def write_pickle(self, payload: Any, *parts: str) -> Path:
        path = self.resolve(*parts)
        with path.open("wb") as handle:
            pickle.dump(payload, handle)
        return path

    def read_pickle(self, *parts: str) -> Any:
        path = self.paths.artifact_path(*parts)
        with path.open("rb") as handle:
            return pickle.load(handle)

    def write_manifest(
        self,
        stage: str,
        manifest_parts: list[str],
        *,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        payload = build_manifest(
            stage=stage,
            repo_root=self.paths.repo_root,
            config=self.config,
            inputs=inputs,
            outputs=outputs,
            metadata=metadata,
        )
        return self.write_json(payload, *manifest_parts)
