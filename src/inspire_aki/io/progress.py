from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic
from typing import Any

from inspire_aki.config import config_hash
from inspire_aki.io.artifacts import ArtifactManager


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProgressLogger:
    artifacts: ArtifactManager
    log_parts: tuple[str, ...]
    stdout: bool = True
    _heartbeat_times: dict[str, float] = field(default_factory=dict)

    @property
    def path(self):
        return self.artifacts.resolve(*self.log_parts)

    def emit_event(
        self,
        *,
        event_type: str,
        stage: str,
        status: str | None = None,
        message: str | None = None,
        wall_time_seconds: float | None = None,
        stdout_message: str | None = None,
        to_stdout: bool | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "event_type": event_type,
            "stage": stage,
            "status": status,
            "message": message,
            "timestamp_utc": _timestamp_utc(),
            "pid": os.getpid(),
            "config_hash": config_hash(self.artifacts.config),
            "artifacts_dir": str(self.artifacts.paths.artifacts_root),
        }
        if wall_time_seconds is not None:
            record["wall_time_seconds"] = wall_time_seconds
        record.update(payload)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"{json.dumps(record, sort_keys=True, default=str)}\n")
        if to_stdout is None:
            to_stdout = self.stdout and stdout_message is not None
        if to_stdout:
            print(stdout_message or self._default_stdout_message(record), flush=True)
        return record

    def stage_start(self, stage: str, *, message: str | None = None, stdout_message: str | None = None, **payload: Any) -> dict[str, Any]:
        default_stdout = stdout_message or f"[{_timestamp_utc()}] START {stage}"
        return self.emit_event(
            event_type="stage_start",
            stage=stage,
            status="started",
            message=message,
            stdout_message=default_stdout,
            **payload,
        )

    def stage_end(
        self,
        stage: str,
        *,
        wall_time_seconds: float | None = None,
        message: str | None = None,
        stdout_message: str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        duration = f" in {wall_time_seconds:.2f}s" if wall_time_seconds is not None else ""
        default_stdout = stdout_message or f"[{_timestamp_utc()}] END {stage}{duration}"
        return self.emit_event(
            event_type="stage_end",
            stage=stage,
            status="completed",
            message=message,
            wall_time_seconds=wall_time_seconds,
            stdout_message=default_stdout,
            **payload,
        )

    def stage_error(
        self,
        stage: str,
        *,
        error: str,
        status: str = "error",
        wall_time_seconds: float | None = None,
        stdout_message: str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        label = "ABORT" if status == "aborted" else "ERROR"
        default_stdout = stdout_message or f"[{_timestamp_utc()}] {label} {stage}: {error}"
        return self.emit_event(
            event_type="stage_error",
            stage=stage,
            status=status,
            message=error,
            wall_time_seconds=wall_time_seconds,
            stdout_message=default_stdout,
            **payload,
        )

    def heartbeat(
        self,
        *,
        stage: str,
        heartbeat_key: str = "default",
        interval_seconds: int = 60,
        stdout_message: str | None = None,
        **payload: Any,
    ) -> dict[str, Any] | None:
        now = monotonic()
        last = self._heartbeat_times.get(heartbeat_key)
        if last is not None and (now - last) < interval_seconds:
            return None
        self._heartbeat_times[heartbeat_key] = now
        return self.emit_event(
            event_type="heartbeat",
            stage=stage,
            status="running",
            stdout_message=stdout_message,
            **payload,
        )

    @staticmethod
    def _default_stdout_message(record: dict[str, Any]) -> str:
        message = record.get("message")
        if message:
            return f"[{record['timestamp_utc']}] {record['event_type']} {record['stage']}: {message}"
        return f"[{record['timestamp_utc']}] {record['event_type']} {record['stage']}"
