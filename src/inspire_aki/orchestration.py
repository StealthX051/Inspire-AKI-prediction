from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, TextIO

from inspire_aki.config import REPO_ROOT


_STAGE_COMMANDS: dict[str, tuple[str, ...]] = {
    "tune_sequence": ("tune", "sequence"),
    "train_tabular": ("train", "tabular"),
}


@dataclass
class StageSubprocess:
    stage: str
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: TextIO
    started_at: float


@dataclass
class StageSubprocessResult:
    stage: str
    returncode: int
    wall_time_seconds: float
    log_path: Path
    payload: dict[str, Any]


class OverlapInterruptedError(KeyboardInterrupt):
    def __init__(self, results: dict[str, StageSubprocessResult]) -> None:
        super().__init__("Overlapped subprocess stages interrupted.")
        self.results = results


def _stage_command(stage: str, config_path: str | None) -> list[str]:
    if stage not in _STAGE_COMMANDS:
        raise ValueError(f"Unsupported subprocess stage '{stage}'.")
    command = [sys.executable, "-m", "inspire_aki", *_STAGE_COMMANDS[stage]]
    if config_path is not None:
        command.extend(["--config", config_path])
    return command


def _parse_payload(log_path: Path) -> dict[str, Any]:
    raw = log_path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else {"payload": payload}
    except json.JSONDecodeError:
        return {"log_path": str(log_path)}


def launch_stage_subprocess(stage: str, *, config_path: str | None, log_path: Path) -> StageSubprocess:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    process = subprocess.Popen(
        _stage_command(stage, config_path),
        cwd=REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    return StageSubprocess(
        stage=stage,
        process=process,
        log_path=log_path,
        log_handle=log_handle,
        started_at=perf_counter(),
    )


def finalize_stage_subprocess(handle: StageSubprocess) -> StageSubprocessResult:
    returncode = handle.process.wait()
    handle.log_handle.flush()
    handle.log_handle.close()
    return StageSubprocessResult(
        stage=handle.stage,
        returncode=returncode,
        wall_time_seconds=perf_counter() - handle.started_at,
        log_path=handle.log_path,
        payload=_parse_payload(handle.log_path) if returncode == 0 else {"log_path": str(handle.log_path)},
    )


def terminate_stage_subprocess(handle: StageSubprocess) -> None:
    if handle.process.poll() is not None:
        return
    handle.process.terminate()
    try:
        handle.process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        handle.process.kill()
        handle.process.wait(timeout=5)


def run_overlap_stages(stages: list[str], *, config_path: str | None, log_dir: Path) -> dict[str, StageSubprocessResult]:
    handles = {
        stage: launch_stage_subprocess(stage, config_path=config_path, log_path=log_dir / f"{stage}.log")
        for stage in stages
    }
    results: dict[str, StageSubprocessResult] = {}
    try:
        while handles:
            completed = [stage for stage, handle in handles.items() if handle.process.poll() is not None]
            if not completed:
                sleep(0.1)
                continue
            for stage in completed:
                handle = handles.pop(stage)
                result = finalize_stage_subprocess(handle)
                results[stage] = result
                if result.returncode != 0:
                    for remaining_stage, remaining_handle in list(handles.items()):
                        terminate_stage_subprocess(remaining_handle)
                        results[remaining_stage] = finalize_stage_subprocess(remaining_handle)
                        handles.pop(remaining_stage)
                    return results
        return results
    except KeyboardInterrupt as exc:
        for remaining_stage, remaining_handle in list(handles.items()):
            terminate_stage_subprocess(remaining_handle)
            results[remaining_stage] = finalize_stage_subprocess(remaining_handle)
            handles.pop(remaining_stage)
        raise OverlapInterruptedError(results) from exc
