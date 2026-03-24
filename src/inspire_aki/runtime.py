from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - dependency guard
    threadpool_limits = None


_PROFILE_CHOICES = {"balanced", "aggressive", "conservative"}


@dataclass(frozen=True)
class SystemResources:
    cpu_count: int
    total_ram_gb: int
    available_ram_gb: int
    gpu_available: bool
    gpu_name: str | None
    gpu_total_memory_gb: int | None
    gpu_free_memory_gb: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StageRuntimePlan:
    stage: str
    profile: str
    usable_cpus: int
    usable_ram_gb: int
    csv_read_threads: int
    preop_feature_workers: int
    tabular_column_workers: int
    timeseries_workers: int
    timeseries_partitions: int
    sequence_workers: int
    sequence_partitions: int
    evaluation_workers: int
    bootstrap_workers: int
    report_workers: int
    shap_workers: int
    train_model_threads: int
    hpo_model_threads: int
    dataloader_workers: int
    torch_num_threads: int
    nested_blas_threads: int
    gpu_enabled: bool
    sequence_use_gpu: bool
    xgb_use_gpu: bool
    max_concurrent_gpu_jobs: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ceil_fraction(value: int, fraction: float) -> int:
    return int(math.ceil(value * fraction))


def _resolve_int(value: Any, default: int) -> int:
    if value in {None, "auto"}:
        return int(default)
    return int(value)


def _resolve_bool(value: Any, default: bool) -> bool:
    if value in {None, "auto"}:
        return bool(default)
    return bool(value)


def _linux_meminfo() -> tuple[int, int] | None:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None
    parsed: dict[str, int] = {}
    with meminfo_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            key, _, value = line.partition(":")
            parsed[key] = int(value.strip().split()[0])
    total_kb = parsed.get("MemTotal")
    available_kb = parsed.get("MemAvailable")
    if total_kb is None or available_kb is None:
        return None
    return max(1, total_kb // (1024 * 1024)), max(1, available_kb // (1024 * 1024))


def _sysconf_meminfo() -> tuple[int, int]:
    page_size = os.sysconf("SC_PAGE_SIZE")
    phys_pages = os.sysconf("SC_PHYS_PAGES")
    total_bytes = page_size * phys_pages
    total_gb = max(1, int(total_bytes / (1024**3)))
    return total_gb, total_gb


def _gpu_resources() -> tuple[bool, str | None, int | None, int | None]:
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency guard
        return False, None, None, None

    if not torch.cuda.is_available():
        return False, None, None, None
    try:
        device_idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
    except Exception:  # pragma: no cover - device-specific fallback
        return True, None, None, None
    return (
        True,
        props.name,
        max(1, int(total_bytes / (1024**3))),
        max(1, int(free_bytes / (1024**3))),
    )


def detect_system_resources() -> SystemResources:
    cpu_count = os.cpu_count() or 1
    meminfo = _linux_meminfo()
    if meminfo is None:
        total_ram_gb, available_ram_gb = _sysconf_meminfo()
    else:
        total_ram_gb, available_ram_gb = meminfo
    gpu_available, gpu_name, gpu_total_memory_gb, gpu_free_memory_gb = _gpu_resources()
    return SystemResources(
        cpu_count=cpu_count,
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_total_memory_gb=gpu_total_memory_gb,
        gpu_free_memory_gb=gpu_free_memory_gb,
    )


def _profile_adjustment(profile: str) -> tuple[float, float]:
    if profile == "aggressive":
        return 1.25, 1.25
    if profile == "conservative":
        return 0.75, 0.75
    return 1.0, 1.0


def _auto_plan(resources: SystemResources, config: dict[str, Any], workload_hint: dict[str, Any] | None = None) -> StageRuntimePlan:
    runtime_cfg = config.get("runtime", {})
    stage_cfg = runtime_cfg.get("stages", {})
    gpu_cfg = runtime_cfg.get("gpu", {})
    profile = runtime_cfg.get("profile", "balanced")
    cpu_factor, memory_factor = _profile_adjustment(profile)

    cpu_reserve = max(
        int(runtime_cfg.get("cpu_reserve_min", 4)),
        _ceil_fraction(resources.cpu_count, float(runtime_cfg.get("cpu_reserve_fraction", 0.125))),
    )
    ram_reserve_gb = max(
        int(runtime_cfg.get("ram_reserve_gb_min", 16)),
        _ceil_fraction(resources.total_ram_gb, float(runtime_cfg.get("ram_reserve_fraction", 0.15))),
    )
    usable_cpus = max(1, resources.cpu_count - cpu_reserve)
    usable_ram_gb = max(1, resources.available_ram_gb - ram_reserve_gb)

    csv_read_threads = min(8, max(1, int(round((usable_cpus / 2) * cpu_factor))))
    preop_feature_workers = min(4, max(1, int(round((usable_cpus / 7) * cpu_factor))))
    tabular_column_workers = min(8, max(1, int(round((usable_cpus / 3) * cpu_factor))))
    timeseries_workers = min(8, max(1, int(round((usable_cpus / 3) * cpu_factor))))
    timeseries_partitions = min(64, max(4, int(round((usable_cpus + 4) * memory_factor))))
    sequence_workers = min(8, max(1, int(round((usable_cpus / 3) * cpu_factor))))
    sequence_partitions = min(64, max(4, int(round((usable_cpus + 4) * memory_factor))))
    evaluation_workers = min(8, max(1, int(round((usable_cpus / 3) * cpu_factor))))
    report_workers = min(4, max(1, int(round((usable_cpus / 7) * cpu_factor))))
    shap_workers = min(2, max(1, int(round((usable_cpus / 14) * cpu_factor))))
    train_model_threads = min(16, max(2, int(round(((resources.cpu_count / 2)) * cpu_factor))))
    hpo_model_threads = min(8, max(2, int(round((usable_cpus / 3) * cpu_factor))))
    dataloader_workers = min(7, max(1, sequence_workers - 1))
    torch_num_threads = min(8, max(1, int(round((usable_cpus / 3) * cpu_factor))))
    bootstrap_workers = min(8, max(1, int(round((usable_cpus / 3) * cpu_factor))))

    group_count = int((workload_hint or {}).get("group_count", 0))
    if group_count >= 4:
        bootstrap_workers = 1

    nested_blas_threads = max(1, int(runtime_cfg.get("nested_blas_threads", 1)))
    gpu_enabled = _resolve_bool(gpu_cfg.get("enabled", "auto"), resources.gpu_available)
    sequence_use_gpu = gpu_enabled and _resolve_bool(gpu_cfg.get("sequence_use_gpu", "auto"), resources.gpu_available)
    xgb_use_gpu = gpu_enabled and _resolve_bool(gpu_cfg.get("xgb_use_gpu", False), False)
    max_concurrent_gpu_jobs = max(1, int(gpu_cfg.get("max_concurrent_jobs", 1)))

    plan = StageRuntimePlan(
        stage=str((workload_hint or {}).get("stage", "default")),
        profile=profile,
        usable_cpus=usable_cpus,
        usable_ram_gb=usable_ram_gb,
        csv_read_threads=_resolve_int(stage_cfg.get("csv_read_threads", "auto"), csv_read_threads),
        preop_feature_workers=_resolve_int(stage_cfg.get("preop_feature_workers", "auto"), preop_feature_workers),
        tabular_column_workers=_resolve_int(stage_cfg.get("tabular_column_workers", "auto"), tabular_column_workers),
        timeseries_workers=_resolve_int(stage_cfg.get("timeseries_workers", "auto"), timeseries_workers),
        timeseries_partitions=_resolve_int(stage_cfg.get("timeseries_partitions", "auto"), timeseries_partitions),
        sequence_workers=_resolve_int(stage_cfg.get("sequence_workers", "auto"), sequence_workers),
        sequence_partitions=_resolve_int(stage_cfg.get("sequence_partitions", "auto"), sequence_partitions),
        evaluation_workers=_resolve_int(stage_cfg.get("evaluation_workers", "auto"), evaluation_workers),
        bootstrap_workers=_resolve_int(stage_cfg.get("bootstrap_workers", "auto"), bootstrap_workers),
        report_workers=_resolve_int(stage_cfg.get("report_workers", "auto"), report_workers),
        shap_workers=_resolve_int(stage_cfg.get("shap_workers", "auto"), shap_workers),
        train_model_threads=_resolve_int(stage_cfg.get("train_model_threads", "auto"), train_model_threads),
        hpo_model_threads=_resolve_int(stage_cfg.get("hpo_model_threads", "auto"), hpo_model_threads),
        dataloader_workers=_resolve_int(stage_cfg.get("dataloader_workers", "auto"), dataloader_workers),
        torch_num_threads=_resolve_int(stage_cfg.get("torch_num_threads", "auto"), torch_num_threads),
        nested_blas_threads=nested_blas_threads,
        gpu_enabled=gpu_enabled,
        sequence_use_gpu=sequence_use_gpu,
        xgb_use_gpu=xgb_use_gpu,
        max_concurrent_gpu_jobs=max_concurrent_gpu_jobs,
    )
    return plan


def build_stage_runtime_plan(config: dict[str, Any], stage: str, workload_hint: dict[str, Any] | None = None) -> StageRuntimePlan:
    profile = config.get("runtime", {}).get("profile", "balanced")
    if profile not in _PROFILE_CHOICES:
        raise ValueError(f"Unknown runtime.profile '{profile}'. Expected one of {sorted(_PROFILE_CHOICES)}.")
    resources = detect_system_resources()
    hint = dict(workload_hint or {})
    hint["stage"] = stage
    return _auto_plan(resources, config, hint)


@contextmanager
def thread_limited_context(limits: int) -> Iterator[None]:
    limit = max(1, int(limits))
    env_keys = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_MAX_THREADS")
    previous = {key: os.environ.get(key) for key in env_keys}
    try:
        for key in env_keys:
            os.environ[key] = str(limit)
        if threadpool_limits is None:
            yield
        else:
            with threadpool_limits(limits=limit):
                yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def runtime_summary(config: dict[str, Any]) -> dict[str, Any]:
    stage_names = [
        "preprocess_preop",
        "preprocess_tabular",
        "preprocess_timeseries",
        "preprocess_sequence",
        "train_tabular",
        "train_sequence",
        "tune_tabular",
        "tune_sequence",
        "evaluate_metrics",
        "report_curves",
        "report_shap",
    ]
    resources = detect_system_resources()
    return {
        "runtime_profile": config.get("runtime", {}).get("profile", "balanced"),
        "system_resources": resources.as_dict(),
        "stage_runtime_plans": {
            stage_name: build_stage_runtime_plan(config, stage_name).as_dict()
            for stage_name in stage_names
        },
    }


def worker_count(config: dict[str, Any] | None = None) -> int:
    if not isinstance(config, dict):
        return max(1, (os.cpu_count() or 1) - 2)
    return build_stage_runtime_plan(config, "default").usable_cpus


def configure_torch_threads(config: dict[str, Any] | None = None, stage: str = "train") -> int | None:
    try:
        import torch
    except ImportError:  # pragma: no cover - optional dependency guard
        return None

    if not isinstance(config, dict):
        count = max(1, (os.cpu_count() or 1) - 2)
    else:
        plan = build_stage_runtime_plan(config, stage)
        count = plan.torch_num_threads if stage != "hpo" else min(plan.torch_num_threads, plan.hpo_model_threads)
    torch.set_num_threads(count)
    return count
