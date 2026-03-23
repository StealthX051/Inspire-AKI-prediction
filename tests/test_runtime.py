from __future__ import annotations

from inspire_aki.runtime import worker_count


def test_worker_count_uses_cpu_minus_two(monkeypatch, loaded_synthetic_config) -> None:
    monkeypatch.setattr("inspire_aki.runtime.os.cpu_count", lambda: 10)
    assert worker_count(loaded_synthetic_config) == 8


def test_worker_count_respects_floor(monkeypatch, loaded_synthetic_config) -> None:
    monkeypatch.setattr("inspire_aki.runtime.os.cpu_count", lambda: 2)
    assert worker_count(loaded_synthetic_config) == 1


def test_worker_count_respects_runtime_override(monkeypatch, loaded_synthetic_config) -> None:
    monkeypatch.setattr("inspire_aki.runtime.os.cpu_count", lambda: 16)
    loaded_synthetic_config["runtime"]["cpu_reserve"] = 4
    loaded_synthetic_config["runtime"]["min_worker_count"] = 3
    assert worker_count(loaded_synthetic_config) == 12
