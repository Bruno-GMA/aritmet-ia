"""Stage measurements with explicit process and system scopes.

psutil data is observational resource use, not electrical energy or GPU power.
Collection failures remain null and never interrupt generation/evaluation.
"""

from __future__ import annotations

import statistics
import time
from contextlib import AbstractContextManager
from typing import Any

import psutil

SCHEMA_VERSION = 1


def _read(callback: Any) -> tuple[float | int | None, str | None]:
    try:
        value = callback()
        return value, None
    except (psutil.Error, OSError, ValueError, AttributeError) as exc:
        return None, type(exc).__name__


def process_snapshot() -> dict[str, Any]:
    process = psutil.Process()
    cpu, cpu_error = _read(lambda: process.cpu_percent(interval=None))
    rss, rss_error = _read(lambda: process.memory_info().rss / (1024 * 1024))
    threads, threads_error = _read(process.num_threads)
    return {
        "scope": "python_process",
        "pid": process.pid,
        "cpu_percent": cpu,
        "rss_mb": rss,
        "num_threads": threads,
        "errors": {
            key: value
            for key, value in {
                "cpu_percent": cpu_error,
                "rss_mb": rss_error,
                "num_threads": threads_error,
            }.items()
            if value
        },
    }


def system_snapshot() -> dict[str, Any]:
    cpu, cpu_error = _read(lambda: psutil.cpu_percent(interval=None))
    ram, ram_error = _read(lambda: psutil.virtual_memory().used / (1024 * 1024))
    return {
        "scope": "system",
        "cpu_percent": cpu,
        "ram_used_mb": ram,
        "errors": {
            key: value
            for key, value in {"cpu_percent": cpu_error, "ram_used_mb": ram_error}.items()
            if value
        },
    }


class Profiler(AbstractContextManager["Profiler"]):
    """Collect one bounded stage measurement; ``result`` is JSON serializable."""

    def __init__(
        self,
        stage: str,
        *,
        run_id: str | None = None,
        query_id: str | None = None,
        mode: str | None = None,
    ):
        self.stage, self.run_id, self.query_id, self.mode = stage, run_id, query_id, mode
        self.result: dict[str, Any] | None = None

    def __enter__(self) -> Profiler:
        self._start = time.perf_counter()
        process_snapshot()  # primes process CPU accounting without using it as a measurement
        return self

    def __exit__(self, *exc: Any) -> None:
        self.result = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "query_id": self.query_id,
            "mode": self.mode,
            "stage": self.stage,
            "elapsed_ms": (time.perf_counter() - self._start) * 1000,
            "process": process_snapshot(),
            "system": system_snapshot(),
            "ollama_process": {
                "status": "not_collected",
                "reason": "separate_process_not_attributed",
            },
            "energy_joules": None,
            "gpu": None,
        }


def summarize_stage_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize measurements without turning unavailable observations into zero."""
    elapsed = [record["elapsed_ms"] for record in records if record.get("elapsed_ms") is not None]
    return {
        "n_records": len(records),
        "n_elapsed": len(elapsed),
        "elapsed_ms": {
            "mean": statistics.fmean(elapsed) if elapsed else None,
            "median": statistics.median(elapsed) if elapsed else None,
            "p95": _percentile(elapsed, 95) if elapsed else None,
            "stddev": statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0 if elapsed else None,
        },
        "scope": "per_stage_observed_client_process_and_system",
        "energy_joules": None,
        "gpu": None,
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)
