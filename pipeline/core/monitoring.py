"""
Monitoramento de hardware/recursos por etapa do pipeline — spec monitoring.md.

Coleta tempo (perf_counter), % CPU e memória usada (psutil), de forma
estruturada e serializável em JSON.

Nenhum print/input aqui — toda I/O fica em cli.py / run_eval.py (ver AGENTS.md).
"""

from __future__ import annotations

import time
from typing import Any

import psutil


def _now_ms() -> float:
    return time.perf_counter() * 1000.0


def snapshot(stage: str) -> dict[str, Any]:
    """Captura um ponto único de métricas, marcado com o nome da etapa.

    Nota: não mede duração (use Profiler para isso); registra apenas o
    estado instantâneo de CPU/RAM.
    """
    return {
        "stage": stage,
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_mb": psutil.virtual_memory().used / (1024 * 1024),
    }


class Profiler:
    """
    Context manager que mede a duração de uma etapa e coleta CPU/RAM.

    Uso:
        profiler = Profiler("retrieval")
        with profiler:
            ...
        metrics = profiler.result
    """

    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.result: dict[str, Any] | None = None

    def __enter__(self) -> "Profiler":
        self._start_ms = _now_ms()
        return self

    def __exit__(self, *exc: Any) -> None:
        elapsed_ms = _now_ms() - self._start_ms
        self.result = {
            "stage": self.stage,
            "elapsed_ms": elapsed_ms,
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_mb": psutil.virtual_memory().used / (1024 * 1024),
        }