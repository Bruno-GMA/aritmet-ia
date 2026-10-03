# Spec — Monitoramento de hardware/energia (`psutil`)

> Atualização schema v2: os registros por etapa distinguem processo Python de
> sistema e deixam processo Ollama, GPU e energia como não coletados quando não há
> atribuição verificável. Ver `../../specs/evidencias/04-correcao-matematica-hardware.md`.

## Objetivo

Capturar, de forma estruturada e serializável, métricas de execução por etapa do
pipeline (tempo, CPU, RAM), para comparar custo entre modos e valores de `k`.

## Escopo

Apenas **`psutil`**. Sem `codecarbon` na primeira entrega.

## Dependência

- Adicionar `psutil` a `pipeline/requirements.txt`.

## Módulo

`pipeline/core/monitoring.py` — **sem I/O**.

```python
class Profiler:
    def __enter__(self) -> "Profiler": ...
    def __exit__(self, ...) -> None: ...

def snapshot(stage: str) -> dict:
    """
    Captura ponto único:
      {"stage": stage, "elapsed_ms": ..., "cpu_percent": ..., "ram_used_mb": ...}
    """
```

Métricas:

| Chave | Fonte | Descrição |
| --- | --- | --- |
| `elapsed_ms` | `time.perf_counter` | tempo de execução da etapa |
| `cpu_percent` | `psutil.cpu_percent(interval=None)` | % de CPU no momento |
| `ram_used_mb` | `psutil.virtual_memory().used` | memória usada (MB) |

## Integração

Em `run_eval.py`, envolver:

- `retrieval.search(query)` → `stage = "retrieval"`.
- `generation.generate(query, ...)` → `stage = "generation"`.

As métricas são gravadas junto ao resultado em `results.jsonl`.

## Critérios de aceite

1. `monitoring.py` não contém `print`/`input`.
2. O dicionário de métricas é serializável via `json.dumps`.
3. Cada etapa relevante grava seu `stage` no `results.jsonl`.
