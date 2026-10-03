"""
Carga e validação do conjunto de avaliação (eval_set.jsonl) — spec eval-harness.md.

Nenhum print/input aqui.
"""

from __future__ import annotations

import json
from typing import Any

from pipeline.eval.ranking import normalize_id, validate_ids

REQUIRED_FIELDS = ["id", "questão", "csv_id_esperado", "nível", "operação"]


def load_eval_set(path: str) -> list[dict[str, Any]]:
    """Lê eval_set.jsonl, valida campos e retorna lista de amostras."""
    samples: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Linha {line_no}: objeto esperado")
            missing = [c for c in REQUIRED_FIELDS if c not in obj]
            if missing:
                raise ValueError(f"Linha {line_no}: campos ausentes {missing}")
            obj["id"] = normalize_id(obj["id"])
            obj["csv_id_esperado"] = normalize_id(obj["csv_id_esperado"])
            samples.append(obj)
    validate_ids([sample["id"] for sample in samples], "eval_set")
    return samples
