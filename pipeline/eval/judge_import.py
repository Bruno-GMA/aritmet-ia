"""
Reimportação e validação das notas do LLM de fronteira — spec llm-as-judge.md.

Extrai o JSON colado pelo usuário e normaliza numa lista de dicts.
Nenhum print/input aqui.
"""

from __future__ import annotations

import json
from typing import Any

REQUIRED_FIELDS = ["id", "mode", "score"]


def parse_judge_output(text: str) -> list[dict[str, Any]]:
    """
    Extrai e valida o JSON colado; retorna lista normalizada de notas.

    Aceita o texto com marcações de bloco de código (```json ... ```) comuns
    na saída de LLMs de fronteira.
    """
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    data = json.loads(text)

    if isinstance(data, dict):
        candidates = [data]
    elif isinstance(data, list):
        candidates = data
    else:
        raise ValueError("Saída do juiz deve ser um objeto ou lista JSON.")

    normalized: list[dict[str, Any]] = []
    for item in candidates:
        missing = [c for c in REQUIRED_FIELDS if c not in item]
        if missing:
            raise ValueError(f"Nota inválida, campos ausentes {missing}: {item}")
        normalized.append(
            {
                "id": str(item["id"]),
                "mode": str(item["mode"]),
                "score": int(item["score"]),
                "justification": str(item.get("justification", "")),
            }
        )
    return normalized