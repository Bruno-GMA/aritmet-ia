"""Validation and normalization for the manual multidimensional judge output."""

from __future__ import annotations

import json
from typing import Any

from pipeline.eval.judge_prompt import CRITERIA, RUBRIC_VERSION


def _unwrap(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else ""
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
    return json.loads(cleaned.strip())


def parse_judge_output(text: str) -> list[dict[str, Any]]:
    """Parse v2 scores, validating all dimensions before any artifact is written."""
    data = _unwrap(text)
    items = data.get("scores") if isinstance(data, dict) and "scores" in data else data
    if not isinstance(items, list):
        raise ValueError("Saída do juiz deve ser uma lista JSON ou objeto com 'scores'.")
    normalized: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise ValueError("Cada nota precisa de um id textual.")
        criteria = item.get("criteria")
        if not isinstance(criteria, dict) or set(criteria) != set(CRITERIA):
            raise ValueError("Cada nota deve conter exatamente os cinco critérios da rubrica.")
        parsed: dict[str, dict[str, Any]] = {}
        for name in CRITERIA:
            value = criteria[name]
            if not isinstance(value, dict):
                raise ValueError(f"Critério inválido: {name}")
            score = value.get("score")
            if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
                raise ValueError(f"Nota de {name} deve ser inteiro entre 1 e 5.")
            justification = value.get("justification")
            evidence = value.get("evidence")
            if not isinstance(justification, str) or not justification.strip():
                raise ValueError(f"{name} requer justificativa textual.")
            if not isinstance(evidence, list) or not all(isinstance(x, str) for x in evidence):
                raise ValueError(f"{name} requer lista de evidências textuais.")
            status = value.get("status", "applicable")
            if name == "fundamentacao" and status not in {"applicable", "not_applicable"}:
                raise ValueError("Status de fundamentacao inválido.")
            if name != "fundamentacao" and "status" in value:
                raise ValueError("Somente fundamentacao pode declarar status.")
            parsed[name] = {
                "score": score,
                "justification": justification,
                "evidence": evidence,
                **({"status": status} if name == "fundamentacao" else {}),
            }
        normalized.append({"id": item["id"], "rubric_version": RUBRIC_VERSION, "criteria": parsed})
    return normalized
