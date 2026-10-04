"""Pure deterministic reranker; it never creates candidates or reads evaluation labels."""

from __future__ import annotations

import re
from typing import Any

METHOD_VERSION = "distance-operation-v1"
SIMILARITY_WEIGHT = 0.8
OPERATION_WEIGHT = 0.2
_TERMS = {
    "soma": ("soma", "adição", "adicao", "mais", "total", "juntar"),
    "subtração": ("subtração", "subtracao", "menos", "diferença", "restam"),
    "multiplicação": ("multiplicação", "multiplicacao", "vezes", "grupos", "cada"),
    "divisão": ("divisão", "divisao", "dividir", "repartir"),
}


def infer_operation(query: str) -> str | None:
    text = re.sub(r"\s+", " ", query.lower())
    operation, count = max(
        ((name, sum(term in text for term in terms)) for name, terms in _TERMS.items()),
        key=lambda item: item[1],
    )
    return operation if count else None


def rerank_with_trace(
    query: str, candidates: list[dict[str, Any]], final_k: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return selected candidates plus an audit record for every input candidate."""
    if final_k <= 0:
        raise ValueError("final_k deve ser maior que zero")
    unique: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()
    for rank, candidate in enumerate(candidates, 1):
        candidate_id = str(candidate.get("csv_id", ""))
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        unique.append((rank, candidate))
    if not unique:
        return [], []
    distances = [float(candidate.get("distance", 0.0)) for _, candidate in unique]
    low, high = min(distances), max(distances)
    inferred = infer_operation(query)
    scored: list[tuple[float, int, str, dict[str, Any]]] = []
    for original_rank, candidate in unique:
        distance = float(candidate.get("distance", 0.0))
        similarity = 1.0 if low == high else (high - distance) / (high - low)
        match = float(inferred is not None and inferred in str(candidate.get("operacao", "")).lower())
        score = SIMILARITY_WEIGHT * similarity + OPERATION_WEIGHT * match
        enriched = dict(candidate)
        enriched["reranking"] = {
            "method": METHOD_VERSION,
            "original_rank": original_rank,
            "original_distance": distance,
            "normalized_similarity": similarity,
            "operation_match": match,
            "score": score,
        }
        scored.append((score, original_rank, str(candidate.get("csv_id", "")), enriched))
    scored.sort(key=lambda row: (-row[0], row[1], row[2]))
    trace: list[dict[str, Any]] = []
    selected = [row[3] for row in scored[:final_k]]
    for rank, candidate in enumerate(selected, 1):
        candidate["reranking"]["reranked_rank"] = rank
    selected_ids = {id(candidate) for candidate in selected}
    for reranked_rank, (_, original_rank, _, candidate) in enumerate(scored, 1):
        audit = dict(candidate["reranking"])
        audit.update({"csv_id": candidate.get("csv_id"), "selected": id(candidate) in selected_ids})
        audit["reranked_rank"] = reranked_rank
        trace.append(audit)
    return selected, trace


def rerank(query: str, candidates: list[dict[str, Any]], final_k: int) -> list[dict[str, Any]]:
    """Compatibility convenience returning only the final_k selected candidates."""
    return rerank_with_trace(query, candidates, final_k)[0]
