"""
Métricas de retrieval (hit@k, recall@k, MRR) — spec eval-harness.md.

Nenhum print/input aqui.
"""

from __future__ import annotations

from typing import Any


def rank_of(retrieved_ids: list[str], expected_id: str) -> int | None:
    """Retorna a posição 1-based do expected_id; None se ausente."""
    for idx, rid in enumerate(retrieved_ids, start=1):
        if str(rid) == str(expected_id):
            return idx
    return None


def hit_at_k(retrieved_ids: list[str], expected_id: str, k: int) -> float:
    """1.0 se expected_id está entre os k primeiros; senão 0.0."""
    r = rank_of(retrieved_ids[:k], expected_id)
    return 1.0 if r is not None else 0.0


def recall_at_k(retrieved_ids: list[str], expected_id: str, k: int) -> float:
    """Recall binário = hit_at_k (ground truth é um único item esperado)."""
    return hit_at_k(retrieved_ids, expected_id, k)


def reciprocal_rank(retrieved_ids: list[str], expected_id: str) -> float:
    """1/rank; 0.0 se não recuperado."""
    r = rank_of(retrieved_ids, expected_id)
    return 1.0 / r if r is not None else 0.0


def mrr(list_of_retrieved_ids: list[list[str]], expected_ids: list[str]) -> float:
    """Mean Reciprocal Rank sobre um conjunto de consultas."""
    if not expected_ids:
        return 0.0
    total = sum(
        reciprocal_rank(rids, exp)
        for rids, exp in zip(list_of_retrieved_ids, expected_ids)
    )
    return total / len(expected_ids)


def consistency_at_k(operacoes_recuperadas: list[str], operacao_gold: str) -> float:
    """1.0 se a operação gold é a maioria (estrita) entre as recuperadas; senão 0.0.

    Alinhada a Raganato et al. (2021, CDA): mede preservação do conceito,
    não do léxico. Empate de maioria => 0.0.
    """
    from collections import Counter

    if not operacoes_recuperadas:
        return 0.0
    counts = Counter(operacoes_recuperadas)
    top_val, top_n = counts.most_common(1)[0]
    tied = [op for op, n in counts.items() if n == top_n]
    if len(tied) > 1:
        return 0.0
    return 1.0 if top_val == operacao_gold else 0.0


def group_by(operacoes: list[str]) -> dict[str, list[int]]:
    """Agrupa índices de amostras por valor de operação (para estratificação)."""
    groups: dict[str, list[int]] = {}
    for i, op in enumerate(operacoes):
        groups.setdefault(op, []).append(i)
    return groups