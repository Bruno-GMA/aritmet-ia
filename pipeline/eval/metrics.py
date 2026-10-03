"""
Métricas de retrieval (hit@k, recall@k, MRR) — spec eval-harness.md.

Nenhum print/input aqui.
"""

from __future__ import annotations

import math
from typing import Any

from pipeline.eval.ranking import (
    Qrels,
    aggregate_rankings,
    corpus_fingerprint,
    evaluate_ranking,
    normalize_id,
    validate_ids,
    validate_k,
)


def evaluate_retrieval(
    retrieved_ids: list[str],
    expected_id: str,
    k: int,
    *,
    corpus_ids: list[str],
    query_id: str,
    qrels: Qrels | None = None,
) -> dict[str, Any]:
    """ID-only adapter or explicit qrels; expected document remains eligible."""
    query_id = normalize_id(query_id)
    grades: dict[str, int] | None
    if qrels is None:
        expected_id = normalize_id(expected_id)
        if expected_id not in corpus_ids:
            raise ValueError(f"csv_id esperado fora do índice: {expected_id}")
        grades = {doc: int(doc == expected_id) for doc in corpus_ids}
        source, version = "single_expected_id", "1"
        corpus_id = corpus_fingerprint(corpus_ids)
    else:
        qrels.validate_corpus(corpus_ids)
        grades = qrels.judgments.get(query_id)
        source, version, corpus_id = qrels.source, qrels.version, qrels.corpus_id
    result = evaluate_ranking(
        retrieved_ids,
        k,
        corpus_ids=corpus_ids,
        grades=grades,
        source=source,
        relevance_version=version,
        corpus_id=corpus_id,
    )
    result["query_id"] = query_id
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return aggregate_rankings(rows)


def rank_of(retrieved_ids: list[str], expected_id: str) -> int | None:
    """Retorna a posição 1-based do expected_id; None se ausente."""
    retrieved_ids = validate_ids(retrieved_ids, "ranking")
    expected_id = normalize_id(expected_id)
    for idx, rid in enumerate(retrieved_ids, start=1):
        if str(rid) == str(expected_id):
            return idx
    return None


def hit_at_k(retrieved_ids: list[str], expected_id: str, k: int) -> float:
    """1.0 se expected_id está entre os k primeiros; senão 0.0."""
    validate_k(k)
    validate_ids(retrieved_ids, "ranking")
    r = rank_of(retrieved_ids[:k], expected_id)
    return 1.0 if r is not None else 0.0


def recall_at_k(retrieved_ids: list[str], expected_id: str, k: int) -> float:
    """Recall binário = hit_at_k (ground truth é um único item esperado)."""
    return hit_at_k(retrieved_ids, expected_id, k)


def precision_at_k(retrieved_ids: list[str], expected_id: str, k: int) -> float:
    """Precision for the single-expected-ID adapter (short lists keep divisor k)."""
    return hit_at_k(retrieved_ids, expected_id, k) / k


def ndcg_at_k(retrieved_ids: list[str], expected_id: str, k: int) -> float:
    """Binary nDCG for one expected document; use evaluate_retrieval for graded qrels."""
    validate_k(k)
    validate_ids(retrieved_ids, "ranking")
    rank = rank_of(retrieved_ids[:k], expected_id)
    return 1 / math.log2(rank + 1) if rank else 0.0


def reciprocal_rank(retrieved_ids: list[str], expected_id: str) -> float:
    """1/rank; 0.0 se não recuperado."""
    r = rank_of(retrieved_ids, expected_id)
    return 1.0 / r if r is not None else 0.0


def mrr(list_of_retrieved_ids: list[list[str]], expected_ids: list[str]) -> float:
    """Mean Reciprocal Rank sobre um conjunto de consultas."""
    if len(list_of_retrieved_ids) != len(expected_ids):
        raise ValueError("Quantidade de rankings difere dos IDs esperados")
    if not expected_ids:
        return 0.0
    total = sum(
        reciprocal_rank(rids, exp) for rids, exp in zip(list_of_retrieved_ids, expected_ids)
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
