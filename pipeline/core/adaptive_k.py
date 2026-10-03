"""
Seleção adaptativa de TOP_K — spec adaptive-k.md.

Para cada k candidato, roda retrieval.search(query, k) e registra:
  - hit:  se o csv_id esperado (ground truth) foi recuperado;
  - rank: posição 1-based do csv_id esperado (None se ausente);
  - distance: distância do csv_id esperado (None se ausente);
  - operacao_gold: operação esperada (ground truth);
  - operacoes_recuperadas: operações das k linhas recuperadas, em ordem;
  - operacao_consistente: a operação gold é a MAIORIA entre as k recuperadas.

Métrica de consistência alinhada a Raganato et al. (2021, CDA/counterfactual):
mede se o encoder preserva o CONCEITO (operação) e não só o léxico, mesmo
quando o csv_id exato não está no topo. Empate de maioria => inconsistente (0).

Nenhum print/input aqui — toda I/O fica em cli.py / run_eval.py (ver AGENTS.md).
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from pipeline.core import retrieval
from pipeline.eval import metrics
from pipeline.eval.ranking import Qrels, validate_k


def majority_operation(operacoes: list[str]) -> str:
    """Operação mais frequente; empate de maioria resolve conservadoramente ('')."""
    if not operacoes:
        return ""
    counts = Counter(operacoes)
    top_val, top_n = counts.most_common(1)[0]
    tied = [op for op, n in counts.items() if n == top_n]
    if len(tied) > 1:
        return ""
    return top_val


def is_consistent(operacoes_recuperadas: list[str], operacao_gold: str) -> bool:
    """True se a operação gold é a maioria (estrita) entre as recuperadas."""
    majority = majority_operation(operacoes_recuperadas)
    return bool(majority) and majority == operacao_gold


def evaluate_k(
    query: str,
    expected_csv_id: str,
    k_values: list[int],
    operacao_gold: str = "",
    *,
    corpus_ids: list[str] | None = None,
    query_id: str | None = None,
    qrels: Qrels | None = None,
) -> dict[int, dict[str, Any]]:
    """
    Avalia o desempenho de retrieval para vários valores de k.

    Args:
        query: pergunta do usuário.
        expected_csv_id: csv_id (ground truth) esperado entre os recuperados.
        k_values: lista de k a avaliar.
        operacao_gold: operação esperada (ground truth) para a métrica de
                       consistência (opcional).

    Returns:
        {k: {"hit", "rank", "distance", "operacao_gold",
             "operacoes_recuperadas", "operacao_consistente"}, ...}
    """
    for k in k_values:
        validate_k(k)
    if len(set(k_values)) != len(k_values):
        raise ValueError("Valores de k duplicados")
    if qrels is not None and query_id is None:
        raise ValueError("query_id obrigatório com qrels")
    if corpus_ids is None:
        corpus_ids = retrieval.corpus_ids()
    results: dict[int, dict[str, Any]] = {}
    for k in k_values:
        rows = retrieval.search(query, top_k=k)
        rank = None
        distance = None
        hit = False
        operacoes = [str(row.get("operacao", "")) for row in rows]

        for idx, row in enumerate(rows, start=1):
            if str(row.get("csv_id", "")) == str(expected_csv_id):
                rank = idx
                distance = row.get("distance")
                hit = True
                break

        results[k] = {
            "metrics": metrics.evaluate_retrieval(
                [row["csv_id"] for row in rows],
                expected_csv_id,
                k,
                corpus_ids=corpus_ids,
                query_id=query_id or expected_csv_id,
                qrels=qrels,
            ),
            "hit": hit,
            "rank": rank,
            "distance": distance,
            "operacao_gold": operacao_gold,
            "operacoes_recuperadas": operacoes,
            "operacao_consistente": is_consistent(operacoes, operacao_gold),
        }
    return results


def hit_at_k(eval_results: dict[int, dict[str, Any]], k: int) -> float:
    """Fração de amostras com hit no valor k (esperado depois de agregar sobre o eval set)."""
    vals = eval_results.get(k)
    if vals is None:
        return 0.0
    return 1.0 if vals["hit"] else 0.0


def mrr_for_row(eval_results: dict[int, dict[str, Any]], k: int) -> float:
    """Reciprocal rank para um valor k específico (0 se não houve hit)."""
    vals = eval_results.get(k)
    if vals is None or vals["rank"] is None:
        return 0.0
    return 1.0 / vals["rank"]


def consistency_at_k(eval_results: dict[int, dict[str, Any]], k: int) -> float:
    """1.0 se a operação gold é a maioria entre os k recuperados; senão 0.0."""
    vals = eval_results.get(k)
    if vals is None:
        return 0.0
    return 1.0 if vals["operacao_consistente"] else 0.0
