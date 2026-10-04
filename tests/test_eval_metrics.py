import pytest

from pipeline.eval import metrics
from pipeline.eval.ranking import Qrels


def test_single_id_adapter_and_new_metrics():
    row = metrics.evaluate_retrieval(
        ["b", "a"],
        "a",
        2,
        corpus_ids=["a", "b"],
        query_id="query",
    )
    assert row["recall_at_k"] == row["hit_at_k"] == 1
    assert row["precision_at_k"] == 0.5
    assert row["mrr_at_k"] == 0.5
    assert row["source"] == "single_expected_id"
    assert row["excluded_ids"] == []
    assert metrics.recall_at_k(["a"], "a", 2) == 1
    assert metrics.precision_at_k(["a"], "a", 2) == 0.5
    assert metrics.ndcg_at_k(["a"], "a", 2) == 1
    with pytest.raises(ValueError):
        metrics.mrr([["a"]], [])


def test_qrels_adapter_uses_all_relevant_ids():
    qrels = Qrels("human", "v1", "fixture", {"q": dict.fromkeys("abcde", 1)})
    result = metrics.evaluate_retrieval(
        ["a", "f"],
        "a",
        2,
        corpus_ids=list("abcdef"),
        query_id="q",
        qrels=qrels,
    )
    assert result["precision_at_k"] == 0.5
    assert result["recall_at_k"] == 0.2
    assert result["judgment_coverage"] == 5 / 6
    assert (
        metrics.evaluate_retrieval(
            [],
            "a",
            2,
            corpus_ids=list("abcdef"),
            query_id="absent",
            qrels=qrels,
        )["status"]
        == "unjudged"
    )


def test_missing_expected_document_is_error():
    with pytest.raises(ValueError, match="fora do índice"):
        metrics.evaluate_retrieval([], "unknown", 2, corpus_ids=["a"], query_id="q")
