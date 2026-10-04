from pipeline.core.reranking import rerank_with_trace


def test_reranking_records_all_input_candidates_and_selects_final_k():
    candidates = [
        {"csv_id": "a", "distance": 0.1, "operacao": "subtração"},
        {"csv_id": "b", "distance": 0.2, "operacao": "soma"},
    ]
    selected, trace = rerank_with_trace("qual é a soma total?", candidates, final_k=1)
    assert [row["csv_id"] for row in selected] == ["a"]
    assert {row["csv_id"] for row in trace} == {"a", "b"}
    assert trace[0]["original_rank"] == 1


def test_reranking_rejects_invalid_final_k():
    try:
        rerank_with_trace("soma", [], final_k=0)
    except ValueError as exc:
        assert "final_k" in str(exc)
    else:
        raise AssertionError("final_k inválido deveria falhar")
