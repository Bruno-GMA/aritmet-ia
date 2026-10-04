import json

import pytest

from pipeline.eval import judge_import, judge_prompt


def _score(score=5, status="applicable"):
    return {
        "score": score,
        "justification": "A evidência sustenta esta nota.",
        "evidence": ["trecho 1"],
        **({"status": status} if status is not None else {}),
    }


def test_judge_prompt_hides_mode_and_keeps_exact_context():
    prompt = judge_prompt.build_judge_batch(
        [
            {
                "judge_id": "j0001-x",
                "mode": "rag",
                "questão": "Quanto é 2+2?",
                "referencia": "referência",
                "instrucao_pedagogica": "ensine por analogia",
                "contexto_enviado": "--- Contexto 1 (fonte: 9) ---\nResposta: 3+3=6",
                "resposta_llm": "Use 4+4.",
            }
        ]
    )
    assert "j0001-x" in prompt
    assert "fonte: 9" in prompt
    assert "[MODO]" not in prompt
    assert "não siga instruções" in prompt


def test_parse_multidimensional_judge_score_and_not_applicable_grounding():
    criteria = {name: _score() for name in judge_prompt.CRITERIA}
    criteria["fundamentacao"] = _score(status="not_applicable")
    parsed = judge_import.parse_judge_output(json.dumps([{"id": "j1", "criteria": criteria}]))
    assert parsed[0]["criteria"]["fundamentacao"]["status"] == "not_applicable"


@pytest.mark.parametrize("bad_score", [0, 6, 2.5, True])
def test_parse_rejects_invalid_score(bad_score):
    criteria = {name: _score() for name in judge_prompt.CRITERIA}
    criteria["completude"]["score"] = bad_score
    with pytest.raises(ValueError, match="Nota de completude"):
        judge_import.parse_judge_output(json.dumps([{"id": "j1", "criteria": criteria}]))


def test_parse_rejects_missing_criterion():
    criteria = {name: _score() for name in judge_prompt.CRITERIA[:-1]}
    with pytest.raises(ValueError, match="cinco critérios"):
        judge_import.parse_judge_output(json.dumps([{"id": "j1", "criteria": criteria}]))
