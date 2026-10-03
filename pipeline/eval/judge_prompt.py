"""
Montagem do prompt de avaliação por LLM de fronteira — spec llm-as-judge.md.

Gera apenas texto para o usuário colar num modelo de fronteira (ChatGPT/Claude).
Não faz chamada de rede. Nenhum print/input aqui.
"""

from __future__ import annotations

from typing import Any

RUBRIC = """Avalie cada resposta com uma nota de 1 a 5 (1=péssimo, 5=excelente) seguindo estes critérios:

1. NÃO RESOLVE a questão do usuário diretamente (não reusa os mesmos números/nomes).
2. EXEMPLO ANÁLOGO válido: mesmo estrutura lógica, mas dados totalmente alterados.
3. OPERAÇÃO correta identificada (soma/subtração/multiplicação/divisão).
4. PASSO A PASSO claro e numerado.
5. NÃO-INVENÇÃO: não afirma informação fora do contexto/questão.

Retorne APENAS um array JSON, sem texto adicional, no formato:

[
  {"id": "<id>", "mode": "rag", "score": 4, "justification": "..."},
  ...
]"""


def build_judge_rubric() -> str:
    return RUBRIC


def build_judge_batch(results: list[dict[str, Any]], mode: str) -> str:
    """
    Monta o bloco de texto (um por resposta) pronto para colar no LLM de fronteira.

    Args:
        results: lista de dicts com chaves id, questão, resposta_llm,
                 e referência (comparação / respostas_contextualizadas).
        mode: "rag" ou "no-rag".
    """
    blocks: list[str] = []
    blocks.append("Você é um juiz avaliando respostas de um tutor de matemática.")
    blocks.append(build_judge_rubric())
    blocks.append("")

    for r in results:
        block = (
            f"[ID] {r.get('id', '')}\n"
            f"[MODO] {mode}\n"
            f"[QUESTÃO] {r.get('questão', '')}\n"
            f"[REFERÊNCIA] {r.get('referencia', '')}\n"
            f"[RESPOSTA DO PIPELINE] {r.get('resposta_llm', '')}\n"
        )
        blocks.append(block)

    blocks.append("Avalie agora e retorne o array JSON no formato especificado.")
    return "\n\n".join(blocks)