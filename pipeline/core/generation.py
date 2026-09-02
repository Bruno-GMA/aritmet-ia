"""
Geração de resposta via LLM (Ollama) — spec §8.

Fluxo:
  1. Para cada linha recuperada, trunca `respostas_contextualizadas`
     em MAX_CONTEXT_CHARS (corte simples, sem verificação de sentença).
  2. Monta prompt final = system_prompt + contexto truncado + pergunta.
  3. Chama CHAT_MODEL via Ollama.

Nenhum print/input aqui — toda I/O fica em cli.py (ver AGENTS.md).
"""

from __future__ import annotations

from typing import Any

import ollama

from pipeline.config import (
    CHAT_MODEL,
    MAX_CONTEXT_CHARS,
    OLLAMA_HOST,
)
from pipeline.system_prompt import get_system_prompt


def truncate_context(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Truncamento simples por caracteres (spec §8)."""
    return text[:max_chars]


def _build_context_block(results: list[dict[str, Any]]) -> str:
    """
    Monta bloco de contexto a partir das linhas recuperadas.
    Cada entrada inclui a questão original e a resposta contextualizada truncada.
    """
    blocks: list[str] = []
    for i, r in enumerate(results, start=1):
        resp = truncate_context(str(r.get("resposta_contextualizada", "")))
        block = (
            f"--- Contexto {i} ---\n"
            f"Questão: {r.get('questao', '')}\n"
            f"Nível: {r.get('nivel', '')}\n"
            f"Resposta contextualizada:\n{resp}\n"
        )
        blocks.append(block)
    return "\n".join(blocks)


def generate(query: str, context_results: list[dict[str, Any]]) -> str:
    """
    Gera resposta usando o LLM com o contexto recuperado.

    Args:
        query: pergunta do usuário (corrigida ou original).
        context_results: lista de dicts retornados por retrieval.search().

    Returns:
        Texto da resposta do LLM.
    """
    system_prompt = get_system_prompt()
    context_block = _build_context_block(context_results)

    user_message = (
        f"Contexto recuperado:\n\n{context_block}\n\n"
        f"Pergunta do usuário:\n{query}"
    )

    client = ollama.Client(host=OLLAMA_HOST)
    response = client.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    return response["message"]["content"]
