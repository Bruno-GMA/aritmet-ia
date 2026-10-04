# Spec — Flag `--no-rag`

## Objetivo

Permitir que o pipeline gere resposta **sem a etapa de retrieval**, produzindo um
*baseline* para comparar o efeito do RAG (resposta com contexto recuperado vs.
resposta sem contexto).

## Escopo

- Flag **exclusivamente de CLI**. Não adicionar `ENABLE_RAG` ao `config.py`.
- Afeta apenas o fluxo de geração: pula `retrieval.search()` e gera com lista
  de contexto vazia.

## Arquivos afetados

| Arquivo | Mudança |
| --- | --- |
| `pipeline/cli.py` | migrar para `argparse`; adicionar `--no-rag`. Quando ativo, pular `retrieval.search(query)` e chamar `generation.generate(query, [])`. |
| `pipeline/core/generation.py` | quando `context_results == []`, montar `user_message` **sem** o bloco `Contexto recuperado`. |

## Contrato

```python
# generation.py
def generate(query: str, context_results: list[dict]) -> str:
    if context_results:
        context_block = _build_context_block(context_results)
        user_message = f"Contexto recuperado:\n\n{context_block}\n\nPergunta do usuário:\n{query}"
    else:
        user_message = f"Pergunta do usuário:\n{query}"
    ...
```

Regra de I/O mantida: nenhum `print`/`input` entra em `core/generation.py`.

## Critérios de aceite

1. `generate("pergunta", [])` retorna resposta válida, sem mencionar contexto.
2. O prompt final no modo `--no-rag` **não** contém `Contexto recuperado:` vazio.
3. `cli.py` continua sendo o único módulo com I/O.
4. Rodar `python pipeline/cli.py --no-rag` exibe pergunta → geração direta, sem a linha "Buscando contexto relevante...".

## Notas

- O system prompt (`system_prompt.py`) **não** muda: ele já instrui o modelo a
  se declarar incapaz quando falta contexto, o que também serve ao baseline.