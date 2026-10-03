# Spec — LLM-as-Judge (modelo de fronteira, ida e volta por cópia)

## Objetivo

Avaliar a qualidade das respostas geradas pelo pipeline usando um **LLM de
fronteira externo** (ex.: ChatGPT, Claude Sonnet). O modelo de fronteira **não é
chamado pelo runtime** — funciona por **cópia**: o pipeline gera um prompt, o
usuário cola no modelo, e cola as notas de volta.

## Modelo juiz

- `JUDGE_MODEL` é um LLM de fronteira (externo ao Ollama).
- Definido como conceito no código (ex.: referência em `config.py` como string
  documental), mas **não** há cliente de API no pipeline.

## Módulos

### `judge_prompt.py` (sem I/O — apenas gera texto)

```python
def build_judge_rubric() -> str: ...
def build_judge_batch(results: list[dict], mode: str) -> str: ...
```

Cada bloco do batch contém:

```
[ID] <id>
[MODO] rag | no-rag
[QUESTÃO] <questão>
[REFERÊNCIA] <comparação / respostas_contextualizadas esperadas>
[RESPOSTA DO PIPELINE] <saída de generation.generate()>
```

### Rubrica (escala 1–5 + justificativa)

Critérios:

1. **Não resolve a questão do usuário diretamente** (não reusa números/nomes).
2. **Exemplo análogo válido** (mesma estrutura lógica, dados alterados).
3. **Operação correta identificada** (soma/subtração/multiplicação/divisão).
4. **Passo a passo claro e numerado**.
5. **Não-invenção** (não afirma informação fora do contexto/questão).

### Contrato JSON de retorno (instruído no prompt)

O juiz deve devolver, para cada item, um objeto:

```json
{"id": "eval_001", "mode": "rag", "score": 4, "justification": "..."}
```

### `judge_import.py` (sem I/O)

```python
def parse_judge_output(text: str) -> list[dict]:
    """Extrai e valida o JSON colado; retorna lista normalizada de notas."""
```

## Fluxo fim-a-fim

1. `run_eval --eval` gera respostas + `judge_prompt.txt`.
2. Usuário cola `judge_prompt.txt` no ChatGPT/Claude, recebe o JSON de notas.
3. `run_eval --judge` reaceita o JSON colado (`input()`/arquivo), grava
   `judge_scores.jsonl` e recalcula `metrics.json` agregando as notas.

## Critérios de aceite

1. `judge_prompt.py`/`judge_import.py` sem I/O.
2. A rubrica e o contrato JSON estão explícitos no prompt gerado.
3. As notas coladas consolidam em `metrics.json` (média por modo, distribuição).
4. As respostas do pipeline são consumidas **sem alterar** o caminho de geração.