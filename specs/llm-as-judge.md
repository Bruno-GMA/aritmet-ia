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

### Rubrica v2 (escala 1–5 + justificativa + evidência)

Critérios vigentes:

1. **Qualidade pedagógica** — ensino por analogia, operação explicada e passos.
2. **Correção semântica** — interpretação e matemática da resposta.
3. **Coerência lógica** — conclusões derivam dos passos e não se contradizem.
4. **Completude** — cobre os elementos requeridos pelo tutor.
5. **Fundamentação** — apoio no contexto/pergunta ou derivação matemática válida.

Cada dimensão recebe `score`, `justification` e `evidence`. Em `no-rag`,
`fundamentacao.status` é `not_applicable`: a ausência de contexto não é erro;
avaliam-se a derivação e a coerência com a pergunta. As âncoras 1/3/5 e os
erros exemplificados estão no próprio prompt gerado (`judge-rubric-v2`).

### Contrato JSON de retorno (instruído no prompt)

O juiz deve devolver, para cada item, um objeto:

```json
{"id": "j0001-opaque", "criteria": {"qualidade_pedagogica": {"score": 4, "justification": "...", "evidence": ["..."]}, "correcao_semantica": {"score": 4, "justification": "...", "evidence": ["..."]}, "coerencia_logica": {"score": 4, "justification": "...", "evidence": ["..."]}, "completude": {"score": 4, "justification": "...", "evidence": ["..."]}, "fundamentacao": {"score": 4, "justification": "...", "evidence": ["..."], "status": "applicable"}}}
```

### `judge_import.py` (sem I/O)

```python
def parse_judge_output(text: str) -> list[dict]:
    """Extrai e valida o JSON colado; retorna lista normalizada de notas."""
```

## Fluxo fim-a-fim

1. `run_eval --eval` gera respostas + `judge_prompt.txt`.
2. Usuário cola `judge_prompt.txt` no ChatGPT/Claude, recebe o JSON de notas.
3. `run_eval --judge --judge-model "..."` reaceita o JSON colado, valida IDs
   opacos, duplicatas, cinco critérios e faixas, grava `judge_raw.json` e
   `judge_scores.jsonl` e recalcula `metrics.json` por dimensão/modo/operação.
   Itens faltantes ficam explícitos, sem virar nota zero.

## Critérios de aceite

1. `judge_prompt.py`/`judge_import.py` sem I/O.
2. A rubrica e o contrato JSON estão explícitos no prompt gerado.
3. As notas coladas consolidam em `metrics.json` (média por modo, distribuição).
4. As respostas do pipeline são consumidas **sem alterar** o caminho de geração.
