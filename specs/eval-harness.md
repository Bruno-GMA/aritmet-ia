# Spec — Harness de Avaliação

## Objetivo

Avaliar o pipeline de forma reproduzível, com **ground truth = `csv_id`** da
questão (alinhado à arquitetura pergunta-contra-pergunta), nos modos `rag` e
`no-rag`.

## Estrutura

```
pipeline/eval/
├── __init__.py
├── dataset.py        # carga + validação do eval_set.jsonl
├── metrics.py        # adaptadores e métricas de recuperação
├── ranking.py        # qrels, métricas e agregação puras
├── judge_prompt.py   # monta prompt p/ LLM de fronteira
├── judge_import.py   # reimporta notas coladas (JSON)
└── run_eval.py       # orquestrador (ÚNICO com I/O)
pipeline/data/eval_set.jsonl
```

## Formato do dataset — `eval_set.jsonl`

Uma amostra por linha:

```json
{"id": "eval_001", "questão": "Caio tem R$ 1604,96 para repartir...", "csv_id_esperado": "1", "nível": "difícil", "operação": "divisão"}
```

- Amostras balanceadas por `nível` (fácil / intermediário / difícil) **e** por
  `operação` (soma / subtração / multiplicação / divisão).
- `csv_id_esperado` é o `ID` da linha do CSV considerada a resposta correta
  (ground truth de retrieval).
- `operação` é o rótulo de ground truth usado na métrica de consistência de
  operação (ver `adaptive-k.md`).

## Módulos

### `dataset.py` (sem I/O)

```python
def load_eval_set(path: str) -> list[dict]:
    """Lê eval_set.jsonl, valida campos e retorna lista de amostras."""
```

### `metrics.py` (sem I/O)

```python
def hit_at_k(ranks: list[int | None], k: int) -> float: ...
def recall_at_k(retrieved_ids: list[str], expected_id: str, k: int) -> float: ...
def mrr(ranks: list[int | None]) -> float: ...
```

## Orquestrador — `run_eval.py`

Subcomandos (`argparse`):

- `--eval` — roda o pipeline por amostra e modo, gera artefatos.
- `--adaptive-k` — delega a `core/adaptive_k.py`.
- `--judge` — reaceita notas do LLM de fronteira (ver `llm-as-judge.md`).

### Artefatos de saída

| Arquivo | Conteúdo |
| --- | --- |
| `pipeline/eval/output/<execução>/results.jsonl` | linha por resposta: `id`, `mode`, `questão`, `csv_id_esperado`, `contexto_recuperado`, `resposta_llm`, `métricas psutil`, `ground_truth`. |
| `pipeline/eval/output/<execução>/judge_prompt.txt` | bloco pronto para colar no LLM de fronteira. |
| `pipeline/eval/output/<execução>/metrics.json` | consolidação: `hit@k`, `recall@k`, `mrr`, notas do juiz por modo. |

## Critérios de aceite

1. `dataset.py` e `metrics.py` sem `print`/`input`.
2. `run_eval.py` é o único módulo de avaliação com I/O.
3. O modo `no-rag` gera resposta sem chamar `retrieval.search`.
4. Resultados preservam o `csv_id` como referência de ground truth.

## Atualização: recuperação schema v2

O contrato vigente de métricas, qrels, diretórios e casos limite está em
[`retrieval-metrics.md`](retrieval-metrics.md) e substitui os exemplos legados de
assinatura acima. Há hit, precision, recall, MRR e nDCG por consulta, globalmente e
por k/operação. `--qrels` é opcional; o adaptador de ID único continua sendo padrão.
`--output-dir` seleciona um diretório novo; `--judge` exige o diretório existente da
execução. Métricas de recuperação em no-rag são não aplicáveis (`null`).
