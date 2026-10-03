# Spec — Adaptive-K (seleção do melhor `TOP_K`)

## Objetivo

Variar o número de vizinhos recuperados (`k`) e medir `hit@k`, **MRR** e
**consistência de operação** sobre o conjunto de avaliação, globalmente e
estratificado por `operação`, para selecionar o valor de `TOP_K` com melhor
custo/benefício.

## Config

```python
# config.py
TOP_K = 3                    # mantido como default de runtime
TOP_K_RANGE = [1, 2, 3, 4, 5, 6, 7]   # candidatos avaliados
```

## Módulo

`pipeline/core/adaptive_k.py` — **sem I/O** (regra AGENTS.md).

```python
def evaluate_k(query, expected_csv_id, k_values, operacao_gold="") -> dict:
    """
    Para cada k, roda retrieval.search(query, k) e registra:
      - "hit": bool
      - "rank": int | None
      - "distance": float | None
      - "operacao_gold": str
      - "operacoes_recuperadas": list[str]  (operações das k linhas, em ordem)
      - "operacao_consistente": bool

    Retorna {k: {...}, ...}
    """
```

Reutiliza `retrieval.search(query, top_k=k)` — **não** duplica a geração de
embedding nem a lógica de busca.

## Métricas

- **hit@k** = fração das amostras onde `expected_csv_id` aparece entre os `k`
  primeiros resultados.
- **MRR** (Mean Reciprocal Rank) = média de `1/rank`; `0` quando o item correto
  não é recuperado.
- **Consistência de operação (`cons@k`)** = fração das amostras onde a
  `operação` gold é a **maioria estrita** entre as `k` operações recuperadas.
  Empate de maioria => **inconsistente (0)** — conservador, para não inflar a
  métrica.

## Embasamento científico

- **DPR (Karpukhin et al., 2020)** e **BEIR (Thakur et al., 2021)**: `hit@k`,
  `recall@k` e MRR são as métricas canônicas de retrieval; reportar por
  subgrupo (aqui, `operação` tratada como "dataset") é prática padrão.
- **CDA / contrafactual (Raganato et al., 2021)** e **Shwartz & Dagan
  (diagnóstico de hard cases)**: a `operação` **não** deve virar feature de
  busca (daria a resposta a priori); serve para **diagnosticar** se o encoder
  captura o *conceito* (operação) ou apenas o *léxico*. A métrica de
  consistência mede exatamente essa propriedade.
- **RAGAS (Es et al., 2023)**: separa qualidade da recuperação (context
  relevance) da geração; `operação` é atributo de **avaliação de retrieval**.

## Relatório

O orquestrador (`run_eval.py`) agrega sobre o eval set e imprime, para o
global e para cada `operação`:

```
k | hit@k | MRR | cons@k
1 | 0.83  | 0.83| 0.90
2 | 0.91  | 0.87| 0.92
...
```

Indicando o menor `k` com ganho marginal desprezível (ponto de cotovelo).

## Critérios de aceite

1. `evaluate_k` retorna o dict mapeando `k -> métricas` sem efeitos colaterais.
2. Usa `retrieval.search(query, k)` diretamente (sem reimplementar busca).
3. Consistência usa maioria **estrita** (empate => 0).
4. Sem `print`/`input` no módulo.

## Atualização schema v2

Além dos diagnósticos de hit/consistência legados, cada linha de `evaluate_k`
contém `metrics` no contrato de `retrieval-metrics.md`. O harness grava as cinco
métricas com denominadores, fontes e linhas por consulta em
`<output-dir>/adaptive_k_metrics.json`, globalmente e por operação. Qrels opcionais
usam o ID da consulta; o adaptador de csv_id único continua disponível.
