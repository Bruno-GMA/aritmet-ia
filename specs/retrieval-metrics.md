# Métricas de recuperação — schema v2

## Definições e compatibilidade

Esta versão substitui o contrato anterior. Não compare diretamente recall/nDCG de
artefatos sem `schema_version: 2` com os novos resultados: o recall antigo do SOMAIA
usava o denominador de precision e seu IDCG considerava apenas o ranking recuperado.

- `precision_at_k`: relevantes únicos recuperados / k, mesmo em listas curtas.
- `recall_at_k`: relevantes recuperados / total de relevantes elegíveis conhecidos.
- `mrr_at_k`: 1 / posição do primeiro relevante no top-k; a agregação é a média.
- `ndcg_at_k`: DCG / IDCG, ganho `2^grau - 1`, desconto `log2(rank + 1)`;
  IDCG usa os maiores graus do corpus elegível, inclusive documentos não recuperados.
- `hit_at_k`: 1 quando algum relevante aparece no top-k, senão 0.

Graus são inteiros não negativos; qualquer grau positivo é relevante para as métricas
binárias. O cálculo exige `k > 0`, IDs únicos, válidos e pertencentes ao corpus.
O conjunto recuperado nunca é usado para inferir o denominador do recall.

## Contrato de qrels

O argumento `--qrels` recebe JSON UTF-8; IDs de consulta e documento são strings:

```json
{
  "schema_version": 1,
  "source": "anotacao_humana",
  "version": "v1",
  "corpus_id": "corpus-avaliacao-v1",
  "qrels": {
    "consulta_1": {"documento_a": 2, "documento_b": 1, "documento_c": 0},
    "consulta_sem_relevantes": {}
  }
}
```

`source`, `version` e `corpus_id` são obrigatórios. `corpus_id` é a versão declarada
pelo autor; o relatório também inclui SHA-256 dos IDs elegíveis, que não substitui
um hash dos textos/modelos. IDs anotados fora do corpus são rejeitados. Chaves JSON
duplicadas, graus negativos/fracionários/booleanos e IDs inválidos são rejeitados.

Consulta ausente em `qrels` é `unjudged`: todas as métricas ficam `null` e não entram
na média. Consulta explicitamente presente sem relevantes conhecidos é `no_relevant`:
métricas zero, com status e cobertura registrados. Um documento não julgado recebe
`null` em `relevance_grades` e ganho zero na computação; isso **não prova irrelevância**.
Com anotações incompletas, recall e IDCG se referem apenas à relevância conhecida.
Reporte `judgment_coverage`, `n_judged` e `n_unjudged_retrieved` junto às métricas.

## Artefatos e interpretação

Cada consulta registra `query_id`, `k`, fonte/versão da relevância, corpus declarado,
IDs recuperados, graus, total elegível/relevante, exclusões, cobertura e métricas.
Agregados são médias macro com `valid_counts`, `status_counts`, fontes, versões e k.
Sem observações válidas, a média é `null`, não zero. Os artefatos históricos são
preservados; novos experimentos usam novos caminhos.

Separe consultas e corpus, documentando perguntas idênticas ou parafraseadas e o
que se pretende medir. Recuperar o ID esperado de uma pergunta já indexada mede
localização, não necessariamente generalização nem utilidade pedagógica. Não derive
qrels das respostas do próprio sistema. Cobertura total de um adaptador automático
não equivale a cobertura de anotação humana.

## Verificação

O caso canônico possui cinco documentos relevantes elegíveis e um irrelevante. Um
ranking top-2 com apenas um relevante deve produzir precision=0,5 e recall=0,2.
Testes adicionais cobrem relevância graduada, relevante ideal não recuperado, ordem,
lista curta/vazia, nenhuma relevância, qrels incompletos, exclusão da própria consulta,
IDs inválidos/duplicados e agregação sem observações.

## Integração aritmet.IA

O corpus elegível é a junção das linhas efetivamente indexadas em `questions_vec`
com `questions_meta`; não é inferido do CSV atual nem do top-k. `csv_id` duplicado
no índice é erro. A recuperação não exclui o ID esperado: ele é a referência de
localização, não o ID independente da consulta de avaliação.

Sem `--qrels`, o adaptador `single_expected_id` dá grau 1 ao `csv_id_esperado` e zero
aos demais documentos indexados. Por definição, recall=hit nesse modo; isso não
significa que o corpus tenha apenas um documento pedagogicamente útil.
Com `--qrels`, as chaves de consulta são `id` do eval_set, e as de documento são
`csv_id` do índice. O conjunto de avaliação mantém seus campos atuais para
compatibilidade com a referência usada pela geração/juiz.

```bash
# Execute na raiz aritmet.ia.
python -m pipeline.eval.run_eval --eval --output-dir pipeline/eval/output/experimento-v2
python -m pipeline.eval.run_eval --adaptive-k --qrels qrels.json --output-dir pipeline/eval/output/k-v2
python -m pipeline.eval.run_eval --judge --output-dir pipeline/eval/output/experimento-v2
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

`--eval` e `--adaptive-k` criam diretórios novos; sem `--output-dir`, usam
`pipeline/eval/output/<timestamp>/`. `--judge` recebe o diretório da avaliação a
consolidar. `metrics.json` contém as cinco métricas em `metrics` e linhas em
`per_query`; `results.jsonl` inclui `retrieval_quality` separado da instrumentação
`retrieval_metrics`. Em no-rag, `retrieval_quality` é `null` (não aplicável).
`adaptive_k_metrics.json` exporta agregados e linhas por k e por operação, sem gerar
respostas. Conjuntos vazios produzem zero observações e médias `null`.

Os testes executam busca sqlite-vec e exportação reais, com embedding/geração
controlados. O núcleo mantém um embedding por questão, sem chunking, e não possui
interação de terminal. A camada de métricas é local ao projeto e não depende do SOMAIA.
