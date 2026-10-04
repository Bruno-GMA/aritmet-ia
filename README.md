# aritmet.IA

Assistente matemático local, baseado em RAG sobre um CSV de questões e respostas, construído com **Ollama** + **sqlite-vec**. Arquitetura desenhada para permitir portabilidade futura para mobile (Expo/React Native).


## Visão geral

O usuário faz uma pergunta em linguagem livre. A pipeline:

1. **Busca** as `TOP_K` linhas do CSV mais similares por embedding, comparando pergunta contra pergunta.
2. **Gera** a resposta com um SLM, usando apenas o contexto recuperado (passo a passo numerado).

## Decisões arquiteturais

- **Sem LlamaIndex** — implementação manual sobre `sqlite-vec` para controle total do schema e portabilidade.
- **1 embedding por linha do CSV**: somente a coluna `questão` é indexada/buscada. Sem chunking, sem `CHUNK_OVERLAP`.
- `respostas_contextualizadas` **não é indexada** — é metadata entregue ao LLM como contexto na geração.
- Toda I/O de terminal fica isolada em `pipeline/cli.py`: o núcleo (`pipeline/core/`) não contém `input()`/`print()`.

## Estrutura planejada

```
projeto/
├── pipeline/
│   ├── config.py
│   ├── system_prompt.py
│   ├── core/
│   │   ├── ingest.py         # embedding por linha (questão) + storage
│   │   ├── retrieval.py      # busca por similaridade (questão)
│   │   └── generation.py     # chamada ao chat_model
│   ├── storage/
│   │   └── vector_store.db   # arquivo sqlite-vec
│   ├── data/
│   │   └── base_com_operacao.csv  # path definido em config.py (CSV_PATH)
│   ├── cli.py                # entrypoint terminal
│   └── requirements.txt
└── app-aritmetIA/
    └── (React Native + Expo — implementação futura)
```

## Configuração (`pipeline/config.py`)

| Chave | Valor | Status |
| --- | --- | --- |
| `CHAT_MODEL` | `gemma2:2b` | confirmado |
| `EMBED_MODEL` | `nomic-embed-text` | confirmado |
| `OLLAMA_HOST` | `http://localhost:11434` | confirmado |
| `TOP_K` | `3` | confirmado |
| `MAX_CONTEXT_CHARS` | `800` | **assumido** — validar |

Colunas do CSV: `ID`, `nível`, `operação` (obrigatória), `questão` (indexada), `comparação`, `respostas_contextualizadas` (contexto na geração, não indexada).

## Dependências

```
ollama
sqlite-vec
pandas
numpy
psutil
```

## Como executar

1. Instale as dependências: `pip install -r pipeline/requirements.txt`.
2. Garanta que o Ollama está rodando em `http://localhost:11434` com `CHAT_MODEL` e `nomic-embed-text` baixados.
3. Defina o caminho do CSV em `CSV_PATH` dentro de `pipeline/config.py`.

### CLI (assistente interativo)

```bash
python pipeline/cli.py                # RAG completo
python pipeline/cli.py --no-rag       # geração direta; não abre banco nem cria embeddings
python pipeline/cli.py --rerank       # experimental: reordena 9 candidatos e usa TOP_K
```

A ingestão é feita automaticamente na primeira execução (gera `storage/vector_store.db`).

### Harness de avaliação

```bash
# 1. Roda o pipeline nas amostras (modos rag e no-rag) e gera artefatos
python -m pipeline.eval.run_eval --eval

# 2. Avaliação de qualidade via LLM de fronteira (ChatGPT/Claude):
#    copie o conteúdo de pipeline/eval/output/judge_prompt.txt e cole no modelo.
#    Cole de volta o JSON de notas no comando abaixo (termine com a linha 'EOF'):

python -m pipeline.eval.run_eval --judge

# 3. (Opcional) medir hit@k / MRR variando o TOP_K
python -m pipeline.eval.run_eval --adaptive-k

# 4. (Opcional) apontar um eval_set customizado
python -m pipeline.eval.run_eval --eval --eval-set /caminho/para/eval_set.jsonl
```

Artefatos gerados em `pipeline/eval/output/`:

- `results.jsonl` — uma linha por resposta (consulta, contexto, resposta, métricas psutil, ground truth).
- `judge_prompt.txt` — bloco pronto para colar no LLM de fronteira.
- `judge_scores.jsonl` — notas consolidadas do juiz (após `--judge`).
- `metrics.json` — `hit@k`, MRR e notas do juiz por modo.

## Riscos / pendências

- **Portabilidade mobile do `sqlite-vec`**: Expo Go padrão não carrega extensões nativas SQLite arbitrárias sem *dev client* customizado (prebuild); validar antes de tratar o caminho de armazenamento como definitivo.
- Confirmar wheel pré-compilada de `sqlite-vec` para o SO/arquitetura de execução.
- `truncate_context` corta por caractere (`text[:max_chars]`), sem respeitar limite de frase — pode cortar uma explicação matemática no meio.
- O truncamento é por resposta individual, não um orçamento total: com `TOP_K=3` e `MAX_CONTEXT_CHARS=800`, o pior caso é ~2400 caracteres de contexto, além de `system_prompt` e pergunta. Deve ser validado contra a janela de contexto real do modelo.

## Roadmap

- [ ] Implementação do pipeline em Python.
- [ ] Validação de `MAX_CONTEXT_CHARS`.
- [ ] Etapa `app-aritmetIA` (React Native + Expo) — porta do núcleo para mobile.

## Avaliação de recuperação — schema v2

As instruções atualizadas estão em [specs/retrieval-metrics.md](specs/retrieval-metrics.md). Esta seção substitui os exemplos
anteriores de métricas e caminhos de saída: há cinco métricas, qrels opcionais,
cobertura de julgamentos e agregação com denominadores explícitos. Artefatos sem
`schema_version: 2` não são diretamente comparáveis ao novo recall/nDCG.

```bash
python -m pipeline.eval.run_eval --adaptive-k --qrels qrels.json --output-dir pipeline/eval/output/k-v2
python -m pipeline.eval.run_eval --eval --output-dir pipeline/eval/output/eval-v2
python -m pipeline.eval.run_eval --eval --rerank --output-dir pipeline/eval/output/rerank-v1
python -m pipeline.eval.run_eval --judge --output-dir pipeline/eval/output/eval-v2 \
  --judge-model "modelo-e-versão-declarados"
```

Sem `--qrels`, usa-se o csv_id esperado. Avaliações criam diretórios novos;
`--judge` recebe o diretório da avaliação para consolidar notas. Para testes:
`python -m pip install -r requirements-dev.txt` e `python -m pytest tests -q`.

Cada execução `--eval` agora grava também `manifest.json` (modelos, hashes dos
insumos, ordem dos modos e política de cache) e inclui `paired_comparison` em
`metrics.json`. Falhas são mantidas por `id`/modo em `results.jsonl`; métricas de
recuperação no modo `no-rag` são `null` por não serem aplicáveis.

O pacote `judge_prompt.txt` usa IDs opacos e ordem embaralhada reproduzível; o
mapa privado fica em `judge_map.json`. A importação da rubrica v2 exige cinco
notas (pedagogia, correção semântica, coerência lógica, completude e
fundamentação), justificativas e evidências. Ela preserva `judge_raw.json`,
informa itens ausentes sem atribuir nota zero e não permite sobrescrever uma
importação existente.

## Reranking experimental

O reranking permanece desligado por padrão. Quando solicitado por `--rerank`,
recupera `candidate_k=9` e escolhe `final_k=TOP_K` pelo método local e
determinístico `distance-operation-v1` (similaridade normalizada: 0,8;
compatibilidade de operação inferida: 0,2). Os artefatos de avaliação registram
ranks, distâncias originais, componentes e seleção. Compare um diretório baseline
e outro `--rerank` sobre o mesmo `eval_set` antes de habilitá-lo permanentemente.
