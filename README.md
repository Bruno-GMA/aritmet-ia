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
│   │   └── base.csv          # path definido em config.py (CSV_PATH)
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

Colunas do CSV: `ID`, `nível`, `questão` (indexada), `comparação`, `respostas_contextualizadas` (contexto na geração, não indexada).

## Dependências

```
ollama
sqlite-vec
pandas
numpy
```

## Como executar

1. Instale as dependências: `pip install -r requirements.txt` (arquivo gerado a partir da seção acima).
2. Garanta que o Ollama está rodando em `http://localhost:11434` com `gemma2:2b` e `nomic-embed-text`.
3. Defina o caminho do CSV em `CSV_PATH` dentro de `config.py`.
4. Gere os embeddings: execute a ingestão (`pipeline/core/ingest.py`).
5. Rode o CLI: `python pipeline/cli.py`.

## Riscos / pendências

- **Portabilidade mobile do `sqlite-vec`**: Expo Go padrão não carrega extensões nativas SQLite arbitrárias sem *dev client* customizado (prebuild); validar antes de tratar o caminho de armazenamento como definitivo.
- Confirmar wheel pré-compilada de `sqlite-vec` para o SO/arquitetura de execução.
- `truncate_context` corta por caractere (`text[:max_chars]`), sem respeitar limite de frase — pode cortar uma explicação matemática no meio.
- O truncamento é por resposta individual, não um orçamento total: com `TOP_K=3` e `MAX_CONTEXT_CHARS=800`, o pior caso é ~2400 caracteres de contexto, além de `system_prompt` e pergunta. Deve ser validado contra a janela de contexto real do modelo.

## Roadmap

- [ ] Implementação do pipeline em Python.
- [ ] Validação de `MAX_CONTEXT_CHARS`.
- [ ] Etapa `app-aritmetIA` (React Native + Expo) — porta do núcleo para mobile.