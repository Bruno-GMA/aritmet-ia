"""
Configuração centralizada do pipeline RAG.
Todos os valores seguem a especificação em prompt_pipeline_rag_ollama.md §3.
"""

import os

# ── Modelos ────────────────────────────────────────────────
CHAT_MODEL = "gemma3:4b"
EMBED_MODEL = "nomic-embed-text"

OLLAMA_HOST = "http://localhost:11434"

# ── CSV ────────────────────────────────────────────────────
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "base.csv")

COLUMN_ID = "ID"
COLUMN_NIVEL = "nível"
EMBED_COLUMN = "questão"                       # única coluna indexada/embedada para busca
CONTEXT_COLUMN = "respostas_contextualizadas"  # entregue como contexto na geração, não indexada
COLUMN_COMPARACAO = "comparação"

# ── Banco vetorial ─────────────────────────────────────────
VECTOR_DB_PATH = os.path.join(os.path.dirname(__file__), "storage", "vector_store.db")

# ── Retrieval ──────────────────────────────────────────────
TOP_K = 3  # nº de linhas recuperadas por consulta

# ── Geração ────────────────────────────────────────────────
MAX_CONTEXT_CHARS = 800  # [ASSUMIDO] truncamento de segurança por resposta recuperada
