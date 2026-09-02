"""
Ingestão do CSV → embeddings → sqlite-vec.

Spec §6:
  1. Lê CSV com pandas.
  2. Para cada linha, gera UM embedding da coluna `questão` (EMBED_COLUMN).
  3. Persiste no sqlite-vec (vec0): id, texto da questão, embedding,
     metadados completos (nível, comparação, respostas_contextualizadas).

Nenhum print/input aqui — toda I/O fica em cli.py (ver AGENTS.md).
"""

from __future__ import annotations

import json
import os
import struct
from typing import Any

import numpy as np
import pandas as pd
import sqlite3
import sqlite_vec
import ollama

from pipeline.config import (
    CSV_PATH,
    COLUMN_ID,
    COLUMN_NIVEL,
    EMBED_COLUMN,
    CONTEXT_COLUMN,
    COLUMN_COMPARACAO,
    EMBED_MODEL,
    VECTOR_DB_PATH,
    OLLAMA_HOST,
)


# ── helpers ────────────────────────────────────────────────


def _embed(text: str) -> list[float]:
    """Gera embedding via Ollama para um único texto."""
    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.embed(model=EMBED_MODEL, input=text)
    return resp["embeddings"][0]


def _serialize_f32(vec: list[float]) -> bytes:
    """Serializa vetor de floats em little-endian float32 (formato sqlite-vec)."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _get_connection() -> sqlite3.Connection:
    """Abre conexão e carrega extensão sqlite-vec."""
    os.makedirs(os.path.dirname(VECTOR_DB_PATH), exist_ok=True)
    db = sqlite3.connect(VECTOR_DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


# ── API pública ────────────────────────────────────────────


def load_csv() -> pd.DataFrame:
    """Carrega e valida o CSV base."""
    required = [COLUMN_ID, COLUMN_NIVEL, EMBED_COLUMN, COLUMN_COMPARACAO, CONTEXT_COLUMN]
    df = pd.read_csv(CSV_PATH)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")
    return df


def ingest(df: pd.DataFrame) -> int:
    """
    Gera embeddings e persiste no sqlite-vec.

    Retorna o número de linhas indexadas.
    """
    db = _get_connection()
    cur = db.cursor()

    # Descobre dimensão do embedding gerando um de amostra
    sample_vec = _embed("teste")
    dim = len(sample_vec)

    # Recria tabelas
    cur.execute("DROP TABLE IF EXISTS questions_vec")
    cur.execute("DROP TABLE IF EXISTS questions_meta")

    # Tabela de metadados (rowid será o ID compartilhado com vec0)
    cur.execute(
        """
        CREATE TABLE questions_meta (
            rowid   INTEGER PRIMARY KEY,
            csv_id  TEXT,
            nivel   TEXT,
            questao TEXT,
            comparacao TEXT,
            resposta_contextualizada TEXT
        )
        """
    )

    # Tabela virtual vec0
    cur.execute(
        f"CREATE VIRTUAL TABLE questions_vec USING vec0(embedding float[{dim}])"
    )

    count = 0
    for idx, row in df.iterrows():
        questao_text = str(row[EMBED_COLUMN])
        vec = _embed(questao_text)
        blob = _serialize_f32(vec)

        rowid = int(idx) + 1  # 1-based

        cur.execute(
            "INSERT INTO questions_meta (rowid, csv_id, nivel, questao, comparacao, resposta_contextualizada) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                rowid,
                str(row[COLUMN_ID]),
                str(row.get(COLUMN_NIVEL, "")),
                questao_text,
                str(row.get(COLUMN_COMPARACAO, "")),
                str(row.get(CONTEXT_COLUMN, "")),
            ),
        )

        cur.execute(
            "INSERT INTO questions_vec (rowid, embedding) VALUES (?, ?)",
            (rowid, blob),
        )
        count += 1

    db.commit()
    db.close()
    return count
