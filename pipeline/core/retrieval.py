"""
Busca por similaridade vetorial (cosseno) — spec §7.

Fluxo:
  1. Recebe query (corrigida ou original).
  2. Gera embedding da query via EMBED_MODEL.
  3. Busca no vec0 os TOP_K vizinhos mais próximos (embeddings de `questão`).
  4. Retorna linhas completas com metadados (incluindo respostas_contextualizadas).

Nenhum print/input aqui — toda I/O fica em cli.py (ver AGENTS.md).
"""

from __future__ import annotations

import sqlite3
import struct
from typing import Any

import sqlite_vec
import ollama

from pipeline.config import (
    EMBED_MODEL,
    OLLAMA_HOST,
    TOP_K,
    VECTOR_DB_PATH,
)


def _embed(text: str) -> list[float]:
    """Gera embedding via Ollama."""
    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.embed(model=EMBED_MODEL, input=text)
    return resp["embeddings"][0]


def _serialize_f32(vec: list[float]) -> bytes:
    """Serializa vetor de floats em little-endian float32 (formato sqlite-vec)."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _get_connection() -> sqlite3.Connection:
    """Abre conexão somente-leitura e carrega extensão sqlite-vec."""
    db = sqlite3.connect(VECTOR_DB_PATH)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def search(query: str, top_k: int = TOP_K) -> list[dict[str, Any]]:
    """
    Busca as TOP_K linhas mais similares à query.

    Retorna lista de dicts com chaves:
        rowid, csv_id, nivel, questao, comparacao,
        resposta_contextualizada, distance
    """
    query_vec = _embed(query)
    query_blob = _serialize_f32(query_vec)

    db = _get_connection()
    cur = db.cursor()

    rows = cur.execute(
        """
        SELECT
            v.rowid,
            v.distance,
            m.csv_id,
            m.nivel,
            m.questao,
            m.comparacao,
            m.resposta_contextualizada
        FROM questions_vec v
        JOIN questions_meta m ON m.rowid = v.rowid
        WHERE v.embedding MATCH ?
            AND v.k = ?
        ORDER BY v.distance
        """,
        (query_blob, top_k),
    ).fetchall()

    db.close()

    results: list[dict[str, Any]] = []
    for row in rows:
        results.append(
            {
                "rowid": row[0],
                "distance": row[1],
                "csv_id": row[2],
                "nivel": row[3],
                "questao": row[4],
                "comparacao": row[5],
                "resposta_contextualizada": row[6],
            }
        )
    return results
