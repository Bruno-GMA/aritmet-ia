"""
CLI — entrypoint do pipeline RAG (spec §9).

Este é o ÚNICO módulo com input()/print().
Toda lógica de negócio fica em pipeline/core/*.
"""

from __future__ import annotations

import sys
import os

# Garante que o diretório raiz do projeto esteja no path para imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline.core import ingest, retrieval, generation
from pipeline.config import CSV_PATH


def _run_ingest() -> None:
    """Executa ingestão do CSV."""
    print("⏳ Carregando CSV...")
    df = ingest.load_csv()
    print(f"   {len(df)} linhas encontradas no CSV.")

    print("🔄 Gerando embeddings e indexando no sqlite-vec...")
    count = ingest.ingest(df)
    print(f"✅ {count} linhas indexadas com sucesso!\n")


def _ask_loop() -> None:
    """Loop principal de perguntas."""
    print("=" * 60)
    print("  aritmet.IA — Assistente de Matemática (RAG local)")
    print("  Digite 'sair' para encerrar.")
    print("=" * 60)
    print()

    while True:
        try:
            query = input("📝 Pergunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Até mais!")
            break

        if not query:
            continue
        if query.lower() in ("sair", "exit", "quit"):
            print("👋 Até mais!")
            break

        query_final = query

        # ── Retrieval ──────────────────────────────────────
        print("🔍 Buscando contexto relevante...")
        context = retrieval.search(query_final)

        if not context:
            print("⚠️  Nenhum resultado encontrado no banco vetorial.\n")
            continue

        # ── Generation ─────────────────────────────────────
        print("🤖 Gerando resposta...\n")
        answer = generation.generate(query_final, context)
        print(f"{answer}\n")


def main() -> None:
    """Entrypoint principal."""
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV não encontrado em: {CSV_PATH}")
        print("   Coloque o arquivo base.csv em pipeline/data/ e tente novamente.")
        sys.exit(1)

    _run_ingest()
    _ask_loop()


if __name__ == "__main__":
    main()
