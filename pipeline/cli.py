"""
CLI — entrypoint do pipeline RAG (spec §9).

Este é o ÚNICO módulo com input()/print().
Toda lógica de negócio fica em pipeline/core/*.
"""

from __future__ import annotations

import argparse
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


def _ask_loop(use_rag: bool) -> None:
    """Loop principal de perguntas."""
    print("=" * 60)
    print("  aritmet.IA — Assistente de Matemática (RAG local)")
    print("  Digite 'sair' para encerrar.")
    if not use_rag:
        print("  [modo --no-rag] geração sem recuperação de contexto.")
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
        context = []

        # ── Retrieval (pulado em --no-rag) ──────────────────
        if use_rag:
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
    parser = argparse.ArgumentParser(description="aritmet.IA — RAG local")
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Pula a etapa de retrieval (baseline sem contexto).",
    )
    args = parser.parse_args()

    if not args.no_rag and not os.path.exists(CSV_PATH):
        print(f"❌ CSV não encontrado em: {CSV_PATH}")
        print("   Coloque o arquivo base_com_operacao.csv em pipeline/data/ e tente novamente.")
        sys.exit(1)

    if not args.no_rag:
        _run_ingest()
    else:
        print("ℹ️  Modo --no-rag: banco vetorial e embeddings não são necessários.")
    _ask_loop(use_rag=not args.no_rag)


if __name__ == "__main__":
    main()
