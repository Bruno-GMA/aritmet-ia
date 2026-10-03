"""
Orquestrador de avaliação — ÚNICO módulo do eval com I/O (spec eval-harness.md).

Subcomandos:
  --eval            roda o pipeline por amostra e modo, gera artefatos.
  --adaptive-k      mede hit@k/MRR variando k (core/adaptive_k.py).
  --judge           reaceita notas coladas do LLM de fronteira e consolida.

Exemplos:
  python -m pipeline.eval.run_eval --eval
  python -m pipeline.eval.run_eval --adaptive-k
  python -m pipeline.eval.run_eval --judge
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.config import CSV_PATH, TOP_K, TOP_K_RANGE
from pipeline.core import generation, ingest, retrieval
from pipeline.core import adaptive_k as adaptive_k_mod
from pipeline.core.monitoring import Profiler
from pipeline.eval import dataset, judge_import, judge_prompt, metrics

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _load_reference(csv_id: str) -> str:
    """Busca a referência (gabarito) de uma linha pelo csv_id no CSV."""
    df = ingest.load_csv()
    row = df[df["ID"].astype(str) == str(csv_id)]
    if row.empty:
        return ""
    r = row.iloc[0]
    return str(r.get("respostas_contextualizadas", "")) or str(
        r.get("comparação", "")
    )


def _load_operacao(csv_id: str) -> str:
    """Busca a operação de uma linha pelo csv_id no CSV."""
    df = ingest.load_csv()
    row = df[df["ID"].astype(str) == str(csv_id)]
    if row.empty:
        return ""
    return str(row.iloc[0].get("operação", ""))


def _run_ingest_if_needed() -> bool:
    """Ingere o CSV se o banco não existir."""
    from pipeline.config import VECTOR_DB_PATH

    if os.path.exists(VECTOR_DB_PATH):
        return False
    print("⏳ Banco vetorial não encontrado. Executando ingestão...")
    df = ingest.load_csv()
    ingest.ingest(df)
    print("✅ Ingestão concluída.")
    return True


def run_eval(eval_set_path: str) -> None:
    """Roda o pipeline em cada amostra, nos modos rag e no-rag."""
    _ensure_output_dir()
    _run_ingest_if_needed()

    samples = dataset.load_eval_set(eval_set_path)
    results_path = os.path.join(OUTPUT_DIR, "results.jsonl")
    judge_path = os.path.join(OUTPUT_DIR, "judge_prompt.txt")

    judge_items: list[dict] = []
    retrieved_rag: list[list[str]] = []
    expected_ids: list[str] = []
    expected_ops: list[str] = []
    retrieved_ops: list[list[str]] = []

    with open(results_path, "w", encoding="utf-8") as out:
        for sample in samples:
            query = sample["questão"]
            expected = str(sample["csv_id_esperado"])
            op = sample.get("operação", "")
            expected_ids.append(expected)
            expected_ops.append(op)

            for mode in ("rag", "no-rag"):
                context: list[dict] = []
                profiler_retrieval: dict | None = None
                ids: list[str] = []
                ops: list[str] = []

                if mode == "rag":
                    with Profiler("retrieval") as p:
                        context = retrieval.search(query, top_k=TOP_K)
                    profiler_retrieval = p.result
                    ids = [str(r.get("csv_id", "")) for r in context]
                    ops = [str(r.get("operacao", "")) for r in context]
                    retrieved_rag.append(ids)
                    retrieved_ops.append(ops)

                with Profiler("generation") as pg:
                    answer = generation.generate(query, context)
                profiler_generation = pg.result

                record = {
                    "id": sample["id"],
                    "mode": mode,
                    "questão": query,
                    "nível": sample["nível"],
                    "operação": sample.get("operação", ""),
                    "csv_id_esperado": expected,
                    "contexto_recuperado": ids,
                    "resposta_llm": answer,
                    "retrieval_metrics": profiler_retrieval,
                    "generation_metrics": profiler_generation,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

                # itens para o judge (apenas referência + resposta)
                judge_items.append(
                    {
                        "id": sample["id"],
                        "mode": mode,
                        "questão": query,
                        "referencia": _load_reference(expected),
                        "resposta_llm": answer,
                    }
                )

    # prompt do juiz (um batch por modo)
    with open(judge_path, "w", encoding="utf-8") as jf:
        for mode in ("rag", "no-rag"):
            subset = [x for x in judge_items if x.get("mode") == mode]
            jf.write(f"===== MODO: {mode} =====\n")
            jf.write(judge_prompt.build_judge_batch(subset, mode))
            jf.write("\n\n")

    # métricas de retrieval (só modo rag tem recuperação; no-rag usa lista vazia)
    summary = {
        "num_samples": len(samples),
        "top_k": TOP_K,
        "mrr_rag": metrics.mrr(retrieved_rag, expected_ids),
        "hit_at_k_rag": sum(
            metrics.hit_at_k(retrieved_rag[i], expected_ids[i], TOP_K)
            for i in range(len(samples))
        )
        / len(samples),
        "consistency_rag": sum(
            metrics.consistency_at_k(retrieved_ops[i], expected_ops[i])
            for i in range(len(samples))
        )
        / len(samples),
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as mf:
        json.dump(summary, mf, ensure_ascii=False, indent=2)

    print(f"✅ Avaliação concluída: {len(samples)} amostras x 2 modos.")
    print(f"   resultados: {results_path}")
    print(f"   prompt do juiz: {judge_path}")
    print(f"   métricas (sem notas do juiz): {os.path.join(OUTPUT_DIR, 'metrics.json')}")


def run_adaptive_k(eval_set_path: str) -> None:
    """Mede hit@k, MRR e consistência de operação para cada k em TOP_K_RANGE.

    Reporta global e estratificado por operação (DPR/BEIR/RAGAS).
    """
    _run_ingest_if_needed()
    samples = dataset.load_eval_set(eval_set_path)

    # acumula resultado por k sobre todas as amostras
    def _new_agg() -> dict[int, dict[str, float | int]]:
        return {k: {"hits": 0, "rr_sum": 0.0, "cons_sum": 0.0} for k in TOP_K_RANGE}

    agg_global = _new_agg()
    agg_by_op: dict[str, dict[int, dict[str, float | int]]] = {}

    for i, sample in enumerate(samples):
        op = sample.get("operação", "")
        row = adaptive_k_mod.evaluate_k(
            sample["questão"],
            str(sample["csv_id_esperado"]),
            TOP_K_RANGE,
            operacao_gold=op,
        )

        per_op = agg_by_op.setdefault(op, _new_agg())
        for k, v in row.items():
            for agg in (agg_global, per_op):
                agg[k]["hits"] += 1 if v["hit"] else 0
                agg[k]["rr_sum"] += adaptive_k_mod.mrr_for_row(row, k)
                agg[k]["cons_sum"] += adaptive_k_mod.consistency_at_k(row, k)

    def _count(op: str) -> int:
        return sum(1 for s in samples if s.get("operação", "") == op)

    def _print_table(title: str, agg: dict[int, dict[str, float | int]], n: int) -> None:
        print(f"\n{title} (n={n})")
        print(f"{'k':>2} | {'hit@k':>6} | {'MRR':>6} | {'cons@k':>6}")
        print("-" * 32)
        for k in TOP_K_RANGE:
            hit = agg[k]["hits"] / n if n else 0.0
            m = agg[k]["rr_sum"] / n if n else 0.0
            c = agg[k]["cons_sum"] / n if n else 0.0
            print(f"{k:>2} | {hit:6.3f} | {m:6.3f} | {c:6.3f}")

    _print_table("GLOBAL", agg_global, len(samples))
    for op in sorted(agg_by_op.keys()):
        _print_table(f"OPERAÇÃO: {op}", agg_by_op[op], _count(op))


def run_judge() -> None:
    """Reaceita notas coladas do LLM de fronteira e consolida."""
    _ensure_output_dir()
    print("Cole o JSON de notas retornado pelo LLM de fronteira.")
    print("Termine com uma linha contendo 'EOF':")
    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == "EOF":
            break
        lines.append(line)

    text = "\n".join(lines)
    scores = judge_import.parse_judge_output(text)

    scores_path = os.path.join(OUTPUT_DIR, "judge_scores.jsonl")
    with open(scores_path, "w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # consolida com métricas existentes
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            base = json.load(f)
    else:
        base = {}

    by_mode: dict[str, list[int]] = {}
    for s in scores:
        by_mode.setdefault(s["mode"], []).append(s["score"])

    base["judge_scores"] = {
        mode: {"mean": sum(v) / len(v), "count": len(v)}
        for mode, v in by_mode.items()
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(scores)} notas importadas.")
    print(f"   notas: {scores_path}")
    print(f"   métricas atualizadas: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Harness de avaliação do pipeline RAG")
    parser.add_argument(
        "--eval-set",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.jsonl"),
        help="Caminho do eval_set.jsonl",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--eval", action="store_true", help="Roda o pipeline (rag/no-rag)")
    group.add_argument("--adaptive-k", action="store_true", help="Mede hit@k/MRR")
    group.add_argument("--judge", action="store_true", help="Reaceita notas do juiz")

    args = parser.parse_args()
    eval_set_path = os.path.abspath(args.eval_set)

    if args.eval:
        run_eval(eval_set_path)
    elif args.adaptive_k:
        run_adaptive_k(eval_set_path)
    elif args.judge:
        run_judge()


if __name__ == "__main__":
    main()