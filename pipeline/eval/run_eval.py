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
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from pipeline.config import TOP_K, TOP_K_RANGE
from pipeline.core import adaptive_k as adaptive_k_mod
from pipeline.core import generation, ingest, retrieval
from pipeline.core.monitoring import Profiler, summarize_stage_records
from pipeline.eval import dataset, judge_import, judge_prompt, math_evaluation, metrics
from pipeline.eval.ranking import Qrels

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _new_output_dir(output_dir: str | None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(output_dir) if output_dir else Path(OUTPUT_DIR) / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def _load_qrels(path: str | None, corpus_ids: list[str]) -> Qrels | None:
    if path is None:
        return None
    result = Qrels.from_json(Path(path).read_text(encoding="utf-8"))
    result.validate_corpus(corpus_ids)
    return result


def _load_reference(csv_id: str) -> str:
    """Busca a referência (gabarito) de uma linha pelo csv_id no CSV."""
    df = ingest.load_csv()
    row = df[df["ID"].astype(str) == str(csv_id)]
    if row.empty:
        return ""
    r = row.iloc[0]
    return str(r.get("respostas_contextualizadas", "")) or str(r.get("comparação", ""))


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


def run_eval(
    eval_set_path: str, qrels_path: str | None = None, output_dir: str | None = None
) -> None:
    """Roda o pipeline em cada amostra, nos modos rag e no-rag."""
    _run_ingest_if_needed()

    samples = dataset.load_eval_set(eval_set_path)
    corpus_ids = retrieval.corpus_ids()
    qrels = _load_qrels(qrels_path, corpus_ids)
    output = _new_output_dir(output_dir)
    results_path = output / "results.jsonl"
    judge_path = output / "judge_prompt.txt"
    per_query: list[dict] = []
    math_rows: list[dict] = []
    stage_records: list[dict] = []
    run_id = output.name

    judge_items: list[dict] = []
    expected_ops: list[str] = []
    retrieved_ops: list[list[str]] = []

    with open(results_path, "w", encoding="utf-8") as out:
        for sample in samples:
            query = sample["questão"]
            expected = str(sample["csv_id_esperado"])
            op = sample.get("operação", "")
            expected_ops.append(op)

            for mode in ("rag", "no-rag"):
                context: list[dict] = []
                profiler_retrieval: dict | None = None
                ids: list[str] = []
                ops: list[str] = []

                if mode == "rag":
                    with Profiler(
                        "retrieval", run_id=run_id, query_id=sample["id"], mode=mode
                    ) as p:
                        context = retrieval.search(query, top_k=TOP_K)
                    profiler_retrieval = p.result
                    if profiler_retrieval:
                        stage_records.append(profiler_retrieval)
                    ids = [str(r.get("csv_id", "")) for r in context]
                    ops = [str(r.get("operacao", "")) for r in context]
                    retrieved_ops.append(ops)
                    per_query.append(
                        metrics.evaluate_retrieval(
                            ids,
                            expected,
                            TOP_K,
                            corpus_ids=corpus_ids,
                            query_id=sample["id"],
                            qrels=qrels,
                        )
                    )

                with Profiler("generation", run_id=run_id, query_id=sample["id"], mode=mode) as pg:
                    answer = generation.generate(query, context)
                profiler_generation = pg.result
                if profiler_generation:
                    stage_records.append(profiler_generation)
                math_result = math_evaluation.evaluate_analogous_answer(answer)
                math_rows.append(math_result)

                record = {
                    "id": sample["id"],
                    "mode": mode,
                    "questão": query,
                    "nível": sample["nível"],
                    "operação": sample.get("operação", ""),
                    "csv_id_esperado": expected,
                    "contexto_recuperado": ids,
                    "resposta_llm": answer,
                    "math_evaluation": math_result,
                    "retrieval_quality": per_query[-1] if mode == "rag" else None,
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

    summary = {
        **metrics.aggregate(per_query),
        "legacy_compatible": False,
        "num_samples": len(samples),
        "top_k": TOP_K,
        "per_query": per_query,
        "consistency_rag": (
            sum(
                metrics.consistency_at_k(ops, gold)
                for ops, gold in zip(retrieved_ops, expected_ops)
            )
            / len(samples)
            if samples
            else None
        ),
        "no_rag_retrieval": None,
        "math_evaluation": math_evaluation.aggregate_math_evaluations(math_rows),
        "resource_measurements": summarize_stage_records(stage_records),
    }
    with (output / "metrics.json").open("w", encoding="utf-8") as mf:
        json.dump(summary, mf, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"Avaliação concluída: {len(samples)} amostras x 2 modos. Artefatos: {output}")


def run_adaptive_k(
    eval_set_path: str, qrels_path: str | None = None, output_dir: str | None = None
) -> None:
    """Export all five metrics by k and operation, without generating answers."""
    _run_ingest_if_needed()
    samples = dataset.load_eval_set(eval_set_path)
    corpus_ids = retrieval.corpus_ids()
    qrels = _load_qrels(qrels_path, corpus_ids)
    output = _new_output_dir(output_dir)
    rows_by_k: dict[int, list[dict]] = {k: [] for k in TOP_K_RANGE}
    rows_by_op: dict[str, dict[int, list[dict]]] = {}
    for sample in samples:
        op = sample.get("operação", "")
        result = adaptive_k_mod.evaluate_k(
            sample["questão"],
            str(sample["csv_id_esperado"]),
            TOP_K_RANGE,
            operacao_gold=op,
            corpus_ids=corpus_ids,
            query_id=sample["id"],
            qrels=qrels,
        )
        per_op = rows_by_op.setdefault(op, {k: [] for k in TOP_K_RANGE})
        for k, row in result.items():
            quality = row["metrics"]
            quality["operation_consistency"] = float(row["operacao_consistente"])
            rows_by_k[k].append(quality)
            per_op[k].append(quality)

    def summarize(groups: dict[int, list[dict]]) -> dict[int, dict]:
        return {
            k: {
                **metrics.aggregate(rows),
                "per_query": rows,
                "operation_consistency": (
                    sum(r["operation_consistency"] for r in rows) / len(rows) if rows else None
                ),
            }
            for k, rows in groups.items()
        }

    payload = {
        "schema_version": 2,
        "legacy_compatible": False,
        "by_k": summarize(rows_by_k),
        "by_operation": {op: summarize(groups) for op, groups in rows_by_op.items()},
    }
    with (output / "adaptive_k_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, allow_nan=False)
    for k, summary in payload["by_k"].items():
        print(f"k={k}: {summary['metrics']}")
    print(f"Artefatos: {output}")


def run_judge(output_dir: str) -> None:
    """Reaceita notas coladas do LLM de fronteira e consolida."""
    if not Path(output_dir).is_dir():
        raise ValueError("Diretório de avaliação inexistente")
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

    scores_path = os.path.join(output_dir, "judge_scores.jsonl")
    with open(scores_path, "w", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # consolida com métricas existentes
    metrics_path = os.path.join(output_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, encoding="utf-8") as f:
            base = json.load(f)
    else:
        base = {}

    by_mode: dict[str, list[int]] = {}
    for s in scores:
        by_mode.setdefault(s["mode"], []).append(s["score"])

    base["judge_scores"] = {
        mode: {"mean": sum(v) / len(v), "count": len(v)} for mode, v in by_mode.items()
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
    parser.add_argument("--qrels", help="Julgamentos de relevância JSON versionados")
    parser.add_argument(
        "--output-dir", help="Novo diretório para avaliação; existente para --judge"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--eval", action="store_true", help="Roda o pipeline (rag/no-rag)")
    group.add_argument("--adaptive-k", action="store_true", help="Mede hit@k/MRR")
    group.add_argument("--judge", action="store_true", help="Reaceita notas do juiz")

    args = parser.parse_args()
    eval_set_path = os.path.abspath(args.eval_set)

    if args.eval:
        run_eval(eval_set_path, args.qrels, args.output_dir)
    elif args.adaptive_k:
        run_adaptive_k(eval_set_path, args.qrels, args.output_dir)
    elif args.judge:
        if not args.output_dir:
            parser.error("--judge requer --output-dir da avaliação a consolidar")
        run_judge(args.output_dir)


if __name__ == "__main__":
    main()
