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
import hashlib
import json
import os
import random
import statistics
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
from pipeline.system_prompt import get_system_prompt

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def _new_output_dir(output_dir: str | None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    path = Path(output_dir) if output_dir else Path(OUTPUT_DIR) / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def _sha256(path: str | Path) -> str:
    """Return a stable input fingerprint without copying the experiment input."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(
    output: Path, eval_set_path: str, qrels_path: str | None, rerank_enabled: bool = False
) -> dict:
    """Persist conditions shared by both modes before any model call."""
    from pipeline.config import (
        CHAT_MODEL,
        CSV_PATH,
        EMBED_MODEL,
        RERANK_CANDIDATE_K,
        TOP_K,
        VECTOR_DB_PATH,
    )

    manifest = {
        "schema_version": 1,
        "run_id": output.name,
        "project": "aritmet.IA",
        "modes": ["rag", "no-rag"],
        "mode_order": ["rag", "no-rag"],
        "repetitions": 1,
        "seed": None,
        "generation": {"model": CHAT_MODEL, "output_budget": None},
        "embedding_model": EMBED_MODEL,
        "reranking": {
            "enabled": rerank_enabled,
            "candidate_k": RERANK_CANDIDATE_K,
            "final_k": TOP_K,
            "method": "distance-operation-v1",
            "decision_threshold": "pre_registered: report only; no automatic enablement",
        },
        "inputs": {
            "evaluation_set": {"path": str(Path(eval_set_path).resolve()), "sha256": _sha256(eval_set_path)},
            "corpus": {"path": str(Path(CSV_PATH).resolve()), "sha256": _sha256(CSV_PATH)},
            "qrels": (
                {"path": str(Path(qrels_path).resolve()), "sha256": _sha256(qrels_path)}
                if qrels_path
                else None
            ),
            "vector_index": str(Path(VECTOR_DB_PATH).resolve()),
        },
        "notes": {
            "retrieval_metrics_no_rag": "not_applicable",
            "cache_policy": "no response cache; the vector corpus is shared read-only",
        },
    }
    with (output / "manifest.json").open("x", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
    return manifest


def _paired_summary(records: list[dict]) -> dict:
    """Summarize only complete rag/no-rag pairs; failures remain explicit."""
    grouped: dict[str, dict[str, dict]] = {}
    for record in records:
        grouped.setdefault(record["id"], {})[record["mode"]] = record
    pairs = [entry for entry in grouped.values() if set(entry) == {"rag", "no-rag"}]

    def elapsed(record: dict) -> float | None:
        metric = record.get("generation_metrics") or {}
        value = metric.get("elapsed_ms")
        return float(value) / 1000 if isinstance(value, int | float) else None

    deltas = [
        elapsed(pair["rag"]) - elapsed(pair["no-rag"])
        for pair in pairs
        if pair["rag"].get("status") == pair["no-rag"].get("status") == "ok"
        and elapsed(pair["rag"]) is not None
        and elapsed(pair["no-rag"]) is not None
    ]
    mean = sum(deltas) / len(deltas) if deltas else None
    dispersion = (
        (sum((value - mean) ** 2 for value in deltas) / len(deltas)) ** 0.5 if deltas else None
    )
    return {
        "total_query_ids": len(grouped),
        "valid_pairs": len(deltas),
        "incomplete_pairs": len(grouped) - len(pairs),
        "failed_pairs": sum(any(row.get("status") != "ok" for row in pair.values()) for pair in pairs),
        "generation_latency_delta_rag_minus_no_rag_s": {
            "mean": mean,
            "population_stddev": dispersion,
            "values": deltas,
        },
    }


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
    eval_set_path: str,
    qrels_path: str | None = None,
    output_dir: str | None = None,
    rerank_enabled: bool = False,
) -> None:
    """Roda o pipeline em cada amostra, nos modos rag e no-rag."""
    _run_ingest_if_needed()

    samples = dataset.load_eval_set(eval_set_path)
    corpus_ids = retrieval.corpus_ids()
    qrels = _load_qrels(qrels_path, corpus_ids)
    output = _new_output_dir(output_dir)
    manifest = _write_manifest(output, eval_set_path, qrels_path, rerank_enabled)
    results_path = output / "results.jsonl"
    judge_path = output / "judge_prompt.txt"
    per_query: list[dict] = []
    math_rows: list[dict] = []
    stage_records: list[dict] = []
    records: list[dict] = []
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
                retrieval_quality: dict | None = None
                rerank_trace: list[dict] | None = None
                ids: list[str] = []
                ops: list[str] = []

                try:
                    if mode == "rag":
                        with Profiler(
                            "retrieval", run_id=run_id, query_id=sample["id"], mode=mode
                        ) as p:
                            found = retrieval.search(
                                query,
                                top_k=TOP_K,
                                rerank_enabled=rerank_enabled,
                                return_rerank_trace=True,
                            )
                            context, rerank_trace = found  # type: ignore[misc]
                        profiler_retrieval = p.result
                        if profiler_retrieval:
                            stage_records.append(profiler_retrieval)
                        ids = [str(r.get("csv_id", "")) for r in context]
                        ops = [str(r.get("operacao", "")) for r in context]
                        retrieved_ops.append(ops)
                        retrieval_quality = metrics.evaluate_retrieval(
                            ids,
                            expected,
                            TOP_K,
                            corpus_ids=corpus_ids,
                            query_id=sample["id"],
                            qrels=qrels,
                        )
                        per_query.append(retrieval_quality)
                    with Profiler("generation", run_id=run_id, query_id=sample["id"], mode=mode) as pg:
                        answer = generation.generate(query, context)
                    profiler_generation = pg.result
                    if profiler_generation:
                        stage_records.append(profiler_generation)
                    status, error = "ok", None
                except Exception as exc:
                    answer, profiler_generation = "", None
                    status, error = "error", f"{type(exc).__name__}: {exc}"
                math_result = math_evaluation.evaluate_analogous_answer(answer)
                math_rows.append(math_result)
                sent_context = generation.build_context_block(context) if context else None

                record = {
                    "id": sample["id"],
                    "mode": mode,
                    "questão": query,
                    "nível": sample["nível"],
                    "operação": sample.get("operação", ""),
                    "csv_id_esperado": expected,
                    "contexto_recuperado": ids,
                    "contexto_enviado": sent_context,
                    "reranking": {
                        "enabled": rerank_enabled,
                        "candidate_k": manifest["reranking"]["candidate_k"],
                        "final_k": TOP_K,
                        "method": manifest["reranking"]["method"] if rerank_enabled else None,
                        "candidates": rerank_trace,
                    },
                    "resposta_llm": answer,
                    "math_evaluation": math_result,
                    "retrieval_quality": retrieval_quality,
                    "retrieval_metrics": profiler_retrieval,
                    "generation_metrics": profiler_generation,
                    "status": status,
                    "error": error,
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)

                # itens para o judge (apenas referência + resposta)
                judge_items.append(
                    {
                        "id": sample["id"],
                        "mode": mode,
                        "questão": query,
                        "referencia": _load_reference(expected),
                        "instrucao_pedagogica": get_system_prompt(),
                        "contexto_enviado": sent_context,
                        "resposta_llm": answer,
                    }
                )

    # O judge sees opaque IDs only.  The reproducible permutation helps reduce
    # ordering/mode bias, while the local map preserves full traceability.
    judge_seed = int(hashlib.sha256(run_id.encode()).hexdigest()[:16], 16)
    judge_rng = random.Random(judge_seed)
    judge_rng.shuffle(judge_items)
    judge_map: dict[str, dict[str, str]] = {}
    for index, item in enumerate(judge_items, 1):
        judge_id = f"j{index:04d}-{hashlib.sha256((run_id + item['id'] + item['mode']).encode()).hexdigest()[:8]}"
        item["judge_id"] = judge_id
        judge_map[judge_id] = {
            "run_id": run_id,
            "query_id": str(item["id"]),
            "mode": str(item["mode"]),
        }
    with (output / "judge_map.json").open("x", encoding="utf-8") as jf:
        json.dump({"seed": judge_seed, "items": judge_map}, jf, ensure_ascii=False, indent=2)

    with open(judge_path, "w", encoding="utf-8") as jf:
        jf.write(judge_prompt.build_judge_batch(judge_items))

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
        "manifest": {"run_id": manifest["run_id"], "path": "manifest.json"},
        "judge": {
            "rubric_version": judge_prompt.RUBRIC_VERSION,
            "blind_order_seed": judge_seed,
            "map_path": "judge_map.json",
            "status": "awaiting_manual_review",
        },
        "paired_comparison": _paired_summary(records),
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


def _aggregate_judge_scores(scores: list[dict]) -> dict:
    """Aggregate valid observations only; non-applicable grounding is excluded."""
    groups: dict[tuple[str, str, str], list[int]] = {}
    for score in scores:
        for criterion, value in score["criteria"].items():
            if criterion == "fundamentacao" and value.get("status") == "not_applicable":
                continue
            groups.setdefault((score["mode"], score["operation"], criterion), []).append(value["score"])

    def summary(values: list[int]) -> dict:
        return {
            "mean": statistics.fmean(values) if values else None,
            "population_stddev": statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None,
            "valid_count": len(values),
        }

    by_mode: dict[str, dict[str, dict]] = {}
    for (mode, operation, criterion), values in groups.items():
        by_mode.setdefault(mode, {}).setdefault(operation, {})[criterion] = summary(values)
    return {"by_mode_and_operation": by_mode, "weighting": "none"}


def run_judge(output_dir: str, judge_model: str | None = None) -> None:
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
    output = Path(output_dir)
    map_path = output / "judge_map.json"
    results_path = output / "results.jsonl"
    if not map_path.is_file() or not results_path.is_file():
        raise ValueError("Avaliação não contém judge_map.json e results.jsonl da rubrica v2.")
    score_path = output / "judge_scores.jsonl"
    raw_path = output / "judge_raw.json"
    if score_path.exists() or raw_path.exists():
        raise FileExistsError("Notas do juiz já foram importadas para este diretório.")
    scores = judge_import.parse_judge_output(text)
    judge_map = json.loads(map_path.read_text(encoding="utf-8"))["items"]
    judge_ids = [score["id"] for score in scores]
    if len(judge_ids) != len(set(judge_ids)):
        raise ValueError("A saída do juiz contém IDs duplicados.")
    unknown = set(judge_ids) - set(judge_map)
    if unknown:
        raise ValueError(f"A saída do juiz contém IDs desconhecidos: {sorted(unknown)}")
    result_index = {
        (str(row["id"]), str(row["mode"])): row
        for row in (
            json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line
        )
    }
    for score in scores:
        mapping = judge_map[score["id"]]
        result = result_index[(mapping["query_id"], mapping["mode"])]
        score.update(
            {
                "run_id": mapping["run_id"],
                "query_id": mapping["query_id"],
                "mode": mapping["mode"],
                "operation": result.get("operação", ""),
            }
        )
    missing = sorted(set(judge_map) - set(judge_ids))
    with raw_path.open("x", encoding="utf-8") as raw_file:
        json.dump(
            {
                "imported_at": datetime.now(UTC).isoformat(),
                "judge_model": judge_model or "not_declared",
                "raw_output": text,
                "missing_judge_ids": missing,
            },
            raw_file,
            ensure_ascii=False,
            indent=2,
        )

    with score_path.open("x", encoding="utf-8") as f:
        for s in scores:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # consolida com métricas existentes
    metrics_path = output / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as f:
            base = json.load(f)
    else:
        base = {}

    if "judge_scores" in base:
        base["judge_scores_legacy"] = base.pop("judge_scores")
    base["judge_scores"] = {
        "rubric_version": judge_prompt.RUBRIC_VERSION,
        "judge_model": judge_model or "not_declared",
        "imported_count": len(scores),
        "missing_count": len(missing),
        "missing_judge_ids": missing,
        **_aggregate_judge_scores(scores),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(base, f, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"✅ {len(scores)} notas importadas.")
    print(f"   notas: {score_path}")
    print(f"   métricas atualizadas: {metrics_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Harness de avaliação do pipeline RAG")
    parser.add_argument(
        "--eval-set",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "eval_set.jsonl"),
        help="Caminho do eval_set.jsonl",
    )
    parser.add_argument("--qrels", help="Julgamentos de relevância JSON versionados")
    parser.add_argument("--rerank", action="store_true", help="Ativa reranking experimental no modo RAG")
    parser.add_argument(
        "--output-dir", help="Novo diretório para avaliação; existente para --judge"
    )
    parser.add_argument("--judge-model", help="Modelo/versão declarado pelo avaliador manual")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--eval", action="store_true", help="Roda o pipeline (rag/no-rag)")
    group.add_argument("--adaptive-k", action="store_true", help="Mede hit@k/MRR")
    group.add_argument("--judge", action="store_true", help="Reaceita notas do juiz")

    args = parser.parse_args()
    eval_set_path = os.path.abspath(args.eval_set)

    if args.eval:
        run_eval(eval_set_path, args.qrels, args.output_dir, args.rerank)
    elif args.adaptive_k:
        run_adaptive_k(eval_set_path, args.qrels, args.output_dir)
    elif args.judge:
        if not args.output_dir:
            parser.error("--judge requer --output-dir da avaliação a consolidar")
        run_judge(args.output_dir, args.judge_model)


if __name__ == "__main__":
    main()
