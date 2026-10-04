"""Actual sqlite-vec search and artifact export, deterministic mocked model calls."""

import json
import sqlite3

import pytest
import sqlite_vec

from pipeline import config
from pipeline.core import retrieval
from pipeline.eval import dataset, run_eval


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    db_path = tmp_path / "vectors.db"
    db = sqlite3.connect(db_path)
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.execute("CREATE VIRTUAL TABLE questions_vec USING vec0(embedding float[2])")
    db.execute(
        "CREATE TABLE questions_meta (rowid INTEGER PRIMARY KEY, csv_id TEXT, nivel TEXT, "
        "operacao TEXT, questao TEXT, comparacao TEXT, resposta_contextualizada TEXT)"
    )
    for i, (doc, vec) in enumerate([("a", [1.0, 0.0]), ("b", [0.0, 1.0]), ("c", [-1.0, 0.0])], 1):
        db.execute(
            "INSERT INTO questions_vec(rowid, embedding) VALUES (?, ?)",
            (i, sqlite_vec.serialize_float32(vec)),
        )
        db.execute(
            "INSERT INTO questions_meta VALUES (?, ?, ?, ?, ?, ?, ?)",
            (i, doc, "simples", "soma", f"pergunta {doc}", "2", "1+1=2"),
        )
    db.commit()
    db.close()
    monkeypatch.setattr(retrieval, "VECTOR_DB_PATH", str(db_path))
    monkeypatch.setattr(config, "VECTOR_DB_PATH", str(db_path))
    monkeypatch.setattr(retrieval, "_embed", lambda query: [1.0, 0.0])
    monkeypatch.setattr(run_eval, "TOP_K", 2)
    monkeypatch.setattr(run_eval, "TOP_K_RANGE", [1, 2])
    monkeypatch.setattr(run_eval, "_load_reference", lambda expected: "1+1=2")
    model_calls = []

    def generate(query, context):
        model_calls.append(context)
        return "resposta de teste"

    monkeypatch.setattr(run_eval.generation, "generate", generate)
    samples = tmp_path / "eval.jsonl"
    samples.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": q,
                    "questão": "Quanto é 1+1?",
                    "csv_id_esperado": "a",
                    "nível": "simples",
                    "operação": "soma",
                }
            )
            for q in ["q1", "q2"]
        )
    )
    qrels = tmp_path / "qrels.json"
    qrels.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "annotated",
                "version": "1",
                "corpus_id": "test",
                "qrels": {"q1": {"a": 1, "b": 0, "c": 1}},
            }
        )
    )
    return samples, qrels, model_calls


def test_eval_exports_metrics_nulls_and_no_overwrite(experiment, tmp_path):
    samples, qrels, calls = experiment
    output = tmp_path / "eval-output"
    run_eval.run_eval(str(samples), str(qrels), str(output))
    data = json.loads((output / "metrics.json").read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["modes"] == ["rag", "no-rag"]
    assert manifest["inputs"]["evaluation_set"]["sha256"]
    assert data["schema_version"] == 2
    assert data["metrics"]["precision_at_k"] == data["metrics"]["recall_at_k"] == 0.5
    assert data["valid_counts"]["recall_at_k"] == 1
    assert data["status_counts"] == {"ok": 1, "unjudged": 1}
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert len(records) == len(calls) == 4
    assert all(r["retrieval_quality"] is None for r in records if r["mode"] == "no-rag")
    assert all(r["contexto_enviado"] is None for r in records if r["mode"] == "no-rag")
    assert "fonte: a" in records[0]["contexto_enviado"]
    assert calls[1] == calls[3] == []
    assert data["paired_comparison"]["valid_pairs"] == 2
    original = (output / "metrics.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_eval.run_eval(str(samples), str(qrels), str(output))
    assert (output / "metrics.json").read_bytes() == original


def test_adaptive_k_exports_global_and_operation_without_generation(experiment, tmp_path):
    samples, qrels, calls = experiment
    output = tmp_path / "adaptive"
    run_eval.run_adaptive_k(str(samples), str(qrels), str(output))
    data = json.loads((output / "adaptive_k_metrics.json").read_text())
    assert data["by_k"]["1"]["metrics"]["precision_at_k"] == 1
    assert data["by_k"]["1"]["metrics"]["recall_at_k"] == 0.5
    assert data["by_k"]["2"]["metrics"]["precision_at_k"] == 0.5
    assert data["by_operation"]["soma"] == data["by_k"]
    assert calls == []
    default_output = tmp_path / "single-id"
    run_eval.run_adaptive_k(str(samples), output_dir=str(default_output))
    single = json.loads((default_output / "adaptive_k_metrics.json").read_text())
    assert single["by_k"]["2"]["metrics"]["recall_at_k"] == 1
    assert single["by_k"]["2"]["sources"] == ["single_expected_id"]


def test_duplicate_sample_ids_rejected(experiment):
    samples, _, _ = experiment
    first = samples.read_text().splitlines()[0]
    samples.write_text(first + "\n" + first)
    with pytest.raises(ValueError, match="duplicados"):
        dataset.load_eval_set(str(samples))


def test_empty_dataset_exports_no_observations(experiment, tmp_path):
    samples, _, calls = experiment
    samples.write_text("")
    output = tmp_path / "empty"
    run_eval.run_eval(str(samples), output_dir=str(output))
    data = json.loads((output / "metrics.json").read_text())
    assert data["metrics"]["recall_at_k"] is None
    assert data["valid_counts"]["recall_at_k"] == 0
    assert calls == []
