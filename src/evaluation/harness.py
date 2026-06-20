"""Single entry point for scoring a RAG pipeline."""

import csv
import json
import logging
import random
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable

import numpy as np
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

import config.settings as s
from src.evaluation.metrics import hit_rate, ndcg_at_k

log = logging.getLogger(__name__)

# ── Embedder (created once per process, used for all semantic metrics) ────────
_hf_embedder = HuggingFaceEmbeddings(
    model_name=s.EMBED_MODEL,
    model_kwargs={"device": s.DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

# ── CSV schema ────────────────────────────────────────────────────────────────
_CSV_COLUMNS = [
    "experiment_id",
    "chunking_strategy",
    "retriever_type",
    "query_transformer",
    "use_reranker",
    "hit_rate_at_5",
    "hit_rate_at_10",
    "ndcg_at_5",
    "answer_similarity",
    "answer_accuracy",
    "context_precision",
    "context_faithfulness",
    "num_pairs",
    "timestamp",
]


# ── Public helpers ────────────────────────────────────────────────────────────

def sample_qa_pairs(
    n: int = s.EVAL_SAMPLE_SIZE,
    seed: int = s.RANDOM_SEED,
) -> list[dict]:
    """Sample *n* Q&A pairs from the open-source JSONL using stratified sampling.

    Groups entries by question_type and samples proportionally so each stratum
    is represented. Returns a fresh sample every call — nothing is written to disk.
    """
    jsonl_path = s.DATA_DIR / "financebench_open_source.jsonl"
    raw: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    groups: dict[str, list[dict]] = defaultdict(list)
    for entry in raw:
        groups[entry.get("question_type", "other")].append(entry)

    rng = random.Random(seed)
    total = len(raw)
    sampled: list[dict] = []
    for _qtype, entries in sorted(groups.items()):
        n_take = round(len(entries) / total * n)
        sampled.extend(rng.sample(entries, min(n_take, len(entries))))

    rng.shuffle(sampled)
    sampled = sampled[:n]

    return [
        {
            "financebench_id": entry["financebench_id"],
            "question": entry["question"],
            "answer": entry["answer"],
            "doc_name": entry["doc_name"],
            "page_number": entry["evidence"][0]["evidence_page_num"],
            "evidence_text": entry["evidence"][0]["evidence_text"],
        }
        for entry in sampled
    ]


# ── Main evaluation entry point ───────────────────────────────────────────────

def evaluate(
    pipeline_fn: Callable[[str], tuple[str, list[Document]]],
    qa_pairs: list[dict],
    experiment_id: str,
    metadata: dict,
) -> dict:
    """Score *pipeline_fn* against *qa_pairs* and persist results.

    Computes retrieval metrics (hit-rate, NDCG) and embedding-based semantic
    metrics (answer_similarity, answer_accuracy, context_precision,
    context_faithfulness) without any LLM calls.

    Writes eval/results/{experiment_id}.json and appends a row to
    eval/comparison_table.csv before returning. Partial results are written in
    the finally block so no run vanishes silently.
    """
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    hit5_list: list[bool] = []
    hit10_list: list[bool] = []
    ndcg5_list: list[float] = []
    ans_sim_list: list[float] = []
    ans_acc_list: list[float] = []
    ctx_prec_list: list[float] = []
    ctx_faith_list: list[float] = []
    scores: dict = {}
    num_processed: int = 0
    _completed: bool = False

    try:
        for i, pair in enumerate(qa_pairs):
            question: str = pair["question"]
            gold_page: int = pair["page_number"]
            reference: str = pair["answer"]

            log.info("[%d/%d] %s", i + 1, len(qa_pairs), question[:90])

            try:
                answer, retrieved_docs = pipeline_fn(question)
            except Exception as exc:
                log.warning("pipeline_fn failed on pair %d: %s — skipping", i, exc)
                continue

            num_processed += 1

            # ── Retrieval metrics ────────────────────────────────────────────
            hit5_list.append(hit_rate(retrieved_docs[:5], gold_page))
            hit10_list.append(hit_rate(retrieved_docs[:10], gold_page))
            ndcg5_list.append(ndcg_at_k(retrieved_docs, gold_page, k=5))

            # ── Embedding-based semantic metrics ─────────────────────────────
            contexts = [doc.page_content for doc in retrieved_docs[:5]]
            sem = _embedding_metrics(question, answer or "", reference, contexts)
            ans_sim_list.append(sem["answer_similarity"])
            ans_acc_list.append(sem["answer_accuracy"])
            ctx_prec_list.append(sem["context_precision"])
            ctx_faith_list.append(sem["context_faithfulness"])

        # ── Aggregate ────────────────────────────────────────────────────────
        scores = {
            "hit_rate_at_5":       _mean(hit5_list),
            "hit_rate_at_10":      _mean(hit10_list),
            "ndcg_at_5":           _mean(ndcg5_list),
            "answer_similarity":   _mean(ans_sim_list),
            "answer_accuracy":     _mean(ans_acc_list),
            "context_precision":   _mean(ctx_prec_list),
            "context_faithfulness":_mean(ctx_faith_list),
        }

        result_payload = {
            "experiment_id": experiment_id,
            "chunking_strategy": metadata.get("chunking_strategy", ""),
            "retriever_type": metadata.get("retriever_type", ""),
            "query_transformer": metadata.get("query_transformer"),
            "use_reranker": metadata.get("use_reranker", False),
            "scores": scores,
            "num_pairs": num_processed,
            "timestamp": timestamp,
        }
        _write_result_json(experiment_id, result_payload)
        _append_csv_row(experiment_id, metadata, scores, num_processed, timestamp)
        _completed = True
        return scores

    finally:
        # Only write partial results when the run failed before completing normally.
        if scores and not _completed:
            try:
                _write_result_json(experiment_id, {
                    "experiment_id": experiment_id,
                    "chunking_strategy": metadata.get("chunking_strategy", ""),
                    "retriever_type": metadata.get("retriever_type", ""),
                    "query_transformer": metadata.get("query_transformer"),
                    "use_reranker": metadata.get("use_reranker", False),
                    "scores": scores,
                    "num_pairs": num_processed,
                    "timestamp": timestamp,
                    "_partial": True,
                })
            except Exception as exc:
                log.error("Failed to write partial results: %s", exc)


# ── Embedding metric computation ──────────────────────────────────────────────

def _embedding_metrics(
    question: str,
    answer: str,
    reference: str,
    contexts: list[str],
) -> dict[str, float]:
    """Compute all semantic metrics for one Q&A pair in a single embed call.

    All embeddings are L2-normalised, so cosine similarity = dot product.

    Metrics
    -------
    answer_similarity   : cosine sim(generated answer, reference answer)
    answer_accuracy     : 1.0 if answer_similarity >= SIMILARITY_THRESHOLD else 0.0
    context_precision   : mean cosine sim(question, each context)
    context_faithfulness: cosine sim(generated answer, mean of context embeddings)
    """
    texts = [question, answer, reference] + (contexts if contexts else [""])
    embeddings = np.array(_hf_embedder.embed_documents(texts))

    q_emb   = embeddings[0]
    a_emb   = embeddings[1]
    r_emb   = embeddings[2]
    ctx_embs = embeddings[3:]

    answer_similarity = float(np.dot(a_emb, r_emb))
    answer_accuracy   = 1.0 if answer_similarity >= s.SIMILARITY_THRESHOLD else 0.0

    ctx_sims        = ctx_embs @ q_emb               # shape (n_ctx,)
    context_precision   = float(np.mean(ctx_sims)) if len(ctx_sims) else 0.0

    mean_ctx_emb        = ctx_embs.mean(axis=0)
    context_faithfulness = float(np.dot(a_emb, mean_ctx_emb)) if len(ctx_embs) else 0.0

    return {
        "answer_similarity":    answer_similarity,
        "answer_accuracy":      answer_accuracy,
        "context_precision":    context_precision,
        "context_faithfulness": context_faithfulness,
    }


# ── Private I/O helpers ───────────────────────────────────────────────────────

def _mean(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def _write_result_json(experiment_id: str, payload: dict) -> None:
    s.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = s.RESULTS_DIR / f"{experiment_id}.json"
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Result written → %s", out_path)


def _append_csv_row(
    experiment_id: str,
    metadata: dict,
    scores: dict,
    num_pairs: int,
    timestamp: str,
) -> None:
    csv_path = s.COMPARISON_TABLE
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "experiment_id":        experiment_id,
            "chunking_strategy":    metadata.get("chunking_strategy", ""),
            "retriever_type":       metadata.get("retriever_type", ""),
            "query_transformer":    metadata.get("query_transformer", ""),
            "use_reranker":         metadata.get("use_reranker", False),
            "hit_rate_at_5":        round(scores.get("hit_rate_at_5", 0.0), 4),
            "hit_rate_at_10":       round(scores.get("hit_rate_at_10", 0.0), 4),
            "ndcg_at_5":            round(scores.get("ndcg_at_5", 0.0), 4),
            "answer_similarity":    round(scores.get("answer_similarity", 0.0), 4),
            "answer_accuracy":      round(scores.get("answer_accuracy", 0.0), 4),
            "context_precision":    round(scores.get("context_precision", 0.0), 4),
            "context_faithfulness": round(scores.get("context_faithfulness", 0.0), 4),
            "num_pairs":            num_pairs,
            "timestamp":            timestamp,
        })
    log.info("CSV row appended → %s", csv_path)
