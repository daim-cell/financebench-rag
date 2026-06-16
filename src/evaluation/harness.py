"""Single entry point for scoring a RAG pipeline."""

import csv
import json
import logging
import random
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

import config.settings as s
from src.evaluation.metrics import hit_rate, ndcg_at_k

log = logging.getLogger(__name__)

# ── RAGAS (guarded import — langchain_community 0.4+ removed vertexai) ───────
# Stub the missing module so ragas can import without crashing.
import sys as _sys
import types as _types
if "langchain_community.chat_models.vertexai" not in _sys.modules:
    _stub = _types.ModuleType("langchain_community.chat_models.vertexai")
    _stub.ChatVertexAI = None  # ragas imports this name but never calls it at module level
    _sys.modules["langchain_community.chat_models.vertexai"] = _stub

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas import EvaluationDataset, SingleTurnSample
        from ragas import evaluate as _ragas_evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerCorrectness,
            ContextPrecision,
            ContextRecall,
            Faithfulness,
        )
        from ragas.run_config import RunConfig
    _RAGAS_AVAILABLE = True
except Exception as _ragas_err:
    log.warning("RAGAS unavailable (%s) — RAGAS scores will be 0.0", _ragas_err)
    _RAGAS_AVAILABLE = False

# ── Module-level LLM / embedder (created once per process) ───────────────────
_ollama_llm = OllamaLLM(model=s.OLLAMA_MODEL)
_hf_embedder = HuggingFaceEmbeddings(
    model_name=s.EMBED_MODEL,
    model_kwargs={"device": s.DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)

if _RAGAS_AVAILABLE:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        _ragas_llm = LangchainLLMWrapper(_ollama_llm)
        _ragas_embedder = LangchainEmbeddingsWrapper(_hf_embedder)
    _RAGAS_METRICS = [
        AnswerCorrectness(),
        Faithfulness(),
        ContextRecall(),
        ContextPrecision(),
    ]
    _RAGAS_RUN_CONFIG = RunConfig(timeout=300, max_retries=3)

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
    "answer_correctness",
    "faithfulness",
    "context_recall",
    "context_precision",
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

    Writes eval/results/{experiment_id}.json and appends a row to
    eval/comparison_table.csv before returning. Partial results are written in
    the finally block so no run vanishes silently.
    """
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    hit5_list: list[bool] = []
    hit10_list: list[bool] = []
    ndcg5_list: list[float] = []
    ragas_samples: list = []
    scores: dict = {}
    num_processed: int = 0

    try:
        for i, pair in enumerate(qa_pairs):
            question: str = pair["question"]
            gold_page: int = pair["page_number"]

            log.info("[%d/%d] %s", i + 1, len(qa_pairs), question[:90])

            try:
                answer, retrieved_docs = pipeline_fn(question)
            except Exception as exc:
                log.warning("pipeline_fn failed on pair %d: %s — skipping", i, exc)
                continue

            num_processed += 1
            hit5_list.append(hit_rate(retrieved_docs[:5], gold_page))
            hit10_list.append(hit_rate(retrieved_docs[:10], gold_page))
            ndcg5_list.append(ndcg_at_k(retrieved_docs, gold_page, k=5))

            if _RAGAS_AVAILABLE:
                contexts = [doc.page_content for doc in retrieved_docs[:5]]
                ragas_samples.append(
                    SingleTurnSample(
                        user_input=question,
                        retrieved_contexts=contexts if contexts else ["[no context retrieved]"],
                        response=answer or "",
                        reference=pair["answer"],
                    )
                )

        # Aggregate retrieval metrics
        scores["hit_rate_at_5"] = _mean(hit5_list)
        scores["hit_rate_at_10"] = _mean(hit10_list)
        scores["ndcg_at_5"] = _mean(ndcg5_list)

        # RAGAS batch evaluation
        if _RAGAS_AVAILABLE and ragas_samples:
            dataset = EvaluationDataset(samples=ragas_samples)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                ragas_result = _ragas_evaluate(
                    dataset=dataset,
                    metrics=_RAGAS_METRICS,
                    llm=_ragas_llm,
                    embeddings=_ragas_embedder,
                    run_config=_RAGAS_RUN_CONFIG,
                    raise_exceptions=False,
                    show_progress=True,
                )
            scores["answer_correctness"] = _safe_mean(ragas_result, "answer_correctness")
            scores["faithfulness"] = _safe_mean(ragas_result, "faithfulness")
            scores["context_recall"] = _safe_mean(ragas_result, "context_recall")
            scores["context_precision"] = _safe_mean(ragas_result, "context_precision")
        else:
            for key in ("answer_correctness", "faithfulness", "context_recall", "context_precision"):
                scores[key] = 0.0

        result_payload = {
            "experiment_id": experiment_id,
            "chunking_strategy": metadata.get("chunking_strategy", ""),
            "retriever_type": metadata.get("retriever_type", ""),
            "query_transformer": metadata.get("query_transformer"),
            "use_reranker": metadata.get("use_reranker", False),
            "scores": scores,
            "timestamp": timestamp,
        }
        _write_result_json(experiment_id, result_payload)
        _append_csv_row(experiment_id, metadata, scores, num_processed, timestamp)
        return scores

    finally:
        # Safety net: persist whatever partial scores exist on exception
        if scores:
            try:
                partial_payload = {
                    "experiment_id": experiment_id,
                    "chunking_strategy": metadata.get("chunking_strategy", ""),
                    "retriever_type": metadata.get("retriever_type", ""),
                    "query_transformer": metadata.get("query_transformer"),
                    "use_reranker": metadata.get("use_reranker", False),
                    "scores": scores,
                    "timestamp": timestamp,
                    "_partial": True,
                }
                _write_result_json(experiment_id, partial_payload)
            except Exception as write_exc:
                log.error("Failed to write partial results: %s", write_exc)


# ── Private helpers ───────────────────────────────────────────────────────────

def _mean(lst: list) -> float:
    return sum(lst) / len(lst) if lst else 0.0


def _safe_mean(ragas_result, key: str) -> float:
    """Average ragas_result[key], filtering None and NaN values."""
    try:
        vals = ragas_result[key]
    except Exception:
        return 0.0
    finite = []
    for v in vals:
        if v is None:
            continue
        try:
            import math
            if not math.isnan(float(v)):
                finite.append(float(v))
        except (TypeError, ValueError):
            continue
    return sum(finite) / len(finite) if finite else 0.0


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
        writer.writerow(
            {
                "experiment_id": experiment_id,
                "chunking_strategy": metadata.get("chunking_strategy", ""),
                "retriever_type": metadata.get("retriever_type", ""),
                "query_transformer": metadata.get("query_transformer", ""),
                "use_reranker": metadata.get("use_reranker", False),
                "hit_rate_at_5": round(scores.get("hit_rate_at_5", 0.0), 4),
                "hit_rate_at_10": round(scores.get("hit_rate_at_10", 0.0), 4),
                "ndcg_at_5": round(scores.get("ndcg_at_5", 0.0), 4),
                "answer_correctness": round(scores.get("answer_correctness", 0.0), 4),
                "faithfulness": round(scores.get("faithfulness", 0.0), 4),
                "context_recall": round(scores.get("context_recall", 0.0), 4),
                "context_precision": round(scores.get("context_precision", 0.0), 4),
                "num_pairs": num_pairs,
                "timestamp": timestamp,
            }
        )
    log.info("CSV row appended → %s", csv_path)
