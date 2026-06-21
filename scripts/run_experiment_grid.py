"""
Experiment grid runner for FinanceBench RAG benchmarking.

Full factorial grid (72 runs):
  Chunking strategy   : fixed_size | recursive | parent_document  (3)
  Retrieval method    : dense | sparse | hybrid_rrf               (3)
  Query transform     : None | hyde | multi_query | step_back     (4)
  Cross-encoder rerank: False | True                              (2)
  ──────────────────────────────────────────────────────────────────────────
  3 × 3 × 4 × 2 = 72 runs total

Usage:
    python scripts/run_experiment_grid.py [--force] [--chunking STRATEGY] [--retrieval METHOD] [--transform NAME]
"""

import argparse
import logging
import sys
import time
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config.settings as s
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from src.evaluation.harness import evaluate, sample_qa_pairs
from src.pipeline.rag_pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(s.EVAL_DIR / "grid_run.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)

# ── Experiment axes ───────────────────────────────────────────────────────────

CHUNKING_STRATEGIES = ["fixed_size", "recursive", "parent_document"]
RETRIEVAL_TYPES     = ["dense", "sparse", "hybrid_rrf"]
QUERY_TRANSFORMS    = [None, "hyde", "multi_query", "step_back"]
USE_RERANKER        = [False, True]


def build_grid() -> list[dict]:
    """Return the full factorial grid (3 × 3 × 4 × 2 = 72 experiments)."""
    configs: list[dict] = []
    for chunking, retrieval, transform, reranker in product(
        CHUNKING_STRATEGIES, RETRIEVAL_TYPES, QUERY_TRANSFORMS, USE_RERANKER
    ):
        configs.append(
            {
                "chunking_strategy": chunking,
                "retriever_type":    retrieval,
                "query_transformer": transform,
                "use_reranker":      reranker,
            }
        )
    return configs


def make_experiment_id(cfg: dict) -> str:
    """Derive the canonical experiment ID from a config dict."""
    transform_tag = cfg["query_transformer"] or "none"
    rerank_tag    = "rerank" if cfg["use_reranker"] else "base"
    return (
        f"{cfg['chunking_strategy']}"
        f"_{cfg['retriever_type']}"
        f"_{transform_tag}"
        f"_{rerank_tag}"
    )


def _already_done(experiment_id: str) -> bool:
    result_path = s.RESULTS_DIR / f"{experiment_id}.json"
    return result_path.exists()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full RAG experiment grid")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run experiments that already have a result file",
    )
    parser.add_argument(
        "--chunking",
        choices=CHUNKING_STRATEGIES + ["all"],
        default=None,
        help="Restrict grid to one chunking strategy",
    )
    parser.add_argument(
        "--retrieval",
        choices=RETRIEVAL_TYPES,
        default=None,
        help="Restrict grid to one retrieval method",
    )
    parser.add_argument(
        "--transform",
        choices=["hyde", "multi_query", "step_back"],
        default=None,
        help="Restrict grid to experiments that use this query transformer",
    )
    args = parser.parse_args()

    # ── Build and optionally filter the grid ─────────────────────────────────
    grid = build_grid()

    if args.chunking:
        grid = [c for c in grid if c["chunking_strategy"] == args.chunking]
    if args.retrieval:
        grid = [c for c in grid if c["retriever_type"] == args.retrieval]
    if args.transform:
        grid = [c for c in grid if c["query_transformer"] == args.transform]

    total = len(grid)
    log.info("Grid: %d experiments to run (seed=%d, eval_pairs=%d)",
             total, s.RANDOM_SEED, s.EVAL_SAMPLE_SIZE)

    # ── Shared resources (load once, reuse across all experiments) ───────────
    log.info("Loading LLM: %s", s.OLLAMA_MODEL)
    llm = OllamaLLM(model=s.OLLAMA_MODEL)

    log.info("Loading embedder: %s on %s", s.EMBED_MODEL, s.DEVICE)
    embedder = HuggingFaceEmbeddings(
        model_name=s.EMBED_MODEL,
        model_kwargs={"device": s.DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    # Same QA pairs for every run — seed is fixed in settings
    log.info("Sampling %d QA pairs with seed=%d", s.EVAL_SAMPLE_SIZE, s.RANDOM_SEED)
    qa_pairs = sample_qa_pairs(n=s.EVAL_SAMPLE_SIZE, seed=s.RANDOM_SEED)
    log.info("QA pairs loaded: %d", len(qa_pairs))

    s.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Run grid ──────────────────────────────────────────────────────────────
    completed = 0
    skipped   = 0
    failed    = 0

    for idx, cfg in enumerate(grid, start=1):
        experiment_id = make_experiment_id(cfg)
        prefix = f"[{idx:02d}/{total}]  {experiment_id}"

        if not args.force and _already_done(experiment_id):
            log.info("%s  → SKIP (result exists)", prefix)
            skipped += 1
            continue

        log.info("%s  → starting", prefix)
        t0 = time.perf_counter()

        try:
            pipeline = RAGPipeline(
                chunking_strategy=cfg["chunking_strategy"],
                retriever_type=cfg["retriever_type"],
                query_transformer=cfg["query_transformer"],
                use_reranker=cfg["use_reranker"],
                llm=llm,
                embedder=embedder,
            )

            scores = evaluate(
                pipeline_fn=pipeline,
                qa_pairs=qa_pairs,
                experiment_id=experiment_id,
                metadata=cfg,
            )

            elapsed = time.perf_counter() - t0
            log.info(
                "%s  → DONE in %.1fs  hit@5=%.3f  ndcg@5=%.3f  ans_sim=%.3f",
                prefix,
                elapsed,
                scores.get("hit_rate_at_5", 0.0),
                scores.get("ndcg_at_5", 0.0),
                scores.get("answer_similarity", 0.0),
            )
            completed += 1

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            log.error("%s  → FAILED in %.1fs: %s", prefix, elapsed, exc, exc_info=True)
            failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info(
        "Grid complete — completed=%d  skipped=%d  failed=%d  total=%d",
        completed, skipped, failed, total,
    )
    log.info("Results → %s", s.RESULTS_DIR)
    log.info("Comparison table → %s", s.COMPARISON_TABLE)


if __name__ == "__main__":
    main()
