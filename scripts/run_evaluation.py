"""Run the evaluation harness for a single pipeline configuration.

Arguments
---------
  --strategy    Chunking strategy (required): fixed_size | recursive | semantic | parent_document
  --retriever   Retrieval method (default: dense): dense | sparse | hybrid_rrf
  --transformer Query transformation (default: none): hyde | multi_query | step_back
  --reranker    Enable cross-encoder reranking (flag, default: off)
  --k           Number of docs sent to LLM (default: settings.TOP_K_FINAL)
  --pairs       Number of Q&A pairs to evaluate (default: settings.EVAL_SAMPLE_SIZE)
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_core.documents import Document
from langchain_ollama import OllamaLLM

import config.settings as s
from src.evaluation.harness import evaluate, sample_qa_pairs
from src.retrieval.dense import get_dense_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_VALID_STRATEGIES = {"fixed_size", "recursive", "semantic", "parent_document"}
_VALID_RETRIEVERS = {"dense", "sparse", "hybrid_rrf"}
_VALID_TRANSFORMERS = {"hyde", "multi_query", "step_back"}

_PROMPT = (
    "You are a financial analyst. Answer the question below using only the "
    "provided context. Be concise and precise.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:"
)


def build_pipeline(
    strategy: str,
    retriever_type: str,
    transformer: str | None,
    use_reranker: bool,
    k: int,
) -> callable:
    """Return a pipeline_fn(question) -> (answer, docs) for the given config."""

    llm = OllamaLLM(model=s.OLLAMA_MODEL)

    # ── Retriever — always fetch TOP_K_DENSE candidates ──────────────────────
    # k (= TOP_K_FINAL) only controls how many docs the LLM sees; the retriever
    # always returns TOP_K_DENSE so hit_rate_at_10 is meaningful.
    if retriever_type == "dense":
        if strategy == "parent_document":
            from src.retrieval.parent_retrievar import get_parent_document_retriever
            retriever = get_parent_document_retriever(k=s.TOP_K_DENSE)
        else:
            retriever = get_dense_retriever(strategy, k=s.TOP_K_DENSE)
    elif retriever_type == "sparse":
        from src.chunking import get_chunks_for_strategy
        from src.retrieval.sparse import get_sparse_retriever
        retriever = get_sparse_retriever(get_chunks_for_strategy(strategy), k=s.TOP_K_DENSE)
    elif retriever_type == "hybrid_rrf":
        from src.retrieval.hybrid_rrf import get_hybrid_retriever
        retriever = get_hybrid_retriever(strategy, k=s.TOP_K_DENSE)
    else:
        raise ValueError(f"Unknown retriever: {retriever_type}")

    # ── Query transformer (wraps retriever) ───────────────────────────────────
    if transformer == "hyde":
        from src.query_transform.hyde import HyDERetriever
        retriever = HyDERetriever(retriever, llm)
    elif transformer == "multi_query":
        from src.query_transform.multi_query import MultiQueryRetriever
        retriever = MultiQueryRetriever(retriever, llm)
    elif transformer == "step_back":
        from src.query_transform.step_back import StepBackRetriever
        retriever = StepBackRetriever(retriever, llm)

    # ── Reranker (wraps retrieved docs) ──────────────────────────────────────
    reranker = None
    if use_reranker:
        from src.reranking.cross_encoder import rerank
        reranker = rerank

    def pipeline_fn(question: str) -> tuple[str, list[Document]]:
        # Retrieve TOP_K_DENSE candidates
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(question)
        else:
            docs = retriever.invoke(question)

        # Rerank if requested (scores all candidates, trims to TOP_K_DENSE)
        if reranker:
            docs = reranker(question, docs)

        # LLM only sees the top k docs; harness receives the full list so that
        # hit_rate_at_5 and hit_rate_at_10 are computed over 20 candidates.
        llm_docs = docs[:k]
        context = "\n\n".join(d.page_content for d in llm_docs)
        answer = llm.invoke(_PROMPT.format(question=question, context=context))
        return answer, docs

    return pipeline_fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a RAG pipeline configuration against the FinanceBench eval set."
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=sorted(_VALID_STRATEGIES),
        help="Chunking strategy (vector store must already be built)",
    )
    parser.add_argument(
        "--retriever",
        default="dense",
        choices=sorted(_VALID_RETRIEVERS),
        help="Retrieval method (default: dense)",
    )
    parser.add_argument(
        "--transformer",
        default=None,
        choices=sorted(_VALID_TRANSFORMERS),
        help="Query transformation applied before retrieval (default: none)",
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Enable cross-encoder reranking of retrieved docs",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=s.TOP_K_FINAL,
        help=f"Docs sent to the LLM (default: {s.TOP_K_FINAL})",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=s.EVAL_SAMPLE_SIZE,
        help=f"Number of Q&A pairs to evaluate (default: {s.EVAL_SAMPLE_SIZE})",
    )
    args = parser.parse_args()

    # Build experiment_id following the project naming convention:
    # {chunking}_{retriever}_{transformer_or_none}_{rerank_or_base}
    transformer_tag = args.transformer or "none"
    reranker_tag = "rerank" if args.reranker else "base"
    experiment_id = f"{args.strategy}_{args.retriever}_{transformer_tag}_{reranker_tag}"

    log.info("Experiment: %s", experiment_id)
    log.info("  strategy=%s  retriever=%s  transformer=%s  reranker=%s  k=%d  pairs=%d",
             args.strategy, args.retriever, args.transformer, args.reranker, args.k, args.pairs)

    pipeline_fn = build_pipeline(
        strategy=args.strategy,
        retriever_type=args.retriever,
        transformer=args.transformer,
        use_reranker=args.reranker,
        k=args.k,
    )

    qa_pairs = sample_qa_pairs(n=args.pairs)
    log.info("Sampled %d Q&A pairs", len(qa_pairs))

    scores = evaluate(
        pipeline_fn=pipeline_fn,
        qa_pairs=qa_pairs,
        experiment_id=experiment_id,
        metadata={
            "chunking_strategy": args.strategy,
            "retriever_type": args.retriever,
            "query_transformer": args.transformer,
            "use_reranker": args.reranker,
        },
    )

    print("\n── Results: %s ──" % experiment_id)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
