"""Composable RAG pipeline used by experiment-grid runs."""

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever

import config.settings as s

log = logging.getLogger(__name__)

_RAG_PROMPT = (
    "You are a financial analyst assistant. Use only the provided context excerpts "
    "from SEC filings to answer the question. If the answer is not present in the "
    "context, respond with: 'I cannot find this information in the provided context.'\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class RAGPipeline:
    """
    Composable retrieval-augmented generation pipeline.

    Constructs one retrieval chain per configuration axis and exposes a
    single callable interface so the harness can treat every combination
    identically.
    """

    def __init__(
        self,
        chunking_strategy: str,
        retriever_type: str,
        query_transformer: str | None,
        use_reranker: bool,
        llm: BaseLLM,
        embedder: Embeddings,
    ) -> None:
        self._llm = llm
        self._use_reranker = use_reranker
        self._chunking_strategy = chunking_strategy
        self._retriever_type = retriever_type
        self._query_transformer_name = query_transformer

        base_retriever = self._build_base_retriever(chunking_strategy, retriever_type, embedder)
        self._retriever: Any = self._wrap_transformer(base_retriever, query_transformer, llm)

        log.info(
            "RAGPipeline ready: chunking=%s  retrieval=%s  transform=%s  reranker=%s",
            chunking_strategy, retriever_type, query_transformer, use_reranker,
        )

    # ── Construction helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_base_retriever(
        chunking_strategy: str,
        retriever_type: str,
        embedder: Embeddings,
    ) -> BaseRetriever:
        if retriever_type == "dense":
            from src.retrieval.dense import get_dense_retriever
            if chunking_strategy == "parent_document":
                from src.retrieval.parent_retrievar import get_parent_document_retriever
                return get_parent_document_retriever(k=s.TOP_K_DENSE)
            return get_dense_retriever(chunking_strategy, k=s.TOP_K_DENSE, embedder=embedder)

        if retriever_type == "sparse":
            from src.chunking import get_chunks_for_strategy
            from src.retrieval.sparse import get_sparse_retriever
            chunks = get_chunks_for_strategy(chunking_strategy)
            return get_sparse_retriever(chunks, k=s.TOP_K_DENSE)

        if retriever_type == "hybrid_rrf":
            from src.retrieval.hybrid_rrf import get_hybrid_retriever
            return get_hybrid_retriever(chunking_strategy, k=s.TOP_K_DENSE)

        raise ValueError(f"Unknown retriever_type: {retriever_type!r}")

    @staticmethod
    def _wrap_transformer(
        retriever: BaseRetriever,
        query_transformer: str | None,
        llm: BaseLLM,
    ) -> Any:
        if query_transformer is None:
            return retriever
        if query_transformer == "hyde":
            from src.query_transform.hyde import HyDERetriever
            return HyDERetriever(retriever, llm)
        if query_transformer == "multi_query":
            from src.query_transform.multi_query import MultiQueryRetriever
            return MultiQueryRetriever(retriever, llm)
        if query_transformer == "step_back":
            from src.query_transform.step_back import StepBackRetriever
            return StepBackRetriever(retriever, llm)
        raise ValueError(f"Unknown query_transformer: {query_transformer!r}")

    # ── Callable interface ────────────────────────────────────────────────────

    def __call__(self, query: str) -> tuple[str, list[Document]]:
        """Run the full pipeline: retrieve → (rerank) → generate.

        Returns:
            (answer_str, all_retrieved_docs_before_final_trim)
        """
        # Retrieve candidates
        if hasattr(self._retriever, "invoke"):
            docs: list[Document] = self._retriever.invoke(query)
        else:
            docs = self._retriever.get_relevant_documents(query)

        # Cross-encoder reranking over the full candidate pool
        if self._use_reranker:
            from src.reranking.cross_encoder import rerank
            docs = rerank(query, docs)

        # Build context from the final top-k chunks
        context_docs = docs[: s.TOP_K_FINAL]
        context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)

        prompt = _RAG_PROMPT.format(context=context, question=query)
        answer = str(self._llm.invoke(prompt)).strip()

        return answer, docs
