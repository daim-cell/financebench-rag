import logging
from collections import defaultdict

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

import config.settings as s
from src.retrieval.dense import get_dense_retriever
from src.retrieval.sparse import get_sparse_retriever

log = logging.getLogger(__name__)


class HybridRRFRetriever(BaseRetriever):
    """Fuse dense and sparse results via Reciprocal Rank Fusion."""

    dense_retriever: BaseRetriever = Field(...)
    sparse_retriever: BaseRetriever = Field(...)
    k: int = Field(default=s.TOP_K_DENSE)
    rrf_k: int = Field(default=s.RRF_K)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        dense_docs = self.dense_retriever.invoke(query)
        sparse_docs = self.sparse_retriever.invoke(query)

        # Deduplicate by (doc_name, page_num) rather than chunk_id because the
        # dense and sparse legs use different chunk granularities (parent chunks
        # vs page-level docs) and can return the same page with different IDs.
        # The dense doc wins on tie since it's processed first.
        doc_by_page: dict[tuple, Document] = {}
        rrf_scores: dict[tuple, float] = defaultdict(float)

        for ranked_list in (dense_docs, sparse_docs):
            for rank, doc in enumerate(ranked_list):
                page_key = (
                    doc.metadata.get("doc_name", ""),
                    doc.metadata.get("page_num", -1),
                )
                if page_key not in doc_by_page:
                    doc_by_page[page_key] = doc
                rrf_scores[page_key] += 1.0 / (self.rrf_k + rank + 1)

        fused = sorted(doc_by_page.keys(), key=lambda pk: rrf_scores[pk], reverse=True)
        results = [doc_by_page[pk] for pk in fused[: self.k]]
        log.debug(
            "hybrid_rrf: dense=%d sparse=%d → unique_pages=%d (top %d)",
            len(dense_docs), len(sparse_docs), len(fused), self.k,
        )
        return results


def get_hybrid_retriever(
    strategy_name: str,
    k: int = s.TOP_K_DENSE,
) -> HybridRRFRetriever:
    """
    Build a hybrid RRF retriever for the given chunking strategy.

    Both sub-retrievers fetch TOP_K_DENSE candidates independently; the fused
    list is then trimmed to k after scoring.

    Args:
        strategy_name: Chroma collection to load for the dense leg.
        k:             Final number of documents returned after fusion.
    """
    from src.chunking import get_chunks_for_strategy

    if strategy_name == "parent_document":
        from src.retrieval.parent_retrievar import get_parent_document_retriever
        dense = get_parent_document_retriever(k=s.TOP_K_DENSE)
    else:
        dense = get_dense_retriever(strategy_name, k=s.TOP_K_DENSE)

    chunks = get_chunks_for_strategy(strategy_name)
    sparse = get_sparse_retriever(chunks, k=s.TOP_K_DENSE)
    log.info("hybrid_rrf: built retriever for '%s', k=%d", strategy_name, k)
    return HybridRRFRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        k=k,
        rrf_k=s.RRF_K,
    )
