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

        # Map chunk_id → Document (first seen wins)
        doc_by_id: dict[str, Document] = {}
        rrf_scores: dict[str, float] = defaultdict(float)

        for ranked_list in (dense_docs, sparse_docs):
            for rank, doc in enumerate(ranked_list):
                cid = doc.metadata.get("chunk_id") or f"__no_id_{id(doc)}"
                if cid not in doc_by_id:
                    doc_by_id[cid] = doc
                rrf_scores[cid] += 1.0 / (self.rrf_k + rank + 1)

        fused = sorted(doc_by_id.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
        results = [doc_by_id[cid] for cid in fused[: self.k]]
        log.debug(
            "hybrid_rrf: dense=%d sparse=%d → fused=%d (top %d)",
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
    if strategy_name == "parent_document":
        from src.retrieval.parent_retrievar import get_parent_document_retriever
        dense = get_parent_document_retriever(k=s.TOP_K_DENSE)
    else:
        dense = get_dense_retriever(strategy_name, k=s.TOP_K_DENSE)
    sparse = get_sparse_retriever(strategy_name, k=s.TOP_K_DENSE)
    log.info("hybrid_rrf: built retriever for '%s', k=%d", strategy_name, k)
    return HybridRRFRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        k=k,
        rrf_k=s.RRF_K,
    )
