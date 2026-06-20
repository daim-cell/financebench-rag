"""BM25-based sparse retrieval.

Rebuilt from the SAME chunk units as the paired dense retriever — this is
critical for RRF fusion to be meaningful. Fusing ranks from two retrievers
operating on different-granularity units (e.g. whole pages vs. 512-token
chunks) produces a degenerate, non-comparable fusion. BM25 is never
persisted to disk; it is cheap to rebuild and chunk lists are already
cached in memory by the pipeline that calls this.
"""

import logging

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

import config.settings as s

log = logging.getLogger(__name__)


class SparseBM25Retriever(BaseRetriever):
    """Thin wrapper around BM25Retriever that conforms to BaseRetriever."""

    _bm25: BM25Retriever = None

    bm25_k: int = Field(default=s.TOP_K_DENSE)

    def __init__(self, bm25: BM25Retriever, k: int = s.TOP_K_DENSE, **data):
        super().__init__(bm25_k=k, **data)
        object.__setattr__(self, "_bm25", bm25)
        # Over-fetch by 3× so dedup leaves enough unique pages after filtering.
        self._bm25.k = k * 3

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        raw = self._bm25.invoke(query)

        # Deduplicate by (doc_name, page_num): multiple chunks from the same
        # page all have the same page_num — keeping duplicates wastes top-k slots.
        seen: set[tuple] = set()
        unique: list[Document] = []
        for doc in raw:
            page_key = (doc.metadata.get("doc_name", ""), doc.metadata.get("page_num", -1))
            if page_key not in seen:
                seen.add(page_key)
                unique.append(doc)
            if len(unique) == self.bm25_k:
                break

        return unique


def get_sparse_retriever(
    chunks: list[Document],
    k: int = s.TOP_K_DENSE,
) -> SparseBM25Retriever:
    """
    Build an in-memory BM25 retriever over the SAME chunks used for dense retrieval.

    Args:
        chunks: The chunk list produced by a chunking strategy's
                chunk_documents() — must be the identical list embedded into
                the corresponding Chroma collection so RRF fuses comparable units.
        k:      Number of documents to return per query.

    Example:
        from src.chunking.fixed_size import chunk_documents
        from src.ingestion.pdf_extractor import load_processed_docs

        docs = load_processed_docs()
        chunks = chunk_documents(docs)
        sparse_retriever = get_sparse_retriever(chunks, k=20)
        dense_retriever  = get_dense_retriever("fixed_size", k=20)
        # Both now rank the exact same 512-token chunk units.
    """
    if not chunks:
        raise ValueError(
            "get_sparse_retriever received an empty chunk list. "
            "Pass the output of a chunk_documents() call, not raw page text."
        )

    bm25 = BM25Retriever.from_documents(chunks, k=k)
    log.info("sparse: BM25 index built with %d chunks, k=%d", len(chunks), k)
    return SparseBM25Retriever(bm25=bm25, k=k)