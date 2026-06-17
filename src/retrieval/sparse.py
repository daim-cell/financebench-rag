import logging

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

import config.settings as s

log = logging.getLogger(__name__)


def _load_chunks_from_processed(strategy_name: str) -> list[Document]:
    """
    Reconstruct Document objects from the processed JSON files.

    We cannot re-use the Chroma store here because langchain-chroma does not
    expose a bulk "get all documents" method that is fast enough for BM25
    index construction. Reading the processed JSONs is cheaper and idempotent.
    """
    import json

    processed_dir = s.PROCESSED_DIR
    chunks: list[Document] = []

    for path in sorted(processed_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Skipping %s: %s", path.name, exc)
            continue

        doc_name: str = data.get("doc_name", path.stem)
        for page in data.get("pages", []):
            page_num: int = page.get("page_num", 0)
            text: str = page.get("text", "").strip()
            if not text:
                continue
            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "doc_name": doc_name,
                        "page_num": page_num,
                        "chunk_id": f"{doc_name}::p{page_num}::bm25",
                    },
                )
            )

    log.info("sparse: loaded %d page-level documents for BM25 index", len(chunks))
    return chunks


class SparseBM25Retriever(BaseRetriever):
    """Thin wrapper around BM25Retriever that conforms to BaseRetriever."""

    _bm25: BM25Retriever = None  # set in __init__; excluded from pydantic schema

    bm25_k: int = Field(default=s.TOP_K_DENSE)

    def __init__(self, bm25: BM25Retriever, k: int = s.TOP_K_DENSE, **data):
        super().__init__(bm25_k=k, **data)
        object.__setattr__(self, "_bm25", bm25)
        self._bm25.k = k

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        return self._bm25.invoke(query)


def get_sparse_retriever(
    strategy_name: str,
    k: int = s.TOP_K_DENSE,
) -> SparseBM25Retriever:
    """
    Build an in-memory BM25 retriever over chunks from the processed JSON files.

    The index is rebuilt from scratch on every call; BM25 is never persisted.

    Args:
        strategy_name: Ignored for index construction (all processed docs are
                       indexed); kept for API symmetry with get_dense_retriever.
        k:             Number of documents to return per query.
    """
    chunks = _load_chunks_from_processed(strategy_name)
    if not chunks:
        raise RuntimeError(
            f"No processed documents found in {s.PROCESSED_DIR}. "
            "Run scripts/extract_text.py first."
        )

    bm25 = BM25Retriever.from_documents(chunks, k=k)
    log.info("sparse: BM25 index built with %d documents, k=%d", len(chunks), k)
    return SparseBM25Retriever(bm25=bm25, k=k)
