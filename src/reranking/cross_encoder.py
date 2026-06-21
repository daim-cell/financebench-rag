"""Cross-encoder reranking using a sentence-transformers CrossEncoder model.

The cross-encoder scores each (query, passage) pair jointly, producing a
relevance score that is more accurate than the dot-product used in dense
retrieval. It is applied after the initial retrieval step to re-order the
candidate pool before passing the top-k documents to the LLM.
"""

import logging

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

import config.settings as s

log = logging.getLogger(__name__)

# Loaded once per process; CrossEncoder.predict() is thread-safe.
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        log.info("Loading cross-encoder model: %s on %s", s.RERANK_MODEL, s.DEVICE)
        _model = CrossEncoder(s.RERANK_MODEL, device=s.DEVICE)
    return _model


def rerank(query: str, docs: list[Document]) -> list[Document]:
    """Re-order *docs* by cross-encoder relevance score (highest first).

    Args:
        query: The original user question.
        docs:  Candidate documents from the retriever (any count).

    Returns:
        The same documents sorted by descending cross-encoder score.
        Metadata (doc_name, page_num, chunk_id) is preserved unchanged.
    """
    if not docs:
        return docs

    model = _get_model()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    reranked_docs = [doc for _, doc in ranked]

    log.debug(
        "rerank: top score=%.4f  bottom score=%.4f  n=%d",
        float(ranked[0][0]),
        float(ranked[-1][0]),
        len(docs),
    )
    return reranked_docs
