# This does not work cannot amke this work too
import logging

from langchain_core.documents import Document

import config.settings as s
from src.retrieval.dense import get_embedder

log = logging.getLogger(__name__)


def chunk_documents(docs: list[dict], embedder=None) -> list[Document]:
    """
    Chunk processed documents on semantic breakpoints.

    A separate SemanticChunker pass is run per page so that page_num metadata
    stays accurate (the chunker would otherwise merge text across page bounds).

    Args:
        docs:     list of dicts loaded from load_processed_docs().
        embedder: optional pre-built embedder; one is created if not supplied.

    Returns:
        list of Documents with metadata {doc_name, page_num, chunk_id}.
    """
    try:
        from langchain_experimental.text_splitter import SemanticChunker
    except ImportError as exc:
        raise ImportError(
            "SemanticChunker requires langchain-experimental. "
            "Run: pip install langchain-experimental"
        ) from exc

    if embedder is None:
        embedder = get_embedder()
    chunker = SemanticChunker(
        embeddings=embedder,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=s.SEMANTIC_THRESHOLD,
    )

    all_chunks: list[Document] = []
    for d_idx, doc in enumerate(docs, start=1):
        doc_name = doc["doc_name"]
        for page in doc["pages"]:
            page_num: int = page["page_num"]
            text = page["text"]
            # Very short pages can't be meaningfully split — keep whole
            try:
                pieces = chunker.split_text(text)
            except Exception:
                pieces = [text]

            for i, piece in enumerate(pieces):
                if not piece.strip():
                    continue
                all_chunks.append(
                    Document(
                        page_content=piece,
                        metadata={
                            "doc_name": doc_name,
                            "page_num": page_num,
                            "chunk_id": f"{doc_name}::p{page_num}::c{i}",
                        },
                    )
                )
        if d_idx % 10 == 0:
            log.info("  semantic chunking: %d / %d docs processed", d_idx, len(docs))

    log.info("semantic chunking: %d chunks from %d documents", len(all_chunks), len(docs))
    return all_chunks
