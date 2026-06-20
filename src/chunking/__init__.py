"""Chunking utilities shared across the retrieval layer."""

import json
import logging

import config.settings as s
from langchain_core.documents import Document

log = logging.getLogger(__name__)


def load_processed_docs() -> list[dict]:
    """Load all cached extracted documents from s.PROCESSED_DIR.

    Returns a list of dicts with schema:
        {doc_name: str, pages: [{page_num: int, text: str}]}
    """
    docs = []
    for path in sorted(s.PROCESSED_DIR.glob("*.json")):
        try:
            docs.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            log.warning("Skipping %s: %s", path.name, exc)
    if not docs:
        raise RuntimeError(
            f"No processed documents found in {s.PROCESSED_DIR}. "
            "Run scripts/extract_text.py first."
        )
    return docs


def get_chunks_for_strategy(strategy_name: str) -> list[Document]:
    """Return the chunk list for *strategy_name*, matching what was embedded into Chroma.

    For parent_document, returns child chunks (the units stored in Chroma).
    """
    docs = load_processed_docs()

    if strategy_name == "fixed_size":
        from src.chunking.fixed_size import chunk_documents
        return chunk_documents(docs)

    if strategy_name == "recursive":
        from src.chunking.recursive import chunk_documents
        return chunk_documents(docs)

    if strategy_name == "semantic":
        from src.chunking.semantic import chunk_documents
        return chunk_documents(docs)

    if strategy_name == "parent_document":
        from src.chunking.parent_document import chunk_documents
        child_chunks, _ = chunk_documents(docs)
        return child_chunks

    raise ValueError(f"Unknown chunking strategy: {strategy_name!r}")
