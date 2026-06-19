"""Multi-query transformation.

Rewrites the original question 3 ways via the LLM, retrieves candidates for
each rewrite, then unions and deduplicates the results by chunk_id. Documents
that appear in multiple retrieval lists are kept only once (first-seen wins),
which preserves the highest-ranked occurrence.
"""

import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever

log = logging.getLogger(__name__)

_MULTI_QUERY_PROMPT = (
    "You are a financial research assistant. Given the question below, generate "
    "3 alternative phrasings that capture the same information need. Each "
    "rephrasing should approach the question from a slightly different angle "
    "(e.g., different terminology, level of specificity, or framing).\n\n"
    "Original question: {question}\n\n"
    "Return exactly 3 numbered alternatives, one per line:\n"
    "1.\n2.\n3."
)


def _parse_rewrites(raw: str, original: str) -> list[str]:
    """Extract numbered rewrites; fall back to original if parsing fails."""
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    rewrites: list[str] = []
    for line in lines:
        # Strip leading "1." / "1)" / "- " etc.
        for prefix in ("1.", "2.", "3.", "1)", "2)", "3)", "-"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        if line:
            rewrites.append(line)
    if not rewrites:
        log.warning("Multi-query: could not parse LLM rewrites; using original")
        return [original]
    return rewrites[:3]


class MultiQueryRetriever:
    """Union results from 3 LLM-generated query rewrites, deduplicated by chunk_id."""

    def __init__(self, retriever: BaseRetriever, llm: BaseLLM) -> None:
        self._retriever = retriever
        self._llm = llm

    def get_relevant_documents(self, query: str) -> list[Document]:
        raw = self._llm.invoke(_MULTI_QUERY_PROMPT.format(question=query))
        rewrites = _parse_rewrites(str(raw), query)
        log.debug("Multi-query rewrites: %s", rewrites)

        seen: set[str] = set()
        merged: list[Document] = []

        for rewrite in [query] + rewrites:
            for doc in self._retriever.invoke(rewrite):
                cid = doc.metadata.get("chunk_id") or f"__no_id_{id(doc)}"
                if cid not in seen:
                    seen.add(cid)
                    merged.append(doc)

        log.debug("Multi-query: %d unique docs after union", len(merged))
        return merged

    def invoke(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)
