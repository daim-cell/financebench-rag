"""Step-back query transformation.

Prompts the LLM to identify the broader financial concept or principle behind
the specific question, then retrieves on that abstracted query. This widens
the search to include background sections (definitions, policy descriptions)
that might not contain the exact metric being asked about but provide the
conceptual context needed to answer correctly.
"""

import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever

log = logging.getLogger(__name__)

_STEP_BACK_PROMPT = (
    "You are a financial analyst. Given the specific question below, identify "
    "the broader financial concept, accounting principle, or business topic "
    "it relates to. Express that broader topic as a short search query "
    "(one sentence, no more than 20 words).\n\n"
    "Specific question: {question}\n\n"
    "Broader topic:"
)


class StepBackRetriever:
    """Retrieve using an abstracted 'step-back' query and union with the original."""

    def __init__(self, retriever: BaseRetriever, llm: BaseLLM) -> None:
        self._retriever = retriever
        self._llm = llm

    def get_relevant_documents(self, query: str) -> list[Document]:
        raw = self._llm.invoke(_STEP_BACK_PROMPT.format(question=query))
        abstracted = str(raw).strip()
        log.debug("Step-back abstracted query: %s", abstracted[:120])

        seen: set[str] = set()
        merged: list[Document] = []

        # Retrieve on abstracted query first so broader context ranks higher
        for doc in self._retriever.invoke(abstracted):
            cid = doc.metadata.get("chunk_id") or f"__no_id_{id(doc)}"
            if cid not in seen:
                seen.add(cid)
                merged.append(doc)

        # Union with original query results
        for doc in self._retriever.invoke(query):
            cid = doc.metadata.get("chunk_id") or f"__no_id_{id(doc)}"
            if cid not in seen:
                seen.add(cid)
                merged.append(doc)

        log.debug("Step-back: %d unique docs after union", len(merged))
        return merged

    def invoke(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)
