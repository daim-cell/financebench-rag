"""HyDE (Hypothetical Document Embedding) query transformation.

Instead of embedding the raw question, the LLM generates a plausible answer
passage and that hypothetical passage is embedded for retrieval. The intuition
is that the embedding of a realistic answer sits closer to the relevant
document chunks than the embedding of a short question.
"""

import logging

from langchain_core.documents import Document
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever

import config.settings as s

log = logging.getLogger(__name__)

_HYDE_PROMPT = (
    "You are a financial analyst. Write a concise, factual passage (2-4 sentences) "
    "that would directly answer the following question based on a SEC filing.\n\n"
    "Question: {question}\n\n"
    "Passage:"
)


class HyDERetriever:
    """Wrap a retriever with HyDE: embed a hypothetical answer, not the question."""

    def __init__(self, retriever: BaseRetriever, llm: BaseLLM) -> None:
        self._retriever = retriever
        self._llm = llm

    def get_relevant_documents(self, query: str) -> list[Document]:
        hypothesis = self._llm.invoke(_HYDE_PROMPT.format(question=query))
        log.debug("HyDE hypothesis: %s", str(hypothesis)[:120])
        return self._retriever.invoke(str(hypothesis))

    def invoke(self, query: str) -> list[Document]:
        return self.get_relevant_documents(query)
