"""Long-term semantic memory store placeholder."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    """Stored QA memory item."""

    question: str
    answer: str
    documents: list[Document]


class LongTermMemoryStore:
    """Cross-session memory store keyed by semantic similarity."""

    def search(self, question: str, *, k: int = 5) -> list[MemoryRecord]:
        """Return memory records relevant to a question."""
        raise NotImplementedError("Long-term memory search has not been implemented yet.")

    def add(self, question: str, answer: str, documents: list[Document]) -> None:
        """Persist a successful question, answer, and supporting documents."""
        raise NotImplementedError("Long-term memory writes have not been implemented yet.")
