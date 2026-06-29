"""State schema for the FinanceBench QA graph."""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    """Shared state passed between LangGraph nodes."""

    question: str
    generation: str | None
    documents: list[Document]
    graded_docs: list[tuple[Document, str]]
    memory_context: str | None
    rewrite_count: int
    route: Literal["vectorstore", "websearch", "direct"]
