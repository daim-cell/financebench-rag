"""Node placeholders for the FinanceBench LangGraph QA pipeline."""

from __future__ import annotations

import logging
from typing import Any

from src.graph.state import GraphState

logger = logging.getLogger(__name__)


def router(state: GraphState) -> dict[str, Any]:
    """Classify the question route."""
    raise NotImplementedError("Graph router node has not been implemented yet.")


def memory_inject(state: GraphState) -> dict[str, Any]:
    """Inject long-term memory relevant to the question."""
    raise NotImplementedError("Memory injection node has not been implemented yet.")


def retrieve(state: GraphState) -> dict[str, Any]:
    """Retrieve documents from vectorstore, web search, or direct route."""
    raise NotImplementedError("Retrieve node has not been implemented yet.")


def grade_documents(state: GraphState) -> dict[str, Any]:
    """Grade retrieved documents as relevant, ambiguous, or irrelevant."""
    raise NotImplementedError("Document grading node has not been implemented yet.")


def rewrite_query(state: GraphState) -> dict[str, Any]:
    """Rewrite the question and increment the rewrite counter."""
    raise NotImplementedError("Query rewrite node has not been implemented yet.")


def generate(state: GraphState) -> dict[str, Any]:
    """Generate an answer from graded documents and memory context."""
    raise NotImplementedError("Generate node has not been implemented yet.")


def self_rag_reflect(state: GraphState) -> dict[str, Any]:
    """Reflect on groundedness and usefulness of the generation."""
    raise NotImplementedError("Self-RAG reflection node has not been implemented yet.")


def memory_write(state: GraphState) -> dict[str, Any]:
    """Persist successful QA context to long-term memory."""
    raise NotImplementedError("Memory write node has not been implemented yet.")


def debug_memory(state: GraphState) -> dict[str, Any]:
    """Expose memory context for debug graph runs."""
    logger.debug("Memory context: %s", state.get("memory_context"))
    return {}
