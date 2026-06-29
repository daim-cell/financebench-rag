"""Conditional edge helpers for the FinanceBench LangGraph QA pipeline."""

from __future__ import annotations

import logging

from config import settings
from src.graph.state import GraphState

logger = logging.getLogger(__name__)


def route_after_router(state: GraphState) -> str:
    """Return the graph branch selected by the router."""
    return state["route"]


def route_after_grading(state: GraphState) -> str:
    """Route to rewrite or generation after document grading."""
    graded_docs = state.get("graded_docs", [])
    all_irrelevant = bool(graded_docs) and all(grade == "irrelevant" for _, grade in graded_docs)
    if all_irrelevant and state.get("rewrite_count", 0) < settings.MAX_RETRIES:
        return "rewrite_query"
    return "generate"


def route_after_reflection(state: GraphState) -> str:
    """Route to rewrite or memory write after Self-RAG reflection."""
    raise NotImplementedError("Self-RAG reflection routing has not been implemented yet.")
