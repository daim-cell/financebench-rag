"""Graph assembly placeholder for the production FinanceBench QA pipeline."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def build_graph(*, hitl: bool = False, debug: bool = False):
    """Build and compile the FinanceBench LangGraph StateGraph."""
    raise NotImplementedError("LangGraph builder has not been implemented yet.")
