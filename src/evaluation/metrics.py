"""Retrieval metrics for FinanceBench evaluation."""

import logging
import math

from langchain_core.documents import Document

log = logging.getLogger(__name__)


def hit_rate(retrieved_docs: list[Document], gold_page_num: int) -> bool:
    """True if gold_page_num appears in any retrieved doc's metadata."""
    return any(doc.metadata.get("page_num") == gold_page_num for doc in retrieved_docs)


def ndcg_at_k(retrieved_docs: list[Document], gold_page_num: int, k: int = 5) -> float:
    """Binary-relevance NDCG@k. IDCG = 1.0 (single relevant doc at rank 1)."""
    dcg = sum(
        (1.0 if doc.metadata.get("page_num") == gold_page_num else 0.0) / math.log2(rank + 2)
        for rank, doc in enumerate(retrieved_docs[:k])
    )
    return dcg  # NDCG = DCG / IDCG = DCG / 1.0
