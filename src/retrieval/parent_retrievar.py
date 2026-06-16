
import logging

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

import config.settings as s
from src.chunking.parent_document import load_parent_store
from src.retrieval.dense import get_dense_retriever

log = logging.getLogger(__name__)


class ParentDocumentRetriever(BaseRetriever):
    """Retrieve children, return their unique parents in child-rank order."""

    child_retriever: BaseRetriever = Field(...)
    parent_store: dict = Field(default_factory=dict)
    k: int = Field(default=s.TOP_K_DENSE)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        children = self.child_retriever.invoke(query)

        seen: set[str] = set()
        parents: list[Document] = []
        for child in children:
            parent_id = child.metadata.get("parent_id")
            if not parent_id or parent_id in seen:
                continue
            seen.add(parent_id)

            entry = self.parent_store.get(parent_id)
            if entry is None:
                # Fall back to the child itself if parent is missing
                parents.append(child)
                continue

            parents.append(
                Document(
                    page_content=entry["text"],
                    metadata={
                        "doc_name": entry["doc_name"],
                        "page_num": entry["page_num"],
                        "chunk_id": parent_id,
                    },
                )
            )
            if len(parents) >= self.k:
                break

        return parents


def get_parent_document_retriever(k: int = s.TOP_K_DENSE) -> ParentDocumentRetriever:
    """
    Build the parent-document retriever.

    The child retriever pulls from the 'parent_document' Chroma collection
    (which stores child chunks). We over-fetch children (k * 3) because many
    children collapse to the same parent.
    """
    child_retriever = get_dense_retriever("parent_document", k=k * 3)
    parent_store = load_parent_store()
    return ParentDocumentRetriever(
        child_retriever=child_retriever,
        parent_store=parent_store,
        k=k,
    )