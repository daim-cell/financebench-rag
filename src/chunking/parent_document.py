"""
Parent-document chunking.

Strategy: embed and store SMALL child chunks (256 tokens) for precise matching,
but at retrieval time return the LARGER parent chunk (2048 tokens) so the LLM
gets full surrounding context.

This module:
  1. Splits each page into large parent chunks.
  2. Splits each parent into small child chunks, tagging every child with its
     parent_id.
  3. Persists a parent docstore JSON {parent_id: {text, doc_name, page_num}} to
     disk so the parent-document retriever can swap children → parents later.

The CHILD chunks are what get embedded into Chroma (returned as the first
element of the tuple). The build_vectorstore.py script handles that.

The swap-back logic lives in src/retrieval/parent_document_retriever.py.
"""
import json
import logging

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config.settings as s

log = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")

# Where the parent text is persisted for retrieval-time lookup
PARENT_STORE_PATH = s.VECTORSTORE_DIR / "parent_document" / "parent_docstore.json"


def _token_len(text: str) -> int:
    return len(_ENCODER.encode(text))


def chunk_documents(docs: list[dict]) -> tuple[list[Document], list[Document]]:
    """
    Produce (child_chunks, parent_chunks) and persist the parent docstore.

    Returns:
        child_chunks:  small Documents to embed; each carries metadata
                       {doc_name, page_num, chunk_id, parent_id}.
        parent_chunks: large Documents (returned to the LLM at query time);
                       each carries {doc_name, page_num, parent_id}.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.PARENT_CHUNK,
        chunk_overlap=0,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.CHILD_CHUNK,
        chunk_overlap=int(s.CHILD_CHUNK * 0.1),
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    child_chunks: list[Document] = []
    parent_chunks: list[Document] = []
    parent_store: dict[str, dict] = {}

    for doc in docs:
        doc_name = doc["doc_name"]
        for page in doc["pages"]:
            page_num: int = page["page_num"]

            for p_idx, parent_text in enumerate(parent_splitter.split_text(page["text"])):
                parent_id = f"{doc_name}::p{page_num}::P{p_idx}"

                parent_doc = Document(
                    page_content=parent_text,
                    metadata={
                        "doc_name": doc_name,
                        "page_num": page_num,
                        "parent_id": parent_id,
                    },
                )
                parent_chunks.append(parent_doc)
                parent_store[parent_id] = {
                    "text": parent_text,
                    "doc_name": doc_name,
                    "page_num": page_num,
                }

                # Children inherit the parent_id
                for c_idx, child_text in enumerate(child_splitter.split_text(parent_text)):
                    child_chunks.append(
                        Document(
                            page_content=child_text,
                            metadata={
                                "doc_name": doc_name,
                                "page_num": page_num,
                                "chunk_id": f"{parent_id}::c{c_idx}",
                                "parent_id": parent_id,
                            },
                        )
                    )

    # Persist parent docstore for retrieval-time swap
    PARENT_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PARENT_STORE_PATH.write_text(json.dumps(parent_store, ensure_ascii=False))

    log.info(
        "parent_document chunking: %d children, %d parents from %d docs (docstore → %s)",
        len(child_chunks), len(parent_chunks), len(docs), PARENT_STORE_PATH,
    )
    return child_chunks, parent_chunks


def load_parent_store() -> dict[str, dict]:
    """Load the persisted parent docstore from disk."""
    if not PARENT_STORE_PATH.exists():
        raise FileNotFoundError(
            f"Parent docstore not found at {PARENT_STORE_PATH}. "
            "Build the parent_document vector store first."
        )
    return json.loads(PARENT_STORE_PATH.read_text(encoding="utf-8"))