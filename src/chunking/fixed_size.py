"""
Fixed-size chunking: 512 tokens per chunk, 51-token overlap (~10 %).

This is "fixed-size" in the truest sense — TokenTextSplitter slices the text
into uniform token windows with no regard for sentence, paragraph, or section
boundaries. Contrast with recursive.py, which respects document structure.

This naive splitting is intentional: it is the baseline against which the
structure-aware strategies are measured.
"""
import logging

from langchain_core.documents import Document
from langchain_text_splitters import TokenTextSplitter

import config.settings as s

log = logging.getLogger(__name__)


def chunk_documents(docs: list[dict]) -> list[Document]:
    """
    Chunk processed documents into uniform fixed-size token windows.

    Args:
        docs: list of dicts from load_processed_docs(), each with schema
              {doc_name: str, pages: [{page_num: int, text: str}]}

    Returns:
        list of LangChain Documents with metadata:
            {doc_name: str, page_num: int, chunk_id: str}
    """
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base",
        chunk_size=s.CHUNK_SIZE,
        chunk_overlap=s.CHUNK_OVERLAP,
    )

    all_chunks: list[Document] = []
    for doc in docs:
        doc_name = doc["doc_name"]
        for page in doc["pages"]:
            page_num: int = page["page_num"]
            for i, text in enumerate(splitter.split_text(page["text"])):
                all_chunks.append(
                    Document(
                        page_content=text,
                        metadata={
                            "doc_name": doc_name,
                            "page_num": page_num,
                            "chunk_id": f"{doc_name}::p{page_num}::c{i}",
                        },
                    )
                )

    log.info("fixed_size chunking: %d chunks from %d documents", len(all_chunks), len(docs))
    return all_chunks