
import logging

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitter import RecursiveCharacterTextSplitter

import config.settings as s

log = logging.getLogger(__name__)

_ENCODER = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_ENCODER.encode(text))


def chunk_documents(docs: list[dict]) -> list[Document]:
    """
    Chunk processed documents, respecting natural text boundaries.

    Args:
        docs: list of dicts from load_processed_docs().

    Returns:
        list of Documents with metadata {doc_name, page_num, chunk_id}.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.CHUNK_SIZE,
        chunk_overlap=s.CHUNK_OVERLAP,
        length_function=_token_len,
        # Order matters: try paragraph → line → sentence → word → char
        separators=["\n\n", "\n", ". ", " ", ""],
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

    log.info("recursive chunking: %d chunks from %d documents", len(all_chunks), len(docs))
    return all_chunks