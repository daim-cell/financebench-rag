
import logging

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import config.settings as s

log = logging.getLogger(__name__)

_BATCH_SIZE = 200  # chunks per Chroma add_documents call


def get_embedder() -> HuggingFaceEmbeddings:
    """Return a sentence-transformers embedder on MPS."""
    return HuggingFaceEmbeddings(
        model_name=s.EMBED_MODEL,
        model_kwargs={"device": s.DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


def build_chroma_store(
    chunks: list[Document],
    strategy_name: str,
    embedder: HuggingFaceEmbeddings | None = None,
    force_rebuild: bool = False,
) -> Chroma:
    """
    Embed chunks and persist a Chroma collection to disk.

    If the collection already exists and force_rebuild is False, the function
    loads and returns the existing store without re-embedding.

    Args:
        chunks:         List of Documents with the required metadata schema.
        strategy_name:  Used as the subdirectory name under VECTORSTORE_DIR.
        embedder:       Optional pre-built embedder (avoids re-loading model).
        force_rebuild:  Delete and rebuild even if store already exists.
    """
    persist_dir = s.VECTORSTORE_DIR / strategy_name
    persist_dir_str = str(persist_dir)

    if embedder is None:
        embedder = get_embedder()

    # Load existing store if available
    if persist_dir.exists() and any(persist_dir.iterdir()) and not force_rebuild:
        log.info("Loading existing Chroma store: %s", strategy_name)
        return Chroma(persist_directory=persist_dir_str, embedding_function=embedder)

    # Build from scratch
    if persist_dir.exists() and force_rebuild:
        import shutil
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)
    log.info(
        "Building Chroma store for '%s' (%d chunks)…", strategy_name, len(chunks)
    )

    store = Chroma(persist_directory=persist_dir_str, embedding_function=embedder)

    for i in range(0, len(chunks), _BATCH_SIZE):
        batch = chunks[i : i + _BATCH_SIZE]
        store.add_documents(batch)
        pct = min(i + _BATCH_SIZE, len(chunks))
        log.info("  %d / %d chunks embedded", pct, len(chunks))

    log.info("Chroma store built and persisted → %s", persist_dir)
    return store


def get_dense_retriever(
    strategy_name: str,
    k: int = s.TOP_K_DENSE,
    embedder: HuggingFaceEmbeddings | None = None,
) -> BaseRetriever:
    """
    Load a persisted Chroma store and wrap it as a LangChain retriever.

    Args:
        strategy_name: Subdirectory name under VECTORSTORE_DIR.
        k:             Number of candidates to retrieve.
    """
    if embedder is None:
        embedder = get_embedder()

    persist_dir = str(s.VECTORSTORE_DIR / strategy_name)
    store = Chroma(persist_directory=persist_dir, embedding_function=embedder)
    return store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )