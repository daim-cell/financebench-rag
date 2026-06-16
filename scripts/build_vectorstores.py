import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config.settings as s
from extract_text import load_processed_docs
from src.retrieval.dense import get_embedder, build_chroma_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)

AVAILABLE = ["fixed_size", "recursive", "semantic", "parent_document"]


def _get_chunks(strategy: str, docs: list[dict]):
    if strategy == "fixed_size":
        from src.chunking.fixed_size import chunk_documents
        return chunk_documents(docs)
    if strategy == "recursive":
        from src.chunking.recursive import chunk_documents
        return chunk_documents(docs)
    if strategy == "semantic":
        from src.chunking.semantic import chunk_documents
        return chunk_documents(docs)
    if strategy == "parent_document":
        from src.chunking.parent_document import chunk_documents
        return chunk_documents(docs)  # returns (child_chunks, parent_chunks)
    raise ValueError(f"Unknown strategy: {strategy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma vector store")
    parser.add_argument(
        "--strategy",
        default="fixed_size",
        choices=AVAILABLE,
        help="Chunking strategy to use (default: fixed_size)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild even if store already exists",
    )
    args = parser.parse_args()

    if not s.PROCESSED_DIR.exists() or not any(s.PROCESSED_DIR.glob("*.json")):
        log.error("No processed documents found in %s. Run extract_text.py first.", s.PROCESSED_DIR)
        sys.exit(1)

    log.info("Loading processed documents…")
    docs = load_processed_docs()
    log.info("Loaded %d documents", len(docs))

    log.info("Chunking with strategy: %s", args.strategy)
    result = _get_chunks(args.strategy, docs)

    # parent_document returns a tuple
    if isinstance(result, tuple):
        child_chunks, _ = result
        chunks = child_chunks
    else:
        chunks = result

    log.info("Created %d chunks", len(chunks))

    log.info("Loading embedder (%s on %s)…", s.EMBED_MODEL, s.DEVICE)
    embedder = get_embedder()

    build_chroma_store(
        chunks,
        strategy_name=args.strategy,
        embedder=embedder,
        force_rebuild=args.force,
    )
    log.info("Done.")


if __name__ == "__main__":
    main()