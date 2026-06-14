# Codex Instructions

## Project

This project is a RAG benchmarking system for the FinanceBench dataset. It evaluates combinations of:

- chunking strategy
- retrieval method
- query transformation
- cross-encoder reranking

The target output is `eval/comparison_table.csv`, with one row per experiment combination scored on answer correctness, faithfulness, retrieval hit-rate, and NDCG@5.

The locked evaluation set is `eval/qa_pairs_30.json`. After it is created, do not modify it unless the user explicitly asks.

## Hardware Constraints

This project is intended for an M1 MacBook Air.

- Always pass `device=settings.DEVICE` to `SentenceTransformer()` and `CrossEncoder()` calls.
- Use `faiss-cpu` only. Do not add or install `faiss-gpu`.
- If MPS hits an unsupported operation, set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the environment. Do not change project code to use CPU as the default device.

## Configuration

All constants belong in `config/settings.py`. Do not hardcode paths, model names, chunk sizes, retrieval limits, or evaluation output locations in module files.

## Required Metadata

Every LangChain `Document` chunk must preserve this metadata schema through chunking, retrieval, query transformation, reranking, and evaluation:

```python
{
    "doc_name": str,
    "page_num": int,
    "chunk_id": str,
}
```

Use `chunk_id = f"{doc_name}::p{page_num}::c{i}"` unless a module has a documented reason to derive it differently.

## Implementation Contracts

Chunkers expose a module-level function:

```python
def chunk_documents(docs: list[dict]) -> list[Document]:
    ...
```

`src/chunking/parent_document.py` is the exception. It returns `(child_docs, parent_docs)` and manages its own Chroma collection.

Retrievers subclass `BaseRetriever` and implement:

```python
def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
    ...
```

Query transformers wrap a retriever but are not `BaseRetriever` subclasses:

```python
class SomeTransformer:
    def __init__(self, retriever: BaseRetriever, llm: BaseLLM) -> None:
        ...

    def get_relevant_documents(self, query: str) -> list[Document]:
        ...
```

## Data Handling

- Use PyMuPDF (`import fitz`) for SEC filing PDF extraction.
- `scripts/download_pdfs.py` and `scripts/extract_text.py` should be idempotent.
- `data/raw/pdfs/`, `data/processed/`, and `vectorstores/` contain large or generated artifacts and should stay out of git except for `.gitkeep` files.

## Evaluation

Experiment IDs follow:

```text
{chunking}_{retriever}_{transformer_or_none}_{rerank_or_base}
```

Examples:

- `fixed_dense_none_base`
- `recursive_hybrid_hyde_rerank`

Raw run outputs go in `eval/results/{experiment_id}.json`. The comparison table is `eval/comparison_table.csv`.
