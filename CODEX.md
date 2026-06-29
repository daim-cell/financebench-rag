# Codex Instructions

## Project

This project is a RAG benchmarking and agentic QA system built on FinanceBench.

It has two layers:

- Evaluation layer: benchmarks chunking, retrieval, query transformation, and cross-encoder reranking combinations, then records results in `eval/comparison_table.csv`.
- Production layer: a LangGraph `StateGraph` that implements CRAG, Self-RAG reflection, adaptive routing, persistent memory, and human-in-the-loop streaming over the same FinanceBench corpus.

The locked evaluation set is `eval/qa_pairs_30.json`. Do not modify it after creation unless the user explicitly asks.

## Hardware Constraints

This project targets an M1 MacBook Air.

- Always pass `device=settings.DEVICE` to `SentenceTransformer()` and `CrossEncoder()` calls.
- Use `faiss-cpu` only. Do not add or install `faiss-gpu`.
- If MPS hits an unsupported operation, set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the environment. Do not change project code to default to CPU.
- Ollama uses Metal automatically and must be running with `ollama serve` before LLM, graph, or RAGAS calls.

## Configuration

All constants belong in `config/settings.py`. Do not hardcode paths, model names, chunk sizes, retrieval limits, graph retry caps, API keys, dataset IDs, or evaluation paths in module files.

## Stack

- LLM: `langchain-ollama` `OllamaLLM`, model from `settings.OLLAMA_MODEL`
- Embedder: `sentence-transformers`, always using `device=settings.DEVICE`
- Agent graph: `langgraph` `StateGraph`
- Short-term memory: `langgraph` `MemorySaver`
- Web search fallback: `tavily-python`
- Vector store: `chromadb` and `langchain-chroma`
- Sparse search: `rank-bm25` through `BM25Retriever`
- Reranker: `sentence-transformers` `CrossEncoder`, model from `settings.RERANK_MODEL`
- Evaluation: `ragas`
- PDF extraction: PyMuPDF (`import fitz`)

## Required Metadata

Every LangChain `Document` chunk must preserve this metadata through chunking, retrieval, query transformation, reranking, graph nodes, and evaluation:

```python
{
    "doc_name": str,
    "page_num": int,
    "chunk_id": str,
}
```

Metadata preservation is required because retrieval metrics depend on it.

## Graph Architecture

The production pipeline lives in `src/graph/`.

- `state.py`: defines `GraphState`
- `nodes.py`: contains graph node functions
- `edges.py`: contains conditional edge logic
- `builder.py`: assembles and compiles the graph

Every node receives the full `GraphState` and returns a partial dictionary of updated keys. Nodes never mutate state in place.

`GraphState` keys:

```python
question: str
generation: str | None
documents: list[Document]
graded_docs: list[tuple[Document, str]]
memory_context: str | None
rewrite_count: int
route: Literal["vectorstore", "websearch", "direct"]
```

Required nodes:

- `router`: routes the question to `vectorstore`, `websearch`, or `direct`
- `memory_inject`: loads relevant long-term memory before retrieval
- `retrieve`: runs the selected retriever or Tavily; query transformers are called inside this node
- `grade_documents`: grades each document as exactly `relevant`, `ambiguous`, or `irrelevant`
- `rewrite_query`: rewrites the question, increments `rewrite_count`, and can be interrupted for HITL mode
- `generate`: builds the final answer from relevant or ambiguous docs plus memory context
- `self_rag_reflect`: checks groundedness and usefulness before accepting the answer
- `memory_write`: stores successful question, answer, and document context in long-term memory
- `debug_memory`: optional debug-only memory inspection node

Before routing back to retrieval, always check `state["rewrite_count"] < settings.MAX_RETRIES`.

## Memory

Keep short-term and long-term memory separate.

- Short-term memory is `MemorySaver` passed to `graph.compile()`. It stores thread-level graph state.
- Long-term memory lives in `src/memory/long_term_store.py`. It is a separate persistence layer keyed by semantic similarity to the incoming question.

`memory_inject` reads long-term memory. `memory_write` updates it after a successful generation. Do not pass `MemorySaver` where the long-term store is expected.

## Implementation Contracts

Chunkers expose:

```python
def chunk_documents(docs: list[dict]) -> list[Document]:
    ...
```

`src/chunking/parent_document.py` returns `(child_docs, parent_docs)`.

Retrievers subclass `BaseRetriever` and implement:

```python
def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
    ...
```

Query transformers wrap a retriever and are called from the `retrieve` graph node, not modeled as separate graph nodes.

The evaluation harness exposes:

```python
def evaluate(
    pipeline_fn: Callable[[str], tuple[str, list[Document]]],
    qa_pairs: list[dict],
    experiment_id: str,
    metadata: dict,
) -> dict:
    ...
```

It writes `eval/results/{experiment_id}.json` and appends to `eval/comparison_table.csv` before returning. Use `try/finally` so partial results are written on exception.

## Coding Rules

- Use `logging.getLogger(__name__)`; do not use `print` in project modules.
- Nodes return partial dicts and never mutate input state.
- Grader parsing is defensive: unexpected output becomes `"ambiguous"` and never crashes the node.
- BM25 is rebuilt from chunk lists each run and is not persisted.
- Chroma is persisted with `persist_directory=str(settings.VECTORSTORE_DIR / strategy_name)`.
- CrossEncoder input is always `[(query, doc.page_content), ...]` passed to `model.predict()`.
- Results are always written, even when evaluation has a partial exception.
