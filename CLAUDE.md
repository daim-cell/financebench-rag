# CLAUDE.md

## What this project is

A RAG benchmarking and agentic QA system built on the FinanceBench dataset. It has two layers:

- **Evaluation layer** — benchmarks four pipeline axes (chunking strategy, retrieval method, query transformation, cross-encoder reranking) and records every combination in `eval/comparison_table.csv`.
- **Production layer** — a LangGraph `StateGraph` implementing CRAG, Self-RAG reflection, adaptive routing, persistent memory, and human-in-the-loop streaming over the same FinanceBench corpus.

The corpus is 83 SEC filings (10-K, 10-Q, 8-K, earnings reports) with 150 expert-written Q&A pairs. A 30-pair stratified subset lives at `eval/qa_pairs_30.json` — never modify it after creation.

---

## Hardware — M1 MacBook Air (enforce these in every file you touch)

- Always pass `device="mps"` to every `SentenceTransformer()` and `CrossEncoder()` call
- Install and import `faiss-cpu` only — `faiss-gpu` has no Metal support, never use it
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the environment if MPS throws an unsupported-op error — do not change the device to cpu in code
- Ollama uses Metal automatically; it must be running (`ollama serve`) before any LLM, graph, or RAGAS call

---

## Stack

| Role | Library | Key detail |
|---|---|---|
| LLM | `langchain-ollama` `OllamaLLM` | Model name from `settings.OLLAMA_MODEL` |
| Embedder | `sentence-transformers` | Always `device=settings.DEVICE` |
| Agentic graph | `langgraph` `StateGraph` | CRAG loop, Self-RAG, router, memory |
| Short-term memory | `langgraph` `MemorySaver` | Thread-level checkpointer |
| Web search fallback | `tavily-python` | Fires when all docs grade irrelevant |
| Vector store | `chromadb` + `langchain-chroma` | Primary store; `faiss-cpu` for recall benchmarks only |
| Sparse search | `rank-bm25` via `BM25Retriever` | In-memory, rebuilt each run — never persisted |
| Reranker | `sentence-transformers` `CrossEncoder` | Model from `settings.RERANK_MODEL` |
| Framework | `langchain` 0.3+ / `langchain-community` | |
| Evaluation | `ragas` 0.2+ | Wrapped around Ollama via `LangchainLLMWrapper` |

---

## Configuration

All constants live in `config/settings.py`. Never hardcode any of these in module files.

```python
from pathlib import Path
import os

ROOT = Path(__file__).parent.parent

# Paths
DATA_DIR        = ROOT / "data"
PDF_DIR         = DATA_DIR / "raw" / "pdfs"
PROCESSED_DIR   = DATA_DIR / "processed"
VECTORSTORE_DIR = ROOT / "vectorstores"
EVAL_DIR        = ROOT / "eval"
RESULTS_DIR     = EVAL_DIR / "results"
MEMORY_DIR      = ROOT / "memory"

# Models
OLLAMA_MODEL  = "llama3.2:3b"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE        = "mps"

# Chunking
CHUNK_SIZE          = 512
CHUNK_OVERLAP       = 51
CHILD_CHUNK_SIZE    = 256
PARENT_CHUNK_SIZE   = 2048

# Retrieval
TOP_K_DENSE   = 20
TOP_K_FINAL   = 5
RRF_K         = 60

# Graph
MAX_RETRIES        = 3     # Self-RAG and CRAG rewrite loop cap
TAVILY_API_KEY     = os.getenv("TAVILY_API_KEY", "")

# Evaluation
EVAL_PAIRS_PATH  = EVAL_DIR / "qa_pairs_30.json"
COMPARISON_TABLE = EVAL_DIR / "comparison_table.csv"
EVAL_SAMPLE_SIZE = 30
RANDOM_SEED      = 42

HF_DATASET_ID = "PatronusAI/financebench"
```

---

## Graph architecture — StateGraph

The production pipeline is a LangGraph `StateGraph` defined in `src/graph/`. Every node receives the full `GraphState` and returns a partial dict of updated keys. Nodes never mutate state in place.

### State schema

```python
class GraphState(TypedDict):
    question:       str
    generation:     str | None
    documents:      list[Document]
    graded_docs:    list[tuple[Document, str]]   # (doc, "relevant"|"ambiguous"|"irrelevant")
    memory_context: str | None                   # injected from long-term store
    rewrite_count:  int                          # loop guard for CRAG + Self-RAG
    route:          Literal["vectorstore", "websearch", "direct"]
```

### Nodes

**`router`** — classifies `state["question"]` as `vectorstore`, `websearch`, or `direct` via LLM prompt. Returns `{"route": ...}`. Wired with a conditional edge so the graph branches before any retrieval. Questions about the FinanceBench corpus → `vectorstore`; general financial queries with no corpus anchor → `websearch`; greetings or meta questions → `direct`.

**`memory_inject`** — queries the long-term store for memories relevant to `state["question"]`, writes them to `state["memory_context"]`. Runs before `retrieve` so context is available to all downstream nodes.

**`retrieve`** — runs the configured retriever (dense / sparse / hybrid_rrf) or Tavily depending on `state["route"]`. Returns `{"documents": [...]}`. Pre-retrieval transformers (HyDE, multi-query, step-back) are called from inside this node, not as separate nodes.

**`grade_documents`** — passes each doc through an LLM grader that returns exactly `"relevant"`, `"ambiguous"`, or `"irrelevant"`. Parses defensively — any unexpected output becomes `"ambiguous"`, never raises. Returns `{"graded_docs": [...]}`. Conditional edge: if all docs are irrelevant and `rewrite_count < MAX_RETRIES` → `rewrite_query`; otherwise → `generate`.

**`rewrite_query`** — rewrites `state["question"]` using the LLM, informed by `state["memory_context"]`. Increments `rewrite_count`. Routes back to `retrieve`. An interrupt is placed here for human-in-the-loop mode — execution pauses and yields state before the rewrite fires.

**`generate`** — assembles context from relevant and ambiguous docs in `state["graded_docs"]`, injects `state["memory_context"]`, calls `OllamaLLM`, returns `{"generation": ...}`.

**`self_rag_reflect`** — runs two checks on the generation:
- `isGrounded`: does the answer stay within the retrieved context?
- `isUseful`: does it actually address the original question?

If either fails and `rewrite_count < MAX_RETRIES` → routes to `rewrite_query`. Otherwise accepts the generation. This cap must be checked before routing — never allow an unbounded loop.

**`memory_write`** — after a successful generation, stores the (question, answer, docs) tuple in the long-term store. Runs as a terminal node after `self_rag_reflect` passes.

**`debug_memory`** — optional inspection node that logs what was injected into each prompt. Only wired into the graph when `debug=True` is passed to invocation.

### CRAG loop topology

```
router
  ├─ vectorstore/websearch → memory_inject → retrieve → grade_documents
  │                                               ↑           │
  │                                               │    all irrelevant + retries left?
  │                                               └── rewrite_query ◄── [INTERRUPT POINT]
  │                                                           │
  │                                              some relevant/ambiguous?
  │                                                           ↓
  └─ direct ─────────────────────────────────────────────► generate → self_rag_reflect
                                                                            │
                                                              not grounded/useful + retries left?
                                                                            └── rewrite_query (loop)
                                                                            │
                                                              passed or retries exhausted?
                                                                            ↓
                                                                      memory_write → END
```

### Compiling the graph

```python
from langgraph.checkpoint.memory import MemorySaver

graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["rewrite_query"],   # only active when HITL mode is on
)
```

---

## Memory management

Two distinct stores — never conflate them.

**Short-term (thread-level)**: `MemorySaver` passed to `compile()`. Persists the full `GraphState` across turns within the same `thread_id`. Automatic — no node-level code required. Access via `graph.get_state(config)`.

**Long-term (cross-session)**: Implemented in `src/memory/long_term_store.py`. Keyed by semantic similarity to the incoming question. `memory_inject` queries it; `memory_write` updates it after successful generation. The two stores are never the same object. The long-term store is a separate persistence layer (e.g. a simple JSON/SQLite on disk keyed by embedding similarity), not a LangGraph primitive.

**Memory-aware rewriting**: `rewrite_query` receives `state["memory_context"]` and must include it in the rewrite prompt — prior context may reveal a specific fiscal year, entity, or scope the user expects but didn't restate.

---

## Directory layout

```
config/settings.py
src/
  ingestion/pdf_extractor.py
  chunking/                        fixed_size, recursive, semantic, parent_document
  retrieval/                       dense, sparse, hybrid_rrf, parent_document_retriever, chunk_cache
  query_transform/                 hyde, multi_query, step_back   ← called from retrieve node
  reranking/cross_encoder.py
  graph/
    state.py                       GraphState TypedDict
    nodes.py                       all node functions
    edges.py                       conditional edge logic
    builder.py                     assembles and compiles the graph
  memory/
    long_term_store.py
  evaluation/
    harness.py
    metrics.py
scripts/
  download_pdfs.py
  extract_text.py
  build_vectorstore.py
  run_graph.py                     streaming graph entry point
  run_experiment.py                single benchmarking run
  run_experiment_grid.py           full comparison table
eval/
  qa_pairs_30.json                 locked — never overwrite
  comparison_table.csv
  results/
```

---

## Key contracts

### Chunker signature

```python
def chunk_documents(docs: list[dict]) -> list[Document]: ...
```

Exception: `parent_document.py` returns `tuple[list[Document], list[Document]]` — `(child_docs, parent_docs)`.

`doc_name`, `page_num`, and `chunk_id` must be present in every returned Document's metadata.

### Retriever pattern

All retrievers subclass `BaseRetriever` and implement `_get_relevant_documents(query, *, run_manager=None) -> list[Document]`. Must carry `doc_name`, `page_num`, `chunk_id` forward unchanged.

### Grader contract

LLM grader in `grade_documents` must return exactly one of `"relevant"`, `"ambiguous"`, `"irrelevant"` per document. Parse defensively — unexpected output → `"ambiguous"`, never raises, never crashes the node.

### Harness contract

```python
def evaluate(
    pipeline_fn: Callable[[str], tuple[str, list[Document]]],
    qa_pairs: list[dict],
    experiment_id: str,
    metadata: dict,
) -> dict
```

Writes `eval/results/{experiment_id}.json` and appends to `eval/comparison_table.csv` before returning. Partial results written on exception via try/finally.

---

## Coding rules

- **No hardcoded constants** — every path, model name, threshold, k value from `config.settings`
- **Logging not print** — `logging.getLogger(__name__)` at module level in every file
- **Nodes return dicts, never mutate** — every graph node returns a partial `GraphState` dict; never mutate the input state object
- **Rewrite cap always checked** — before routing back to `retrieve`, verify `state["rewrite_count"] < settings.MAX_RETRIES`; no unbounded loops
- **Grader parsing is defensive** — unknown grader output → `"ambiguous"`, never raises
- **Metadata contract is sacred** — `doc_name`, `page_num`, `chunk_id` must survive every node; hit-rate metrics break silently if stripped
- **BM25 is not persisted** — rebuild from chunk list each run
- **Chroma always persisted** — pass `persist_directory=str(settings.VECTORSTORE_DIR / strategy_name)`
- **CrossEncoder input format** — always `[(query, doc.page_content), ...]` tuples to `model.predict()`
- **Long-term ≠ short-term memory** — `MemorySaver` is thread-scoped state; `long_term_store` is cross-session semantic memory; never pass one where the other is expected
- **Results always written** — harness writes JSON and CSV before returning, even on partial exception
- **Type hints everywhere** — all public function signatures fully annotated

---

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "PYTORCH_ENABLE_MPS_FALLBACK=1" >> .venv/bin/activate
export TAVILY_API_KEY=your_key_here   # required for websearch fallback

# Ollama — must be running before any LLM or graph call
ollama serve &
ollama pull llama3.2:3b

# Data pipeline (both idempotent)
python scripts/download_pdfs.py
python scripts/extract_text.py

# Build Chroma vector stores
python scripts/build_vectorstore.py --strategy fixed_size
python scripts/build_vectorstore.py --strategy recursive
python scripts/build_vectorstore.py --strategy semantic
python scripts/build_vectorstore.py --strategy parent_document

# Run agentic graph — streaming, single question
python scripts/run_graph.py --question "What was Amcor's net AR in FY2020?" --thread-id session_1

# Run with human-in-the-loop (pauses before rewrite_query for approval)
python scripts/run_graph.py --question "..." --thread-id session_1 --hitl

# Run with memory debug inspection
python scripts/run_graph.py --question "..." --thread-id session_1 --debug

# Benchmarking — single experiment
python scripts/run_experiment.py --strategy recursive --retriever dense --transformer step_back --reranker

# Benchmarking — full comparison grid
python scripts/run_experiment_grid.py
```