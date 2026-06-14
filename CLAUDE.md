# CLAUDE.md

## What this project is

A RAG benchmarking system built on the FinanceBench dataset. It evaluates and compares the performance impact of four pipeline axes — chunking strategy, retrieval method, query transformation, and cross-encoder reranking — and produces a fully populated `eval/comparison_table.csv` scoring every combination on answer correctness, faithfulness, retrieval hit-rate, and NDCG@5.

The corpus is 83 SEC filings (10-K, 10-Q, 8-K, earnings reports) with 150 expert-written Q&A pairs. A 30-pair stratified subset lives at `eval/qa_pairs_30.json` and is the locked evaluation ground truth — **never modify it after creation**.

---

## Hardware — M1 MacBook Air (enforce these in every file you touch)

- Always pass `device="mps"` to every `SentenceTransformer()` and `CrossEncoder()` call
- Install and import `faiss-cpu` only — `faiss-gpu` has no Metal support, never use it
- Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the environment if MPS throws an unsupported-op error — do not change the device to cpu in code


---

## Stack

| Role | Library | Key detail |
|---|---|---|
| LLM | `langchain-ollama` `OllamaLLM` | Model name comes from `settings.OLLAMA_MODEL` |
| Embedder | `sentence-transformers` | Always `device=settings.DEVICE` |
| Vector store | `chromadb` + `langchain-chroma` | Primary store; `faiss-cpu` used only for recall benchmarks |
| Sparse search | `rank-bm25` via `BM25Retriever` | In-memory, rebuilt each run — never persisted |
| Reranker | `sentence-transformers` `CrossEncoder` | Model from `settings.RERANK_MODEL` |
| Framework | `langchain` 0.3+ / `langchain-community` | |
| Evaluation | `ragas` 0.2+ | Wrapped around Ollama via `LangchainLLMWrapper` |

---

## Configuration

**All constants live in `config/settings.py`. Never hardcode any of these in module files.**

```python
# config/settings.py
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Paths
DATA_DIR        = ROOT / "data"
PDF_DIR         = DATA_DIR / "raw" / "pdfs"
PROCESSED_DIR   = DATA_DIR / "processed"
VECTORSTORE_DIR = ROOT / "vectorstores"
EVAL_DIR        = ROOT / "eval"
RESULTS_DIR     = EVAL_DIR / "results"

# Models
OLLAMA_MODEL  = "llama3.2:3b"          # swap to "llama3.1:8b" on 16 GB
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"  # swap to "nomic-embed-text-v1.5" on 16 GB
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE        = "mps"

# Chunking
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64      
CHILD_CHUNK   = 256      # stored chunk for parent-document retriever
PARENT_CHUNK  = 2048     # returned chunk for parent-document retriever
SEMANTIC_THRESHOLD = 95  # percentile breakpoint for SemanticChunker

# Retrieval
TOP_K_DENSE   = 20       # candidate pool fed into reranker
TOP_K_FINAL   = 5        # chunks actually sent to the LLM
RRF_K         = 60       # RRF fusion constant

# Evaluation
EVAL_PAIRS_PATH    = EVAL_DIR / "qa_pairs_30.json"
COMPARISON_TABLE   = EVAL_DIR / "comparison_table.csv"
```

---

## Directory layout

```
config/settings.py                  ← all constants; the only place magic numbers live
data/
  financebench_open_source.jsonl
  financebench_document_information.jsonl
  raw/pdfs/                         ← 83 PDFs (~500 MB, gitignored)
  processed/{doc_name}.json         ← PyMuPDF page-by-page output (gitignored)
eval/
  qa_pairs_30.json                  ← locked eval ground truth
  comparison_table.csv              ← one row appended per experiment
  results/{experiment_id}.json      ← raw metric scores per run
scripts/
  download_pdfs.py                  ← idempotent: skips existing PDFs
  extract_text.py                   ← idempotent: skips already-processed docs
  run_experiment_grid.py            ← drives the full 24-combination grid
src/
  ingestion/
    pdf_extractor.py
  chunking/
    fixed_size.py
    recursive.py
    semantic.py
    parent_document.py
  retrieval/
    dense.py
    sparse.py
    hybrid_rrf.py
  query_transform/
    hyde.py
    multi_query.py
    step_back.py
  reranking/
    cross_encoder.py
  evaluation/
    harness.py                      ← single entry point for scoring any pipeline
    metrics.py                      ← hit_rate, ndcg_at_k
  pipeline/
    rag_pipeline.py                 ← composable builder used in the experiment grid
vectorstores/                       ← Chroma collections (gitignored)
  fixed_size/
  recursive/
  semantic/
  parent_document/
```

---

## Data schemas

### Processed document — `data/processed/{doc_name}.json`

```json
{
  "doc_name": "AMCOR_2020_10K",
  "pages": [
    { "page_num": 1, "text": "..." },
    { "page_num": 2, "text": "..." }
  ]
}
```

Use PyMuPDF (`import fitz`) for extraction, not pypdf. It handles mixed table/text layouts in SEC filings correctly.

### `Document.metadata` — required on every chunk at every stage

Chunkers create it. Retrievers must preserve it. Transformers must not strip it.

```python
{
    "doc_name": str,   # e.g. "AMCOR_2020_10K"
    "page_num": int,   # source page the text came from
    "chunk_id": str,   # f"{doc_name}::p{page_num}::c{i}"
}
```

### Eval pair — `eval/qa_pairs_30.json`

```json
{
    "financebench_id": 42,
    "question": "What is Amcor's year end FY2020 net AR in USD millions?",
    "answer": "$1616.00",
    "doc_name": "AMCOR_2020_10K",
    "page_number": 49,
    "evidence_text": "..."
}
```

### Experiment result — `eval/results/{experiment_id}.json`

```json
{
    "experiment_id": "recursive_hybrid_hyde_rerank",
    "chunking_strategy": "recursive",
    "retriever_type": "hybrid_rrf",
    "query_transformer": "hyde",
    "use_reranker": true,
    "scores": {
        "hit_rate_at_5": 0.73,
        "hit_rate_at_10": 0.87,
        "ndcg_at_5": 0.61,
        "answer_correctness": 0.52,
        "faithfulness": 0.78,
        "context_recall": 0.69,
        "context_precision": 0.55
    },
    "timestamp": "2025-01-15T10:30:00"
}
```

`experiment_id` naming convention: `{chunking}_{retriever}_{transformer_or_none}_{rerank_or_base}`
Examples: `fixed_dense_none_base`, `recursive_hybrid_hyde_rerank`

---

## Key contracts

### Chunker signature

All chunkers are module-level functions:

```python
def chunk_documents(docs: list[dict]) -> list[Document]:
    """
    docs: list of dicts loaded from data/processed/*.json
    Returns LangChain Documents with the required metadata schema.
    """
```

Exception: `parent_document.py` returns `tuple[list[Document], list[Document]]` — `(child_docs, parent_docs)` — and manages its own Chroma collection internally.

### Retriever pattern

All retrievers subclass `BaseRetriever` and implement `_get_relevant_documents(query: str, *, run_manager=None) -> list[Document]`. They must carry forward `doc_name`, `page_num`, and `chunk_id` in returned document metadata unchanged.

`HybridRRFRetriever` constructor signature:

```python
def __init__(
    self,
    bm25_retriever: BM25Retriever,
    dense_retriever: BaseRetriever,
    k: int = settings.TOP_K_DENSE,
)
```

RRF score per document: `sum(1 / (RRF_K + rank_i) for rank_i in ranks_across_retrievers)`.

### Query transformer pattern

Transformers wrap a retriever but are not BaseRetriever subclasses:

```python
class SomeTransformer:
    def __init__(self, retriever: BaseRetriever, llm: BaseLLM) -> None: ...
    def get_relevant_documents(self, query: str) -> list[Document]: ...
```

- **HyDE**: generate a hypothetical answer passage with the LLM, embed that passage, retrieve with the embedding
- **Multi-query**: rewrite the query 3 ways via LLM, retrieve for each, union and deduplicate by `chunk_id`
- **Step-back**: prompt the LLM for the broader financial concept behind the question, retrieve on the abstracted query, pass both original and abstracted contexts to the answer LLM

### Harness contract — `src/evaluation/harness.py`

```python
def evaluate(
    pipeline_fn: Callable[[str], tuple[str, list[Document]]],
    qa_pairs: list[dict],
    experiment_id: str,
    metadata: dict,
) -> dict:
    """
    pipeline_fn: receives a query string, returns (answer_str, retrieved_docs).
    Writes eval/results/{experiment_id}.json BEFORE returning.
    Appends a row to eval/comparison_table.csv BEFORE returning.
    Never lets a run vanish silently — partial results are written on exception.
    """
```

Initialise `LangchainLLMWrapper` and `LangchainEmbeddingsWrapper` once at module level, not per call.

### RAGPipeline — `src/pipeline/rag_pipeline.py`

```python
class RAGPipeline:
    def __init__(
        self,
        chunking_strategy: str,         # "fixed_size" | "recursive" | "semantic" | "parent_document"
        retriever_type: str,            # "dense" | "sparse" | "hybrid_rrf"
        query_transformer: str | None,  # "hyde" | "multi_query" | "step_back" | None
        use_reranker: bool,
        llm: BaseLLM,
        embedder: Embeddings,
    ) -> None: ...

    def __call__(self, query: str) -> tuple[str, list[Document]]: ...
```

The experiment grid in `scripts/run_experiment_grid.py` loops over all 24 combinations, constructs a `RAGPipeline` per combination, and calls `harness.evaluate()` for each.

---

## Evaluation metrics — `src/evaluation/metrics.py`

```python
def hit_rate(retrieved_docs: list[Document], gold_page_num: int) -> bool:
    """True if gold_page_num appears in any doc.metadata["page_num"]."""

def ndcg_at_k(retrieved_docs: list[Document], gold_page_num: int, k: int = 5) -> float:
    """Binary relevance: 1 if page matches gold, 0 otherwise."""
```

RAGAS metrics used: `answer_correctness`, `faithfulness`, `context_recall`, `context_precision`.
Feed `context_recall` and `context_precision` the raw `page_content` strings from retrieved docs as the context list.

---

## Project-specific coding rules

- **No hardcoded constants** — every path, model name, integer threshold, and k value is imported from `config.settings`
- **Logging not print** — `logging.getLogger(__name__)` at module level in every file
- **Metadata contract is sacred** — `doc_name`, `page_num`, `chunk_id` must survive every transformation; downstream metrics break silently if they are stripped
- **BM25 is not persisted** — rebuild `BM25Retriever` from the chunk text list on each run; do not serialise it
- **Chroma always persisted** — pass `persist_directory=str(settings.VECTORSTORE_DIR / strategy_name)` to `Chroma`; never create in-memory-only collections for the main stores
- **CrossEncoder input format** — always pass `[(query, doc.page_content), ...]` tuples to `model.predict()`; not bare strings
- **Idempotent scripts** — `download_pdfs.py` and `extract_text.py` check for existing output before doing work; re-running them must be safe
- **Results always written** — `harness.evaluate()` writes JSON and appends CSV before returning, even when scoring raises a partial exception; wrap with try/finally
- **Type hints everywhere** — all public function signatures are fully annotated

---

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "PYTORCH_ENABLE_MPS_FALLBACK=1" >> .venv/bin/activate   # or add to shell profile

# Ollama — must be running before any LLM or RAGAS call
ollama serve &
ollama pull llama3.2:3b   # or llama3.1:8b on 16 GB

# Data pipeline (both idempotent)
python scripts/download_pdfs.py
python scripts/extract_text.py

# Build all four Chroma vector stores
python -m src.chunking.fixed_size
python -m src.chunking.recursive
python -m src.chunking.semantic
python -m src.chunking.parent_document

# Evaluate a single pipeline configuration
python -c "
from src.pipeline.rag_pipeline import RAGPipeline
from src.evaluation.harness import evaluate
import json, config.settings as s
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings

llm = OllamaLLM(model=s.OLLAMA_MODEL)
emb = HuggingFaceEmbeddings(model_name=s.EMBED_MODEL, model_kwargs={'device': s.DEVICE})
pipe = RAGPipeline('recursive', 'hybrid_rrf', 'hyde', True, llm, emb)
qa  = json.loads(open(s.EVAL_PAIRS_PATH).read())
print(evaluate(pipe, qa, 'recursive_hybrid_hyde_rerank', {}))
"

# Run the full 24-combination experiment grid
python scripts/run_experiment_grid.py
```