# financebench-rag

A FinanceBench RAG benchmarking project for evaluating how chunking, retrieval, query transformation, and reranking choices affect answer quality and retrieval quality on SEC filing questions.

## Project Goal

The project builds a reproducible experiment grid over the FinanceBench open-source corpus:

- 83 SEC filings, including 10-K, 10-Q, 8-K, and earnings reports
- 150 expert-written Q&A pairs
- a locked 30-question stratified evaluation subset at `eval/qa_pairs_30.json`

The final benchmark output is `eval/comparison_table.csv`, with one row per pipeline combination and metrics for answer correctness, faithfulness, retrieval hit-rate, and NDCG@5.

## Experiment Axes

The benchmark compares four pipeline dimensions:

| Axis | Planned options |
|---|---|
| Chunking | fixed-size, recursive, semantic, parent-document |
| Retrieval | dense, sparse BM25, hybrid reciprocal-rank fusion |
| Query transformation | none, HyDE, multi-query, step-back |
| Reranking | base retrieval, cross-encoder reranking |

Experiment IDs follow:

```text
{chunking}_{retriever}_{transformer_or_none}_{rerank_or_base}
```

Example: `recursive_hybrid_hyde_rerank`

## Stack

| Role | Library |
|---|---|
| LLM | `langchain-ollama` with `OllamaLLM` |
| Embeddings | `sentence-transformers` |
| Vector store | `chromadb` and `langchain-chroma` |
| Sparse search | `rank-bm25` through LangChain BM25 retriever |
| Reranker | `sentence-transformers` `CrossEncoder` |
| Framework | `langchain` and `langchain-community` |
| Evaluation | `ragas` |
| PDF extraction | PyMuPDF |

## Hardware Notes

This repository is configured for an M1 MacBook Air:

- use `device="mps"` through `settings.DEVICE`
- use `faiss-cpu` only
- set `PYTORCH_ENABLE_MPS_FALLBACK=1` if PyTorch MPS hits an unsupported operation

## Directory Layout

```text
config/settings.py
data/
  financebench_open_source.jsonl
  financebench_document_information.jsonl
  raw/pdfs/
  processed/
eval/
  qa_pairs_30.json
  comparison_table.csv
  results/
scripts/
  download_pdfs.py
  extract_text.py
  run_experiment_grid.py
src/
  ingestion/
  chunking/
  retrieval/
  query_transform/
  reranking/
  evaluation/
  pipeline/
vectorstores/
  fixed_size/
  recursive/
  semantic/
  parent_document/
```

Large local artifacts such as PDFs, extracted text, vector stores, and raw evaluation result JSON files are gitignored.

## Data Contracts

Processed documents are stored as `data/processed/{doc_name}.json`:

```json
{
  "doc_name": "AMCOR_2020_10K",
  "pages": [
    { "page_num": 1, "text": "..." }
  ]
}
```

Every LangChain `Document` chunk must preserve this metadata:

```python
{
    "doc_name": str,
    "page_num": int,
    "chunk_id": str,
}
```

Evaluation result files are written to `eval/results/{experiment_id}.json` and summarized in `eval/comparison_table.csv`.

## Development Status

The repository currently contains the project scaffold and contracts. The next implementation milestones are:

1. Download FinanceBench source metadata and PDFs.
2. Extract PDF text with PyMuPDF.
3. Implement chunkers and vector-store builders.
4. Implement dense, sparse, and hybrid retrievers.
5. Add query transformers and optional cross-encoder reranking.
6. Build the evaluation harness and experiment-grid runner.
