"""Central project settings.

Keep model names, paths, chunk sizes, retrieval limits, and evaluation
locations here instead of hardcoding them in implementation modules.
"""

from pathlib import Path

ROOT = Path(__file__).parent.parent

# Paths
DATA_DIR = ROOT / "data"
PDF_DIR = DATA_DIR / "raw" / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = ROOT / "vectorstores"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
MEMORY_DIR = ROOT / "memory"

# Models
OLLAMA_MODEL = "llama3.2:3b"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE = "mps"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
CHILD_CHUNK = 256
PARENT_CHUNK = 2048
SEMANTIC_THRESHOLD = 95

# Retrieval
TOP_K_DENSE       = 20
TOP_K_FINAL       = 5
RRF_K             = 60
RRF_DENSE_WEIGHT  = 5.0  # dense leg outweighs sparse in hybrid RRF fusion
RRF_SPARSE_WEIGHT = 1.0

# Evaluation
EVAL_PAIRS_PATH = EVAL_DIR / "qa_pairs_30.json"
COMPARISON_TABLE = EVAL_DIR / "comparison_table.csv"
EVAL_SAMPLE_SIZE = 30
RANDOM_SEED      = 42
SIMILARITY_THRESHOLD = 0.7   # answer_accuracy = 1 if answer_similarity >= this

# Graph
MAX_RETRIES = 3
TAVILY_API_KEY = ""

# Dataset
HF_DATASET_ID = "PatronusAI/financebench"