import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# =========================
# Paths
# =========================

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"

# Dataset and index served by the API. The sample is committed to the
# repository; point these at data/financebench and data/vectorstore/full
# to serve the full benchmark.
API_DATASET_DIR = DATA_DIR / "sample"
VECTORSTORE_DIR = DATA_DIR / "vectorstore/sample"

# Company name of each filing; used to detect which filings a question is about.
DOCUMENT_INFO_FILE = API_DATASET_DIR / "data/financebench_document_information.jsonl"

# Full FinanceBench, used by the experiment scripts (not committed).
FINANCEBENCH_DIR = DATA_DIR / "financebench"

# =========================
# Indexing
# =========================

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# =========================
# Retrieval
# =========================

TOP_K = 75

RERANKER_MODEL = "BAAI/bge-reranker-base"

# Tokens of reranked context passed to the LLM. Chosen with
# scripts/context_budget_experiment.py: answer correctness rose from 0.29
# (5 chunks, ~800 tokens) to 0.54 at 8000 tokens and plateaued beyond.
CONTEXT_TOKEN_BUDGET = 8000

# Sources returned by the API; the LLM still sees the full context budget.
MAX_SOURCES = 5

# =========================
# Generation
# =========================

GENERATION_MODEL = "gpt-4.1-mini"

# Grades answers against the FinanceBench reference in the experiments.
JUDGE_MODEL = "gpt-4.1"
