import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

FINANCEBENCH_DIR = DATA_DIR / "sample"

PDF_DIR = FINANCEBENCH_DIR / "pdfs"

# Company name of each filing; used to detect which filings a question is about.
DOCUMENT_INFO_FILE = FINANCEBENCH_DIR / "data/financebench_document_information.jsonl"

VECTORSTORE_DIR = DATA_DIR / "vectorstore/sample"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_BATCH_SIZE = 64

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 75

RERANKER_MODEL = "BAAI/bge-reranker-base"

# Tokens of reranked context passed to the LLM. Chosen with
# scripts/context_budget_experiment.py: answer correctness rose from 0.29
# (5 chunks, ~800 tokens) to 0.54 at 8000 tokens and plateaued beyond.
CONTEXT_TOKEN_BUDGET = 8000

# Sources returned by the API; the LLM still sees the full context budget.
MAX_SOURCES = 5

# =========================
# OpenAI
# =========================

OPENAI_MODEL = "gpt-4.1"

TEMPERATURE = 0.0

MAX_TOKENS = 512

CACHE_DIR = DATA_DIR / "cache"

DOCUMENTS_CACHE = CACHE_DIR / "documents.pkl"
CHUNKS_CACHE = CACHE_DIR / "chunks.pkl"

INDEX_BATCH_SIZE = 512

CHECKPOINT_DIR = DATA_DIR / "checkpoints" / "financebench"

SOURCE = ROOT_DIR / "data" / "financebench"
DESTINATION = ROOT_DIR / "data" / "sample"

QUESTIONS_FILE = SOURCE / "data/financebench_open_source.jsonl"
DOCUMENT_INFO_FILE = SOURCE / "data/financebench_document_information.jsonl"
SOURCE_DOCUMENTS = SOURCE / "documents"
DESTINATION_DOCUMENTS = DESTINATION / "documents"

# Companies to keep
SAMPLE_COMPANIES = {
    "Apple",
    "Microsoft",
    "Amazon",
    "NVIDIA",
    "Tesla",
}

# ---------------------------------------------------------------------
