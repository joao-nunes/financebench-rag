import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

FINANCEBENCH_DIR = DATA_DIR / "financebench"

PDF_DIR = FINANCEBENCH_DIR / "pdfs"

VECTORSTORE_DIR = DATA_DIR / "vectorstore/sample"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_BATCH_SIZE = 64

EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 75

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
