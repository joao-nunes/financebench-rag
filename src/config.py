from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 5