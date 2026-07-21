from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

from src.indexing.faiss_store import FAISSStore

def create_vectorstore(
    documents: list[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Build a FAISS vector store from a list of documents.
    """
    return FAISS.from_documents(
        documents=documents,
        embedding=embedding_model,
    )


def save_vectorstore(
    vectorstore: FAISS,
    save_path: str | Path,
) -> None:
    """
    Persist a FAISS index to disk.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vectorstore.save_local(str(save_path))


def load_vectorstore(
    path: str | Path,
    embedding_model: Embeddings,
) -> FAISSStore:
    """
    Load a FAISS vector store from disk.
    """

    store = FAISSStore()

    store.load(
        path=path,
        embedding_model=embedding_model,
    )

    return store