from pathlib import Path

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


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
    load_path: str | Path,
    embedding_model: HuggingFaceEmbeddings,
) -> FAISS:
    """
    Load a persisted FAISS index.
    """

    return FAISS.load_local(
        folder_path=str(load_path),
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )