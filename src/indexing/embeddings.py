from langchain_huggingface import HuggingFaceEmbeddings

from src.config import (
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_DEVICE
)


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Create and return the embedding model.
    """

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": True,
                        "batch_size": EMBEDDING_BATCH_SIZE,},
    )