from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split a list of LangChain Documents into smaller chunks.

    Args:
        documents: List of input documents.

    Returns:
        List of chunked LangChain Documents.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = text_splitter.split_documents(documents)

    return chunks