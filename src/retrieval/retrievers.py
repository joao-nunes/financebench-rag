from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS

from src.config import TOP_K


def create_retriever(
    vectorstore: FAISS,
) -> VectorStoreRetriever:
    """
    Create a retriever from a vector store.
    """

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )


def retrieve(
    retriever: VectorStoreRetriever,
    query: str,
) -> list[Document]:
    """
    Retrieve relevant documents for a query.
    """

    return retriever.invoke(query)