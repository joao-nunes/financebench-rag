from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS

from src.config import TOP_K
from src.retrieval.models import RetrievalResult
from src.retrieval.retriever import Retriever
from src.exceptions import RetrievalError

import logging

logger = logging.getLogger(__name__)


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

class FAISSRetriever(Retriever):

    def __init__(
        self,
        vectorstore: FAISS,
        top_k: int = TOP_K,
    ):
        self.retriever: VectorStoreRetriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievalResult]:

        try: 
            documents = self.retriever.invoke(query)
            return [
            RetrievalResult(
                document_id=doc.metadata.get("id", str(i)),
                content=doc.page_content,
                score=0.0,
                metadata=doc.metadata,
            )
            for i, doc in enumerate(documents)
            ]
        except Exception as e:
            logger.debug("Failed to retrieve chunks from the vector store.",
                         exc_info=True,
            )
            raise RetrievalError("Chunk retrieval failed.") from e

        