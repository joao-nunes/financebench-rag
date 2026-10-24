import logging

import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.config import TOP_K
from src.exceptions import RetrievalError
from src.observability import update_current
from src.retrieval.filing_filter import FilingFilter
from src.retrieval.models import RetrievalResult
from src.retrieval.retriever import Retriever

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
    """
    Dense retrieval over a FAISS store.

    With a filing_filter, the search only ranks chunks of the filings the
    question is about, and falls back to the whole index when the filter
    recognises no company.
    """

    def __init__(
        self,
        vectorstore: FAISS,
        top_k: int = TOP_K,
        filing_filter: FilingFilter | None = None,
    ):
        self.vectorstore = vectorstore
        self.top_k = top_k
        self.filing_filter = filing_filter
        self.retriever: VectorStoreRetriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        self._positions: dict[str, list[int]] = {}
        if filing_filter is not None:
            for position, doc_id in vectorstore.index_to_docstore_id.items():
                document = vectorstore.docstore.search(doc_id)
                filing = document.metadata.get("document_id")
                self._positions.setdefault(filing, []).append(position)

    def retrieve(
        self,
        query: str,
    ) -> list[RetrievalResult]:

        try:
            documents = self._search(query)
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
            logger.debug(
                "Failed to retrieve chunks from the vector store.",
                exc_info=True,
            )
            raise RetrievalError("Chunk retrieval failed.") from e

    def _search(self, query: str) -> list[Document]:
        filings = None
        if self.filing_filter is not None:
            filings = self.filing_filter.match(query)

        positions = [p for f in filings or () for p in self._positions.get(f, ())]
        if not positions:
            logger.info("No filing detected; searching the whole index")
            update_current(metadata={"filings": "all"})
            return self.retriever.invoke(query)

        logger.info("Restricting search to filings: %s", ", ".join(filings))
        update_current(metadata={"filings": filings})

        vector = np.asarray(
            [self.vectorstore.embedding_function.embed_query(query)],
            dtype=np.float32,
        )
        selector = faiss.IDSelectorBatch(np.asarray(positions, dtype=np.int64))
        _, ids = self.vectorstore.index.search(
            vector,
            min(self.top_k, len(positions)),
            params=faiss.SearchParameters(sel=selector),
        )

        return [
            self.vectorstore.docstore.search(self.vectorstore.index_to_docstore_id[i])
            for i in ids[0]
            if i >= 0
        ]
