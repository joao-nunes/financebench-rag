from src.retrieval.models import RetrievalResult
from src.retrieval.reranker import Reranker


class NoOpReranker(Reranker):

    def rerank(
        self,
        query: str,
        chunks: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        return chunks
