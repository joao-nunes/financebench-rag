from abc import ABC, abstractmethod

from src.retrieval.models import RetrievalResult


class Retriever(ABC):

    @abstractmethod
    def retrieve(self, query: str) -> list[RetrievalResult]:
        """Retrieve relevant documents."""
        raise NotImplementedError
