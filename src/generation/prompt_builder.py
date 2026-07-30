from abc import ABC, abstractmethod

from src.generation.templates import RAG_PROMPT
from src.retrieval.models import RetrievalResult


class PromptBuilder(ABC):

    @abstractmethod
    def build(
        self,
        question: str,
        context: list[RetrievalResult],
    ) -> str:
        """Build a prompt for the language model."""
        pass



