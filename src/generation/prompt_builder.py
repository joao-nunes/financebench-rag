from abc import ABC, abstractmethod

from src.generation.templates import RAG_PROMPT
from src.retrieval.retrievers import RetrievalResult


class PromptBuilder(ABC):

    @abstractmethod
    def build(
        self,
        question: str,
        context: list[RetrievalResult],
    ) -> str:
        """Build a prompt for the language model."""
        pass



class SimplePromptBuilder(PromptBuilder):

    def build(
        self,
        question: str,
        context: list[RetrievalResult],
    ) -> str:

        documents = "\n\n".join(
            f"[Document {i+1}]\n{doc.page_content}"
            for i, doc in enumerate(context)
        )

        return RAG_PROMPT.format(
            context=documents,
            question=question,
        )