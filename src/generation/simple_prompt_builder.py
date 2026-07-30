
from src.generation.templates import RAG_PROMPT
from src.retrieval.models import RetrievalResult
from src.generation.prompt_builder import PromptBuilder


class SimplePromptBuilder(PromptBuilder):

    def build(
        self,
        question: str,
        context: list[RetrievalResult],
    ) -> str:

        documents = "\n\n".join(
            f"[Document {i+1}]\n{doc.content}"
            for i, doc in enumerate(context)
        )

        return RAG_PROMPT.format(
            context=documents,
            question=question,
        )