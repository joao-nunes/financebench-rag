from dataclasses import dataclass

from src.retrieval.retriever import Retriever
from src.retrieval.reranking import CrossEncoderReranker
from src.generation.generator import Generator
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.models import RetrievalResult


@dataclass(slots=True)
class PipelineResponse:
    answer: str
    prompt: str
    retrieved_documents: list[RetrievalResult]


class RAGPipeline:

    def __init__(
        self,
        retriever: Retriever,
        reranker: CrossEncoderReranker,
        prompt_builder: PromptBuilder,
        generator: Generator,
    ) -> None:

        self.retriever = retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.generator = generator

    def answer(self, question: str) -> PipelineResponse:
        """
        Complete RAG inference pipeline.
        """

        retrieved = self.retriever.retrieve(question)

        reranked = self.reranker.rerank(
            query=question,
            chunks=retrieved,
        )

        prompt = self.prompt_builder.build(
            question=question,
            context=reranked,
        )

        answer = self.generator.generate(prompt)

        return PipelineResponse(
            answer=answer,
            prompt=prompt,
            retrieved_documents=reranked,
        )