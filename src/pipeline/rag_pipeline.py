from dataclasses import dataclass

from src.retrieval.retriever import Retriever
from src.retrieval.reranking import CrossEncoderReranker
from src.generation.generator import Generator
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.models import RetrievalResult

import logging

logger = logging.getLogger(__name__)

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

        logger.info("Retrieved %d candidate chunks", len(retrieved))

        reranked = self.reranker.rerank(
            query=question,
            chunks=retrieved,
        )

        logger.info("Reranked to top %d chunks", len(reranked))

        prompt = self.prompt_builder.build(
            question=question,
            context=reranked,
        )

        logger.info("Generating answer")

        answer = self.generator.generate(prompt)

        logger.info("Pipeline completed successfully")
        
        return PipelineResponse(
            answer=answer,
            prompt=prompt,
            retrieved_documents=reranked,
        )