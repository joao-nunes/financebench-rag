from dataclasses import dataclass

from src.retrieval.retriever import Retriever
from src.retrieval.reranking import CrossEncoderReranker
from src.generation.generator import Generator
from src.generation.prompt_builder import PromptBuilder
from src.retrieval.models import RetrievalResult
from src.metrics import PipelineMetrics
import time
import logging

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class PipelineResponse:
    answer: str
    retrieved_documents: list[RetrievalResult]
    metrics: PipelineMetrics


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
        metrics = PipelineMetrics()
        pipeline_start = time.perf_counter()
        start = pipeline_start

        retrieved = self.retriever.retrieve(question)
        metrics.retrieved_documents = len(retrieved)
        metrics.retrieval_time = time.perf_counter() - start
        logger.info( "Retrieved %d candidate chunks", metrics.retrieved_documents)

        start = time.perf_counter()
        reranked = self.reranker.rerank(
            query=question,
            chunks=retrieved,
        )
        metrics.reranked_documents = len(reranked)
        metrics.reranking_time = time.perf_counter() - start
        logger.info("Reranked to top %d chunks", metrics.reranked_documents)


        start = time.perf_counter()
        prompt = self.prompt_builder.build(
            question=question,
            context=reranked,
        )

        metrics.prompt_length = len(prompt)

        metrics.prompt_build_time = time.perf_counter() - start
        logger.info("Generating answer")

        start = time.perf_counter()
        answer = self.generator.generate(prompt)

        metrics.generation_time = time.perf_counter() - start

        

        metrics.pipeline_time = time.perf_counter() - pipeline_start

        logger.info("Pipeline completed successfully")
        
        logger.info(
            "Pipeline metrics | total=%.3fs | retrieval=%.3fs | reranking=%.3fs | prompt=%.3fs | generation=%.3fs | retrieved=%d | reranked=%d | prompt_length=%d",
            metrics.pipeline_time,
            metrics.retrieval_time,
            metrics.reranking_time,
            metrics.prompt_build_time,
            metrics.generation_time,
            metrics.retrieved_documents,
            metrics.reranked_documents,
            metrics.prompt_length,
)

        return PipelineResponse(
            answer=answer,
            retrieved_documents=reranked,
            metrics=metrics,
        )