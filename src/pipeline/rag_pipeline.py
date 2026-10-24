import logging
import time
from dataclasses import asdict, dataclass

from src.generation.generator import Generator
from src.generation.prompt_builder import PromptBuilder
from src.metrics import PipelineMetrics
from src.observability import current_trace_id, observation
from src.retrieval.context_selector import TokenBudgetSelector
from src.retrieval.models import RetrievalResult
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineResponse:
    answer: str
    retrieved_documents: list[RetrievalResult]
    metrics: PipelineMetrics
    trace_id: str | None = None


def _summarize(chunks: list[RetrievalResult], limit: int = 10) -> list[dict]:
    """Compact view of ranked chunks for traces: filing, page and score."""

    return [
        {
            "document_id": c.metadata.get("document_id", c.document_id),
            "page": c.metadata.get("page"),
            "score": round(c.score, 4),
        }
        for c in chunks[:limit]
    ]


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker,
        prompt_builder: PromptBuilder,
        generator: Generator,
        context_selector: TokenBudgetSelector | None = None,
    ) -> None:

        self.retriever = retriever
        self.reranker = reranker
        self.prompt_builder = prompt_builder
        self.generator = generator
        self.context_selector = context_selector

    def answer(self, question: str) -> PipelineResponse:
        """
        Complete RAG inference pipeline.
        """
        with observation("rag-pipeline", input=question) as trace:
            try:
                response = self._answer(question)
            except Exception as e:
                logger.exception("Pipeline failed.")
                trace.update(level="ERROR", status_message=str(e))
                raise

            trace.update(
                output=response.answer,
                metadata=asdict(response.metrics),
            )
            response.trace_id = current_trace_id()
            return response

    def _answer(self, question: str) -> PipelineResponse:
        metrics = PipelineMetrics()
        pipeline_start = time.perf_counter()
        start = pipeline_start

        with observation("retrieval", as_type="retriever", input=question) as span:
            retrieved = self.retriever.retrieve(question)
            span.update(output=_summarize(retrieved))
        metrics.retrieved_documents = len(retrieved)
        metrics.retrieval_time = time.perf_counter() - start
        logger.info("Retrieved %d candidate chunks", metrics.retrieved_documents)

        start = time.perf_counter()
        with observation("reranking", input={"candidates": len(retrieved)}) as span:
            reranked = self.reranker.rerank(
                query=question,
                chunks=retrieved,
            )
            span.update(output=_summarize(reranked))
        metrics.reranked_documents = len(reranked)
        metrics.reranking_time = time.perf_counter() - start
        logger.info("Reranked to top %d chunks", metrics.reranked_documents)

        if self.context_selector is not None:
            with observation("context-selection") as span:
                reranked = self.context_selector.select(reranked)
                span.update(output={"chunks": len(reranked)})
        metrics.context_documents = len(reranked)

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
            "Pipeline metrics | total=%.3fs | retrieval=%.3fs | "
            "reranking=%.3fs | prompt=%.3fs | generation=%.3fs | retrieved=%d | "
            "reranked=%d | context=%d | prompt_length=%d",
            metrics.pipeline_time,
            metrics.retrieval_time,
            metrics.reranking_time,
            metrics.prompt_build_time,
            metrics.generation_time,
            metrics.retrieved_documents,
            metrics.reranked_documents,
            metrics.context_documents,
            metrics.prompt_length,
        )

        return PipelineResponse(
            answer=answer,
            retrieved_documents=reranked,
            metrics=metrics,
        )
