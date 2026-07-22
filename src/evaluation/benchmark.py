from __future__ import annotations

from .dataset import EvaluationDataset
from .models import BenchmarkResult
from .pipeline import RAGPipeline
from .retrieval import RetrievalEvaluator


class BenchmarkRunner:
    """Runs a benchmark over an evaluation dataset."""

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        retrieval_evaluator: RetrievalEvaluator,
    ) -> None:
        self._rag_pipeline = rag_pipeline
        self._retrieval_evaluator = retrieval_evaluator

    def evaluate(
        self,
        dataset: EvaluationDataset,
    ) -> list[BenchmarkResult]:
        """Evaluate a RAG pipeline on every sample in a dataset."""

        results: list[BenchmarkResult] = []

        for sample in dataset:

            pipeline_result = self._rag_pipeline.invoke(sample.question)

            retrieval_metrics = self._retrieval_evaluator.evaluate(
                sample,
                pipeline_result,
            )

            results.append(
                BenchmarkResult(
                    sample=sample,
                    result=pipeline_result,
                    retrieval_metrics=retrieval_metrics,
                )
            )

        return results