from __future__ import annotations

from .pipeline import RAGPipeline

from .dataset import FinanceBenchDataset
from .models import (
    BenchmarkResult,
    EvaluationResult,
)
from .retrieval import RetrievalEvaluator


class BenchmarkRunner:

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        retrieval_evaluator: RetrievalEvaluator,
    ):

        self.rag_pipeline = rag_pipeline
        self.retrieval_evaluator = retrieval_evaluator

    def evaluate(
        self,
        dataset: FinanceBenchDataset,
    ) -> list[BenchmarkResult]:

        benchmark_results = []

        for sample in dataset.load():

            result = self.rag_pipeline.invoke(
                sample.question
            )

            retrieval_metrics = (
                self.retrieval_evaluator.evaluate(
                    sample,
                    result,
                )
            )

            benchmark_results.append(
                BenchmarkResult(
                    sample=sample,
                    result=result,
                    retrieval_metrics=retrieval_metrics,
                )
            )

        return benchmark_results