from __future__ import annotations

from .dataset import EvaluationDataset
from .models import BenchmarkResult
from .pipeline import RAGPipeline
from .retrieval import RetrievalEvaluator
from collections.abc import Iterator
from tqdm.auto import tqdm

import json
from pathlib import Path

from src.evaluation.models import BenchmarkResult
from dataclasses import asdict

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

        return list(self.run(dataset))
    

    def run(
        self,
        dataset: EvaluationDataset,
    ) -> Iterator[BenchmarkResult]:

        for sample in tqdm(
            dataset,
            total=len(dataset),
            desc="Running benchmark",
        ):

            pipeline_result = self._rag_pipeline.invoke(
                sample.question
            )

            retrieval_metrics = (
                self._retrieval_evaluator.evaluate(
                    sample,
                    pipeline_result,
                )
            )

            yield BenchmarkResult(
                sample=sample,
                result=pipeline_result,
                retrieval_metrics=retrieval_metrics,
            )

class BenchmarkWriter:
    """
    Incrementally writes benchmark results to a JSONL file.

    One BenchmarkResult is written per line, allowing long-running
    evaluations to be resumed or inspected before completion.
    """

    def __init__(self, output_path: str | Path):
        self._path = Path(output_path)
        self._file = None

    def __enter__(self) -> "BenchmarkWriter":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self._file is not None:
            self._file.close()

    def write(self, result: BenchmarkResult):

        json.dump(
            asdict(result),
            self._file,
            ensure_ascii=False,
        )

        self._file.write("\n")
        self._file.flush()