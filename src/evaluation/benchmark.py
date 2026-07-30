from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from tqdm.auto import tqdm

from src.evaluation.dataset import EvaluationDataset
from src.evaluation.models import BaseEvaluator, BenchmarkResult
from src.evaluation.pipeline import RAGPipeline


class BenchmarkRunner:

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        evaluator: BaseEvaluator,
    ):
        self._rag_pipeline = rag_pipeline
        self._evaluator = evaluator

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

            pipeline_result = self._rag_pipeline.invoke(sample.question)

            metrics = self._evaluator.evaluate(
                sample,
                pipeline_result,
            )

            yield BenchmarkResult(
                sample=sample,
                result=pipeline_result,
                metrics=metrics,
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

    def __enter__(self) -> BenchmarkWriter:
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
