import json
from pathlib import Path

from .models import EvaluationSample
from abc import ABC, abstractmethod


class EvaluationDataset(ABC):
    """Base class for evaluation datasets."""

    @abstractmethod
    def load(self) -> list[EvaluationSample]:
        """Return the evaluation samples."""
        raise NotImplementedError

class FinanceBenchDataset:
    """
    Loader for the FinanceBench benchmark.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load(self) -> list[EvaluationSample]:

        samples = []

        with self.path.open("r", encoding="utf-8") as f:

            for line in f:

                record = json.loads(line)

                sample = EvaluationSample(
                    question=record["question"],
                    reference_answer=record["answer"],
                    source_document=record["doc_name"],
                    metadata={
                        "id": record["financebench_id"],
                        "company": record["company"],
                        "question_type": record["question_type"],
                        "reasoning": record["question_reasoning"],
                        "subset": record["dataset_subset_label"],
                    },
                )

                samples.append(sample)

        return samples

    def __iter__(self):
        return iter(self.load())

    def __len__(self):
        return len(self.load())