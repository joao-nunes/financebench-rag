import json
from pathlib import Path

from .models import EvaluationSample
from abc import ABC, abstractmethod
import pandas as pd

from abc import ABC, abstractmethod

class EvaluationDataset(ABC):

    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def __len__(self):
        ...



import re
from pathlib import Path

YEAR_PATTERN = re.compile(r"(20\d{2})")
FILING_PATTERN = re.compile(r"(10K|10Q|EARNINGS)$")


def parse_document_name(doc_name: str):
    """
    Extract company, year and filing type from a FinanceBench document name.

    Examples
    --------
    APPLE_2022_10K
        -> ("APPLE", 2022, "10K")

    JPMORGAN_2021Q1_10Q
        -> ("JPMORGAN", 2021, "10Q")

    JOHNSON_JOHNSON_2023Q2_EARNINGS
        -> ("JOHNSON_JOHNSON", 2023, "EARNINGS")
    """

    stem = Path(doc_name).stem

    # Year
    year_match = YEAR_PATTERN.search(stem)
    if year_match is None:
        raise ValueError(f"Could not extract year from '{doc_name}'")

    year = int(year_match.group(1))

    # Filing type
    filing_match = FILING_PATTERN.search(stem)
    filing_type = filing_match.group(1) if filing_match else None

    # Company = everything before the year
    company = stem[:year_match.start()].rstrip("_")

    return company, year, filing_type


class FinanceBenchDataset(EvaluationDataset):

    def __init__(
        self,
        samples: list[EvaluationSample],
    ):
        self._samples = samples


class FinanceBenchDataset(EvaluationDataset):

    def __init__(
        self,
        samples: list[EvaluationSample],
    ):
        self._samples = samples



    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
    ) -> "FinanceBenchDataset":

        samples = []

        path = Path(path)

        with path.open("r", encoding="utf-8") as f:

            for line in f:

                record = json.loads(line)

                samples.append(
                    EvaluationSample(
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
                )

        return cls(samples)

    def __iter__(self):
        return iter(self._samples)

    def __len__(self):
        return len(self._samples)

    def to_dataframe(self) -> pd.DataFrame:

        rows = []

        for sample in self:

            company, year, filing_type = parse_document_name(
                sample.source_document
            )

            rows.append(
                {
                    "question_id": sample.metadata["id"],
                    "question": sample.question,
                    "reference_answer": sample.reference_answer,
                    "company": company,
                    "year": year,
                    "filing_type": filing_type,
                    "question_type": sample.metadata["question_type"],
                    "reasoning": sample.metadata["reasoning"],
                    "subset": sample.metadata["subset"],
                    "source_document": sample.source_document,
                }
            )

        return pd.DataFrame(rows)

    def to_csv(self, path: str | Path):

        df = self.to_dataframe()
        df.to_csv(path, index=False)

    
    def subset(
        self,
        question_ids: list[str],
    ) -> "FinanceBenchDataset":

        ids = set(question_ids)

        samples = [
            sample
            for sample in self
            if sample.metadata["id"] in ids
        ]

        return FinanceBenchDataset(samples)
    