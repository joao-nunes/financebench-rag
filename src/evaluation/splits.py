from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class Split:
    train_ids: list[str]
    validation_ids: list[str]
    test_ids: list[str]
    metadata: dict

    @classmethod
    def load(cls, directory: str | Path) -> "Split":
        """
        Load a split from disk.
        """

        directory = Path(directory)

        train_ids = pd.read_csv(
            directory / "train.csv",
            dtype={"question_id": str},
        )["question_id"].tolist()

        validation_ids = pd.read_csv(
            directory / "validation.csv",
            dtype={"question_id": str},
        )["question_id"].tolist()

        test_ids = pd.read_csv(
            directory / "test.csv",
            dtype={"question_id": str},
        )["question_id"].tolist()

        metadata_path = directory / "split_info.json"

        if metadata_path.exists():
            with metadata_path.open() as f:
                metadata = json.load(f)
        else:
            metadata = {}

        return cls(
            train_ids=train_ids,
            validation_ids=validation_ids,
            test_ids=test_ids,
            metadata=metadata,
        )

    def save(self, directory: str | Path):
        """
        Save the split to disk.
        """

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {"question_id": self.train_ids}
        ).to_csv(directory / "train.csv", index=False)

        pd.DataFrame(
            {"question_id": self.validation_ids}
        ).to_csv(directory / "validation.csv", index=False)

        pd.DataFrame(
            {"question_id": self.test_ids}
        ).to_csv(directory / "test.csv", index=False)

        with (directory / "split_info.json").open("w") as f:
            json.dump(self.metadata, f, indent=4)
    
    @property
    def development_ids(self) -> list[int]:
        return self.train_ids + self.validation_ids