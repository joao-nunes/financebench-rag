"""
Access to the full FinanceBench dataset used by the experiment scripts.

Questions are read from data/financebench; the train/validation/test
split lives in data/financebench/splits (see scripts/train_test_split.py).
All model choices are made on the development set (train + validation);
the test set is only checked once.
"""

from __future__ import annotations

import json
from typing import Literal

from src.config import FINANCEBENCH_DIR
from src.evaluation.splits import Split

QUESTIONS_FILE = FINANCEBENCH_DIR / "data/financebench_open_source.jsonl"
DOCUMENT_INFO_FILE = FINANCEBENCH_DIR / "data/financebench_document_information.jsonl"
SPLITS_DIR = FINANCEBENCH_DIR / "splits"


def load_questions(split: Literal["dev", "test", "all"] = "all") -> list[dict]:
    """FinanceBench questions, optionally restricted to one split."""

    with QUESTIONS_FILE.open() as f:
        questions = [json.loads(line) for line in f]

    if split == "all":
        return questions

    ids = set(dev_ids() if split == "dev" else Split.load(SPLITS_DIR).test_ids)
    return [q for q in questions if q["financebench_id"] in ids]


def dev_ids() -> set[str]:
    return set(Split.load(SPLITS_DIR).development_ids)
