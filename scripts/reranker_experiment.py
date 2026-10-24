#!/usr/bin/env python
"""
Compare cross-encoder rerankers on the cached production candidates.

Re-scores the 75 dense candidates per question saved by
context_budget_experiment.py (experiments/context_budget/ranked.pkl), so
every reranker sees exactly the same candidates. Reports how often the
FinanceBench evidence page is ranked high enough to reach the LLM.

Examples
--------
python scripts/reranker_experiment.py --split dev
python scripts/reranker_experiment.py --split test --models BAAI/bge-reranker-v2-m3
"""

from __future__ import annotations

import os

# faiss, torch and scikit-learn each bundle their own libomp; on macOS the
# OpenMP worker threads then segfault. GPU work does not need OpenMP.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm

from src.config import CONTEXT_TOKEN_BUDGET
from src.evaluation.financebench import load_questions
from src.retrieval.context_selector import TokenBudgetSelector

MODELS = [
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "BAAI/bge-reranker-v2-m3",
    "mixedbread-ai/mxbai-rerank-large-v1",
    "Alibaba-NLP/gte-reranker-modernbert-base",
    "cross-encoder/ms-marco-MiniLM-L-12-v2",
]
BUDGETS = (2000, 4000, CONTEXT_TOKEN_BUDGET)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("experiments/context_budget/ranked.pkl"),
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/reranker"))
    return parser.parse_args()


def evidence_metrics(question: dict, ranked: list, n_tokens: list[int]) -> dict:
    evidence = {(e["doc_name"], e["evidence_page_num"]) for e in question["evidence"]}
    hits = [
        {(c.metadata["document_id"], c.metadata.get("page"))} & evidence for c in ranked
    ]

    metrics = {}
    for k in (1, 5):
        metrics[f"recall@{k}"] = len(set().union(*hits[:k])) / len(evidence)

    first = next((i for i, h in enumerate(hits) if h), None)
    metrics["mrr"] = 0.0 if first is None else 1.0 / (first + 1)

    for budget in BUDGETS:
        used, covered = 0, set()
        for i, (h, n) in enumerate(zip(hits, n_tokens, strict=True)):
            if i and used + n > budget:
                break
            used += n
            covered |= h
        metrics[f"recall@{budget}tok"] = len(covered) / len(evidence)

    return metrics


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.split)
    with args.candidates.open("rb") as f:
        candidates = pickle.load(f)

    counter = TokenBudgetSelector(1)
    tokens = {
        q["financebench_id"]: {
            id(c): counter.count_tokens(c.content)
            for c in candidates[q["financebench_id"]]
        }
        for q in questions
    }

    summary_path = args.output / f"summary_{args.split}.csv"
    summary = (
        pd.read_csv(summary_path, index_col=0).to_dict("index")
        if summary_path.exists()
        else {}
    )

    for name in args.models:
        model = CrossEncoder(name, device=device())

        rows = []
        start = time.perf_counter()
        for q in tqdm(questions, desc=name):
            chunks = candidates[q["financebench_id"]]
            scores = model.predict(
                [(q["question"], c.content) for c in chunks],
                batch_size=32,
                show_progress_bar=False,
            )
            ranked = [chunks[i] for i in np.argsort(-np.asarray(scores))]
            n_tokens = [tokens[q["financebench_id"]][id(c)] for c in ranked]
            rows.append(
                {
                    "financebench_id": q["financebench_id"],
                    **evidence_metrics(q, ranked, n_tokens),
                }
            )
        elapsed = time.perf_counter() - start

        frame = pd.DataFrame(rows)
        frame.to_csv(
            args.output / f"{args.split}_{name.replace('/', '__')}.csv", index=False
        )

        summary[name] = {
            **frame.drop(columns="financebench_id").mean().to_dict(),
            "ms_per_question": 1000 * elapsed / len(questions),
        }
        pd.DataFrame.from_dict(summary, orient="index").to_csv(summary_path)
        print(pd.DataFrame.from_dict(summary, orient="index").round(3).to_string())

        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
