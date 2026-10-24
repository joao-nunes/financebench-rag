#!/usr/bin/env python
"""
Measure answer correctness as a function of the context token budget.

Uses the production components end to end: the FAISS index, the
cross-encoder reranker, SimplePromptBuilder and OpenAIGenerator. Each
question is retrieved and reranked once; only the context selection and
generation change between budgets. Answers are graded against the
FinanceBench reference answer by an LLM judge.

Generations and verdicts are cached in --output, so the script can be
interrupted and resumed without paying for the same calls twice.

Examples
--------
python scripts/context_budget_experiment.py --split dev
python scripts/context_budget_experiment.py --split test --budgets 2000 4000
"""

from __future__ import annotations

import os

# faiss, torch and scikit-learn each bundle their own libomp; on macOS the
# OpenMP worker threads then segfault. GPU work does not need OpenMP.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

from src.config import DATA_DIR, OPENAI_MODEL
from src.evaluation.splits import Split
from src.generation.openai_generator import OpenAIGenerator
from src.generation.simple_prompt_builder import SimplePromptBuilder
from src.indexing.embeddings import get_embedding_model
from src.indexing.faiss_store import FAISSStore
from src.retrieval.context_selector import TokenBudgetSelector
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.faiss_retriever import FAISSRetriever

QUESTIONS_FILE = DATA_DIR / "financebench/data/financebench_open_source.jsonl"
SPLITS_DIR = DATA_DIR / "financebench/splits"

# "top5" reproduces the current production behaviour (reranker top_n=5).
DEFAULT_BUDGETS = ["top5", "1000", "2000", "4000", "8000", "16000"]

JUDGE_PROMPT = """
You are grading answers to questions about SEC filings.

Compare the candidate answer with the reference answer and classify it:
- "correct": it reaches the same conclusion and its key figures match the
  reference (allow rounding, different units or formatting, and extra
  correct detail).
- "incorrect": it gives a different figure or conclusion, or is missing
  the key information the question asks for.
- "refused": it declines to answer or says the information is not in the
  context.

Question:
{question}

Reference answer:
{reference}

Candidate answer:
{candidate}
""".strip()


class Verdict(BaseModel):
    verdict: Literal["correct", "incorrect", "refused"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--budgets", nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument(
        "--vectorstore", type=Path, default=DATA_DIR / "vectorstore/full"
    )
    parser.add_argument("--top-k", type=int, default=75)
    parser.add_argument("--reranker", default="BAAI/bge-reranker-base")
    parser.add_argument("--judge-model", default=OPENAI_MODEL)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/context_budget")
    )
    return parser.parse_args()


def load_questions(split: str) -> list[dict]:
    questions = [json.loads(line) for line in QUESTIONS_FILE.open()]

    if split == "all":
        return questions

    splits = Split.load(SPLITS_DIR)
    ids = set(splits.development_ids if split == "dev" else splits.test_ids)

    return [q for q in questions if q["financebench_id"] in ids]


def rank_candidates(
    questions, vectorstore_dir: Path, top_k: int, reranker_model: str, cache: Path
):
    """Retrieve and rerank once per question; the result is budget-independent."""

    ranked = {}
    if cache.exists():
        with cache.open("rb") as f:
            ranked = pickle.load(f)

    missing = [q for q in questions if q["financebench_id"] not in ranked]
    if not missing:
        return ranked

    store = FAISSStore()
    store.load(vectorstore_dir, get_embedding_model())
    retriever = FAISSRetriever(store.store, top_k=top_k)
    reranker = CrossEncoderReranker(model_name=reranker_model, top_n=None)

    for q in tqdm(missing, desc="retrieve + rerank"):
        candidates = retriever.retrieve(q["question"])
        ranked[q["financebench_id"]] = reranker.rerank(q["question"], candidates)

        with cache.open("wb") as f:
            pickle.dump(ranked, f)

    return ranked


def select_context(chunks, budget: str, selector_cache: dict):
    if budget == "top5":
        return chunks[:5]

    if budget not in selector_cache:
        selector_cache[budget] = TokenBudgetSelector(int(budget))

    return selector_cache[budget].select(chunks)


def evidence_in_context(question: dict, chunks) -> float:
    evidence = {(e["doc_name"], e["evidence_page_num"]) for e in question["evidence"]}
    covered = {(c.metadata["document_id"], c.metadata.get("page")) for c in chunks}
    return len(evidence & covered) / len(evidence)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    questions = load_questions(args.split)
    ranked = rank_candidates(
        questions,
        args.vectorstore,
        args.top_k,
        args.reranker,
        args.output / "ranked.pkl",
    )

    records_path = args.output / "records.jsonl"
    done = {}
    if records_path.exists():
        for line in records_path.open():
            r = json.loads(line)
            done[(r["financebench_id"], r["budget"])] = r

    prompt_builder = SimplePromptBuilder()
    generator = OpenAIGenerator()
    judge = OpenAI()
    selectors: dict[str, TokenBudgetSelector] = {}
    counter = TokenBudgetSelector(1)

    jobs = []
    for budget in args.budgets:
        for q in questions:
            if (q["financebench_id"], budget) in done:
                continue
            context = select_context(ranked[q["financebench_id"]], budget, selectors)
            jobs.append((q, budget, context))

    def run(job):
        q, budget, context = job
        answer = generator.generate(prompt_builder.build(q["question"], context))
        verdict = judge.responses.parse(
            model=args.judge_model,
            input=JUDGE_PROMPT.format(
                question=q["question"], reference=q["answer"], candidate=answer
            ),
            text_format=Verdict,
        ).output_parsed.verdict

        return {
            "financebench_id": q["financebench_id"],
            "question_type": q["question_type"],
            "budget": budget,
            "context_chunks": len(context),
            "context_tokens": sum(counter.count_tokens(c.content) for c in context),
            "evidence_recall": evidence_in_context(q, context),
            "answer": answer,
            "verdict": verdict,
        }

    with records_path.open("a") as out, ThreadPoolExecutor(args.workers) as pool:
        for record in tqdm(
            pool.map(run, jobs), total=len(jobs), desc="generate + judge"
        ):
            out.write(json.dumps(record) + "\n")
            out.flush()
            done[(record["financebench_id"], record["budget"])] = record

    ids = {q["financebench_id"] for q in questions}
    frame = pd.DataFrame(
        [r for (qid, b), r in done.items() if qid in ids and b in args.budgets]
    )
    frame["correct"] = frame["verdict"] == "correct"
    frame["refused"] = frame["verdict"] == "refused"

    summary = (
        frame.groupby("budget")[
            [
                "correct",
                "refused",
                "evidence_recall",
                "context_chunks",
                "context_tokens",
            ]
        ]
        .mean()
        .reindex(args.budgets)
    )
    summary.to_csv(args.output / f"summary_{args.split}.csv")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
