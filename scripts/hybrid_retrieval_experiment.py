#!/usr/bin/env python
"""
Compare dense, BM25 and hybrid candidate retrieval on the full index.

For every question the script runs dense (FAISS) and BM25 search once,
to depth --depth, and scores the union of both candidate lists with the
production cross-encoder once. Each configuration below is then a
different way of choosing candidates from those cached results:

    dense         dense top-75 (current production)
    bm25          BM25 top-75
    rrf_75        reciprocal rank fusion of both lists, top-75
    union_75+75   dense top-75 + BM25 top-75 (up to 150 candidates)

Reported metrics: whether the evidence page is among the candidates, and
whether it reaches the LLM after reranking (top 5 / token budgets).

With --oracle-filing both searches only rank chunks of the filing the
question is about. This is an upper bound for a filter that detects the
company and period from the question.

Examples
--------
python scripts/hybrid_retrieval_experiment.py
python scripts/hybrid_retrieval_experiment.py --oracle-filing
"""

from __future__ import annotations

import os

# faiss, torch and scikit-learn each bundle their own libomp; on macOS the
# OpenMP worker threads then segfault. GPU work does not need OpenMP.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import json
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm

from src.config import CONTEXT_TOKEN_BUDGET, DATA_DIR, RERANKER_MODEL
from src.evaluation.splits import Split
from src.indexing.embeddings import get_embedding_model
from src.retrieval.bm25 import BM25Index, reciprocal_rank_fusion
from src.retrieval.context_selector import TokenBudgetSelector

QUESTIONS_FILE = DATA_DIR / "financebench/data/financebench_open_source.jsonl"
SPLITS_DIR = DATA_DIR / "financebench/splits"
BUDGETS = (2000, 4000, CONTEXT_TOKEN_BUDGET)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--vectorstore", type=Path, default=DATA_DIR / "vectorstore/full"
    )
    parser.add_argument("--depth", type=int, default=150)
    parser.add_argument("--oracle-filing", action="store_true")
    parser.add_argument("--workdir", type=Path, default=DATA_DIR / "experiments/hybrid")
    parser.add_argument("--output", type=Path, default=Path("experiments/hybrid"))
    return parser.parse_args()


def configurations(dense: list[int], bm25: list[int]) -> dict[str, list[int]]:
    union = list(dict.fromkeys(dense[:75] + bm25[:75]))
    return {
        "dense": dense[:75],
        "bm25": bm25[:75],
        "rrf_75": reciprocal_rank_fusion([dense, bm25])[:75],
        "rrf_100": reciprocal_rank_fusion([dense, bm25])[:100],
        "union_75+75": union,
    }


def load_chunks(vectorstore: Path):
    with (vectorstore / "index.pkl").open("rb") as f:
        docstore, index_to_id = pickle.load(f)
    return [docstore.search(index_to_id[i]) for i in range(len(index_to_id))]


def cached(path: Path, build):
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)
    value = build()
    with path.open("wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    return value


def search(args, questions, chunks):
    by_filing: dict[str, list[int]] = {}
    for i, chunk in enumerate(chunks):
        by_filing.setdefault(chunk.metadata["document_id"], []).append(i)

    def scope(question):
        if not args.oracle_filing:
            return None
        return np.asarray(by_filing[question["doc_name"]])

    def dense():
        index = faiss.read_index(str(args.vectorstore / "index.faiss"))
        queries = np.asarray(
            get_embedding_model().embed_documents([q["question"] for q in questions]),
            dtype=np.float32,
        )
        if not args.oracle_filing:
            _, ids = index.search(queries, args.depth)
            return {
                q["financebench_id"]: row.tolist()
                for q, row in zip(questions, ids, strict=True)
            }

        results = {}
        for q, query in zip(questions, queries, strict=True):
            ids = scope(q)
            vectors = np.vstack([index.reconstruct(int(i)) for i in ids])
            order = np.argsort(((vectors - query) ** 2).sum(axis=1))[: args.depth]
            results[q["financebench_id"]] = ids[order].tolist()
        return results

    def bm25():
        bm25_path = (
            args.workdir.parent / "bm25.pkl"
            if args.oracle_filing
            else args.workdir / "bm25.pkl"
        )
        if bm25_path.exists():
            index = BM25Index.load(bm25_path)
        else:
            start = time.perf_counter()
            index = BM25Index().build([c.page_content for c in chunks])
            print(f"BM25 index built in {time.perf_counter() - start:.0f}s")
            index.save(bm25_path)
        return {
            q["financebench_id"]: index.search(
                q["question"], args.depth, restrict_to=scope(q)
            )[0].tolist()
            for q in tqdm(questions, desc="bm25")
        }

    return (
        cached(args.workdir / "dense.pkl", dense),
        cached(args.workdir / "bm25_results.pkl", bm25),
    )


def rerank_scores(args, questions, chunks, dense, bm25):
    path = args.workdir / "rerank_scores.pkl"
    scores = cached(path, dict) if path.exists() else {}

    missing = [q for q in questions if q["financebench_id"] not in scores]
    if not missing:
        return scores

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = CrossEncoder(RERANKER_MODEL, device=device)

    for q in tqdm(missing, desc="rerank"):
        qid = q["financebench_id"]
        ids = list(dict.fromkeys(dense[qid] + bm25[qid]))
        values = model.predict(
            [(q["question"], chunks[i].page_content) for i in ids],
            batch_size=32,
            show_progress_bar=False,
        )
        scores[qid] = dict(zip(ids, map(float, values), strict=True))

        with path.open("wb") as f:
            pickle.dump(scores, f, protocol=pickle.HIGHEST_PROTOCOL)

    return scores


def evaluate(question, candidates, chunks, scores, counter) -> dict:
    evidence = {(e["doc_name"], e["evidence_page_num"]) for e in question["evidence"]}

    def hit(i):
        meta = chunks[i].metadata
        return {(meta["document_id"], meta.get("page"))} & evidence

    metrics = {
        "candidate_recall": len(set().union(*map(hit, candidates))) / len(evidence),
        "n_candidates": len(candidates),
    }

    ranked = sorted(candidates, key=scores.__getitem__, reverse=True)
    metrics["recall@5"] = len(set().union(*map(hit, ranked[:5]))) / len(evidence)

    for budget in BUDGETS:
        used, covered = 0, set()
        for rank, i in enumerate(ranked):
            n = counter.count_tokens(chunks[i].page_content)
            if rank and used + n > budget:
                break
            used += n
            covered |= hit(i)
        metrics[f"recall@{budget}tok"] = len(covered) / len(evidence)

    return metrics


def main():
    args = parse_args()
    if args.oracle_filing:
        args.workdir = args.workdir / "oracle_filing"
        args.output = args.output / "oracle_filing"
    args.workdir.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)

    questions = [json.loads(line) for line in QUESTIONS_FILE.open()]
    splits = Split.load(SPLITS_DIR)
    dev_ids = set(splits.development_ids)

    chunks = load_chunks(args.vectorstore)
    print(f"{len(questions)} questions | {len(chunks):,} chunks")

    dense, bm25 = search(args, questions, chunks)
    scores = rerank_scores(args, questions, chunks, dense, bm25)

    counter = TokenBudgetSelector(1)
    rows = []
    for q in tqdm(questions, desc="evaluate"):
        qid = q["financebench_id"]
        for name, candidates in configurations(dense[qid], bm25[qid]).items():
            rows.append(
                {
                    "financebench_id": qid,
                    "split": "dev" if qid in dev_ids else "test",
                    "config": name,
                    **evaluate(q, candidates, chunks, scores[qid], counter),
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output / "results.csv", index=False)

    metrics = [
        c for c in frame.columns if c not in ("financebench_id", "split", "config")
    ]
    summary = frame.groupby(["split", "config"])[metrics].mean()
    summary.to_csv(args.output / "summary.csv")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
