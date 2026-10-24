#!/usr/bin/env python
"""
Compare chunking strategies on FinanceBench using evidence-page relevance.

A retrieved chunk is relevant when it covers one of the pages that
FinanceBench lists as evidence for the question. This is much stricter
than the document-level metrics in benchmark_retrieval.py, which only
check whether any chunk of the right filing was retrieved.

For every strategy the script:
    1. chunks the page cache (data/cache/documents.pkl)
    2. embeds the chunks (cached under --workdir)
    3. runs dense retrieval (top --top-k) + cross-encoder reranking
    4. reports evidence-page metrics at fixed k and at fixed token budgets

Examples
--------
# Strategy search on the filings referenced by the questions
python scripts/chunking_experiment.py --split dev

# Validate a strategy on the full corpus
python scripts/chunking_experiment.py --corpus full --strategies recursive_500 token_256
"""

from __future__ import annotations

import os

# faiss, torch and scikit-learn each bundle their own libomp; on macOS the
# OpenMP worker threads then segfault. GPU work does not need OpenMP.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.config import DATA_DIR, EMBEDDING_DEVICE, EMBEDDING_MODEL
from src.evaluation.evidence import chunk_pages, evidence_pages, score_ranking
from src.evaluation.financebench import load_questions
from src.ingestion.helpers import enrich_chunk
from src.ingestion.splitter import ChunkingConfig, split_documents

STRATEGIES = {
    "recursive_500": ChunkingConfig(500, 100, "chars"),
    "recursive_1000": ChunkingConfig(1000, 200, "chars"),
    "recursive_2000": ChunkingConfig(2000, 400, "chars"),
    "token_128": ChunkingConfig(128, 16, "tokens"),
    "token_256": ChunkingConfig(256, 32, "tokens"),
    "token_448": ChunkingConfig(448, 64, "tokens"),
    "token_256_crosspage": ChunkingConfig(256, 32, "tokens", scope="document"),
    "token_448_crosspage": ChunkingConfig(448, 64, "tokens", scope="document"),
    "whole_page": ChunkingConfig(whole_page=True),
}

DOCUMENTS_CACHE = DATA_DIR / "cache/documents.pkl"

RERANKER_MODEL = "BAAI/bge-reranker-base"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--corpus", choices=["evidence", "full"], default="evidence")
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--top-k", type=int, default=75)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument(
        "--workdir", type=Path, default=DATA_DIR / "experiments/chunking"
    )
    parser.add_argument("--output", type=Path, default=Path("experiments/chunking"))
    return parser.parse_args()


def load_pages(corpus: str) -> list:
    with DOCUMENTS_CACHE.open("rb") as f:
        pages = pickle.load(f)

    if corpus == "evidence":
        keep = {q["doc_name"] for q in load_questions()}
        pages = [p for p in pages if p.metadata["document_id"] in keep]

    return pages


def build_chunks(name: str, pages: list, workdir: Path) -> list:
    path = workdir / "chunks.pkl"

    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)

    start = time.perf_counter()
    chunks = split_documents(pages, STRATEGIES[name])
    chunks = [enrich_chunk(chunk) for chunk in chunks]
    print(f"[{name}] {len(chunks):,} chunks in {time.perf_counter() - start:.0f}s")

    with path.open("wb") as f:
        pickle.dump(chunks, f)

    return chunks


def build_embeddings(name: str, chunks: list, model, workdir: Path) -> np.ndarray:
    path = workdir / "embeddings.npy"

    if path.exists():
        return np.load(path)

    # Sorting by length minimises padding inside each batch.
    order = np.argsort([len(c.page_content) for c in chunks])
    start = time.perf_counter()
    vectors = model.encode(
        [chunks[i].page_content for i in order],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    print(f"[{name}] embedded in {time.perf_counter() - start:.0f}s")

    embeddings = np.empty_like(vectors)
    embeddings[order] = vectors
    np.save(path, embeddings)

    return embeddings


def evaluate(name, chunks, embeddings, questions, embedder, reranker, tokenizer, top_k):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    queries = embedder.encode(
        # Matches production: HuggingFaceEmbeddings adds no query instruction.
        [q["question"] for q in questions],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    _, neighbours = index.search(queries, top_k)

    rows = []
    for q, ids in tqdm(
        zip(questions, neighbours, strict=True), total=len(questions), desc=name
    ):
        candidates = [chunks[i] for i in ids if i >= 0]
        relevant = evidence_pages(q)

        dense_tokens = [len(tokenizer.tokenize(c.page_content)) for c in candidates]
        dense = score_ranking(candidates, relevant, dense_tokens, q["doc_name"])

        row = {
            "financebench_id": q["financebench_id"],
            "question_type": q["question_type"],
            "dense_recall@top_k": float(
                any(chunk_pages(c) & relevant for c in candidates)
            ),
            "dense_page_recall@5": dense["page_recall@5"],
            "dense_page_recall@2000tok": dense["page_recall@2000tok"],
        }

        if reranker is not None:
            scores = reranker.predict(
                [(q["question"], c.page_content) for c in candidates],
                batch_size=32,
                show_progress_bar=False,
            )
            order = np.argsort(-scores)
            reranked = [candidates[i] for i in order]
            tokens = [dense_tokens[i] for i in order]
            row.update(score_ranking(reranked, relevant, tokens, q["doc_name"]))

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    questions = load_questions(args.split)
    pages = load_pages(args.corpus)
    print(f"{len(questions)} questions | {len(pages):,} pages ({args.corpus} corpus)")

    embedder = SentenceTransformer(EMBEDDING_MODEL, device=_device())
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    reranker = (
        None if args.no_rerank else CrossEncoder(RERANKER_MODEL, device=_device())
    )

    run_dir = args.output / f"{args.corpus}_{args.split}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for name in args.strategies:
        workdir = args.workdir / args.corpus / name
        workdir.mkdir(parents=True, exist_ok=True)

        chunks = build_chunks(name, pages, workdir)
        embeddings = build_embeddings(name, chunks, embedder, workdir)

        results = evaluate(
            name,
            chunks,
            embeddings,
            questions,
            embedder,
            reranker,
            tokenizer,
            args.top_k,
        )
        results.to_csv(run_dir / f"{name}.csv", index=False)

        lengths = [
            len(tokenizer.tokenize(c.page_content))
            for c in chunks[:: max(1, len(chunks) // 5000)]
        ]
        summary.append(
            {
                "strategy": name,
                **asdict(STRATEGIES[name]),
                "n_chunks": len(chunks),
                "median_tokens": float(np.median(lengths)),
                "pct_over_512_tokens": float(np.mean(np.array(lengths) > 512)),
                **results.drop(columns=["financebench_id", "question_type"])
                .mean()
                .to_dict(),
            }
        )

        frame = pd.DataFrame(summary).set_index("strategy")
        frame.to_csv(run_dir / "summary.csv")
        print(frame.round(3).to_string())


def _device() -> str:
    if EMBEDDING_DEVICE != "auto":
        return EMBEDDING_DEVICE

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


if __name__ == "__main__":
    main()
