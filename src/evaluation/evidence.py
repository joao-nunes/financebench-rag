"""
Evidence-page retrieval metrics.

A chunk is relevant when it covers one of the (document, page) pairs that
FinanceBench lists as evidence for a question.
"""

from __future__ import annotations

K_VALUES = (1, 3, 5, 10)
TOKEN_BUDGETS = (1000, 2000, 4000)


def evidence_pages(question: dict) -> set[tuple[str, int]]:
    return {(e["doc_name"], e["evidence_page_num"]) for e in question["evidence"]}


def chunk_pages(chunk) -> set[tuple[str, int]]:
    doc = chunk.metadata["document_id"]
    return {(doc, page) for page in chunk.metadata["pages"]}


def score_ranking(
    ranked: list,
    relevant: set,
    n_tokens: list[int],
    doc_name: str,
    k_values: tuple[int, ...] = K_VALUES,
    token_budgets: tuple[int, ...] = TOKEN_BUDGETS,
) -> dict:
    """Evidence-page metrics for one ranked list of chunks."""

    covered = [chunk_pages(c) & relevant for c in ranked]
    metrics = {}

    for k in k_values:
        hit = set().union(*covered[:k]) if covered[:k] else set()
        metrics[f"page_recall@{k}"] = len(hit) / len(relevant)

    first = next((i for i, c in enumerate(covered) if c), None)
    metrics["mrr"] = 0.0 if first is None else 1.0 / (first + 1)

    for budget in token_budgets:
        used, hit = 0, set()
        for c, n in zip(covered, n_tokens, strict=True):
            if used + n > budget:
                break
            used += n
            hit |= c
        metrics[f"page_recall@{budget}tok"] = len(hit) / len(relevant)

    metrics["doc_hit@5"] = float(
        any(c.metadata["document_id"] == doc_name for c in ranked[:5])
    )

    return metrics
