from __future__ import annotations

import math


def hit_rate(relevant: set[str], retrieved: list[str]) -> float:
    """
    Returns 1 if at least one relevant document was retrieved.
    """
    return float(any(doc in relevant for doc in retrieved))


def recall_at_k(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> float:
    """
    Recall@k
    """

    retrieved = retrieved[:k]

    if not relevant:
        return 0.0

    return len(relevant.intersection(retrieved)) / len(relevant)


def precision_at_k(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> float:

    retrieved = retrieved[:k]

    if len(retrieved) == 0:
        return 0.0

    return len(relevant.intersection(retrieved)) / len(retrieved)


def reciprocal_rank(
    relevant: set[str],
    retrieved: list[str],
) -> float:
    """
    Reciprocal Rank.
    """

    for rank, document in enumerate(retrieved, start=1):

        if document in relevant:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    relevant: set[str],
    retrieved: list[str],
    k: int,
) -> float:

    dcg = 0.0

    for i, document in enumerate(retrieved[:k]):

        if document in relevant:
            dcg += 1 / math.log2(i + 2)

    ideal = min(len(relevant), k)

    if ideal == 0:
        return 0.0

    idcg = sum(
        1 / math.log2(i + 2)
        for i in range(ideal)
    )

    return dcg / idcg