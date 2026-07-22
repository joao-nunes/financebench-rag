from __future__ import annotations

from dataclasses import dataclass

from .metrics import (
    hit_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

@dataclass(slots=True)
class RetrievalMetrics:

    hit_rate: float

    recall_at_1: float

    recall_at_5: float

    recall_at_10: float

    precision_at_5: float

    mrr: float

    ndcg_at_10: float


class RetrievalEvaluator:

    def evaluate(
        self,
        ground_truth: list[str],
        retrieved: list[str],
    ) -> RetrievalMetrics:

        relevant = set(ground_truth)

        return RetrievalMetrics(

            hit_rate=hit_rate(
                relevant,
                retrieved,
            ),

            recall_at_1=recall_at_k(
                relevant,
                retrieved,
                1,
            ),

            recall_at_5=recall_at_k(
                relevant,
                retrieved,
                5,
            ),

            recall_at_10=recall_at_k(
                relevant,
                retrieved,
                10,
            ),

            precision_at_5=precision_at_k(
                relevant,
                retrieved,
                5,
            ),

            mrr=reciprocal_rank(
                relevant,
                retrieved,
            ),

            ndcg_at_10=ndcg_at_k(
                relevant,
                retrieved,
                10,
            ),
        )