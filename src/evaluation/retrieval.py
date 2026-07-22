from __future__ import annotations

from dataclasses import dataclass, asdict

from src.evaluation.models import EvaluationResult, EvaluationSample

from .metrics import (
    unbounded_recall,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

@dataclass(slots=True)
class RetrievalMetrics:
    unbounded_recall: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    mrr: float
    ndcg_at_10: float

    def to_dict(self):
        return asdict(self)


class RetrievalEvaluator:

    def evaluate(
        self,
        sample: EvaluationSample,
        result: EvaluationResult,
    ):

        relevant = {sample.source_document}

        retrieved = [ doc.document_id for doc in result.retrieved_documents]

        return RetrievalMetrics(

            unbounded_recall=unbounded_recall(
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