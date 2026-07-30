from __future__ import annotations

from dataclasses import asdict, dataclass

from src.evaluation.models import BaseEvaluator, EvaluationResult, EvaluationSample

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
    ndcg_at_5: float

    def to_dict(self):
        return asdict(self)


class RetrievalEvaluator(BaseEvaluator):

    def evaluate(
        self,
        sample: EvaluationSample,
        result: EvaluationResult,
    ):

        relevant = {sample.source_document}

        retrieved = self._aggregate_chunks(result.retrieved_chunks)
        reranked = self._aggregate_chunks(result.reranked_chunks)
        return RetrievalMetrics(
            hit_rate=hit_rate(
                relevant,
                retrieved,
            ),
            recall_at_1=recall_at_k(
                relevant,
                reranked,
                1,
            ),
            recall_at_5=recall_at_k(
                relevant,
                reranked,
                5,
            ),
            recall_at_10=recall_at_k(
                relevant,
                reranked,
                10,
            ),
            precision_at_5=precision_at_k(
                relevant,
                reranked,
                5,
            ),
            mrr=reciprocal_rank(
                relevant,
                reranked,
            ),
            ndcg_at_5=ndcg_at_k(
                relevant,
                reranked,
                5,
            ),
        )

    def _aggregate_chunks(self, chunks) -> list[str]:
        seen = set()
        documents = []

        for chunk in chunks:
            document_id = chunk.metadata["document_id"]

            if document_id not in seen:
                seen.add(document_id)
                documents.append(document_id)

        return documents
