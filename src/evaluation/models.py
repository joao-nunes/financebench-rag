from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar


from abc import ABC, abstractmethod

from src.retrieval.models import RetrievalResult


@dataclass(slots=True)
class EvaluationSample:
    question: str
    reference_answer: str
    source_document: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedDocument:
    document_id: str
    page_content: str
    score: float
    metadata: dict[str, Any]

    retrieved_chunks: list[RetrievalResult] = field(default_factory=list)


@dataclass(slots=True)
class EvaluationResult:
    """
    Output produced by the RAG system for a single question.

    Metrics are intentionally NOT stored here.
    Metrics belong to the evaluation modules.
    """

    question: str

    prediction: str

    retrieved_chunks: list[RetrievalResult]
    reranked_chunks: list[RetrievalResult]

    latency_ms: float

    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    precision_at_5: float
    mrr: float
    ndcg: float


MetricType = TypeVar("MetricType")


@dataclass
class BenchmarkResult(Generic[MetricType]):
    sample: EvaluationSample
    result: EvaluationResult
    metrics: MetricType


class BaseEvaluator(ABC):

    @abstractmethod
    def evaluate(
        self,
        sample: EvaluationSample,
        result: EvaluationResult,
    ): ...
