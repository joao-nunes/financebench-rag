from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvaluationSample:
    question: str
    reference_answer: str
    source_document: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievedDocument:
    """
    One retrieved document/chunk.
    """

    document_id: str
    score: float
    rank: int

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationResult:
    """
    Output produced by the RAG system for a single question.

    Metrics are intentionally NOT stored here.
    Metrics belong to the evaluation modules.
    """

    question: str

    prediction: str

    retrieved_documents: list[RetrievedDocument]

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


@dataclass(slots=True)
class BenchmarkResult:
    """
    Stores the complete result of evaluating one sample.
    """

    sample: EvaluationSample

    result: EvaluationResult

    retrieval_metrics: RetrievalMetrics

    metadata: dict[str, Any] = field(default_factory=dict)