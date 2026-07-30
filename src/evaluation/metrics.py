from __future__ import annotations

import math
from abc import ABC, abstractmethod

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from src.evaluation.models import (
    EvaluationResult,
    EvaluationSample,
)


class GenerationMetric(ABC):
    @abstractmethod
    def compute(
        self,
        sample: EvaluationSample,
        result: EvaluationResult,
    ) -> float: ...


def hit_rate(
    relevant: set[str],
    retrieved: list[str],
    k: int | None = None,
) -> float:
    """
    Hit Rate (Hit@k).

    Returns 1.0 if at least one relevant document is retrieved,
    otherwise returns 0.0.

    If k is provided, only the top-k retrieved documents are considered.
    """

    if k is not None:
        retrieved = retrieved[:k]

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

    idcg = sum(1 / math.log2(i + 2) for i in range(ideal))

    return dcg / idcg


class GenerationScores(BaseModel):
    context_recall: float = Field(description="Score between 0 and 1.")

    faithfulness: float = Field(description="Score between 0 and 1.")

    answer_correctness: float = Field(description="Score between 0 and 1.")


class AggregatedGenerationMetrics(GenerationMetric):

    def __init__(
        self,
        llm: BaseChatModel,
    ):

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are evaluating a Retrieval-Augmented Generation (RAG) system.

Evaluate the system using the following three metrics.

1. Context Recall

Determine whether the retrieved context contains enough information 
to answer the question correctly.

2. Faithfulness

Determine whether every factual statement in the generated 
answer is supported by the retrieved context.

3. Answer Correctness

Determine whether the generated answer correctly answers 
the question compared to the reference answer.

Each score must be between 0 and 1.

Do not explain your reasoning.
Return only the structured output.
""",
                ),
                (
                    "human",
                    """
Question:
{question}

Reference Answer:
{reference_answer}

Retrieved Context:
{context}

Generated Answer:
{generated_answer}
""",
                ),
            ]
        )

        structured_llm = llm.with_structured_output(GenerationScores)

        self._chain = prompt | structured_llm

    def compute(
        self,
        sample: EvaluationSample,
        result: EvaluationResult,
    ) -> float:

        context = "\n\n".join(chunk.page_content for chunk in result.reranked_chunks)

        scores = self._chain.invoke(
            {
                "question": sample.question,
                "reference_answer": sample.reference_answer,
                "context": context,
                "generated_answer": result.prediction,
            }
        )

        return scores
