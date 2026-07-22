from src.evaluation.models import (
    EvaluationSample,
    EvaluationResult,
    RetrievedDocument,
)

from src.evaluation.retrieval import RetrievalEvaluator


def test_retrieval_evaluator():

    sample = EvaluationSample(
        question="What was Apple's revenue?",
        reference_answer="$394B",
        source_document="APPLE_2022_10K",
    )

    retrieved = [

        RetrievedDocument(
            document_id="APPLE_2021_10K",
            score=0.91,
            rank=1,
        ),

        RetrievedDocument(
            document_id="APPLE_2022_10K",
            score=0.90,
            rank=2,
        ),

        RetrievedDocument(
            document_id="MSFT_2022_10K",
            score=0.83,
            rank=3,
        ),
    ]

    result = EvaluationResult(

        question=sample.question,

        prediction="$394B",

        retrieved_documents=retrieved,

        latency_ms=120,
    )

    evaluator = RetrievalEvaluator()

    metrics = evaluator.evaluate(
        sample,
        result,
    )

    assert metrics.hit_rate == 1
    assert metrics.recall_at_1 == 0
    assert metrics.recall_at_5 == 1
    assert metrics.mrr == 0.5