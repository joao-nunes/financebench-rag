from src.evaluation.dataset import FinanceBenchDataset
from src.evaluation.models import (
    RetrievedDocument,
    EvaluationResult,
)
from src.evaluation.retrieval import RetrievalEvaluator


def test_pipeline():

    dataset = FinanceBenchDataset(
        "./data/financebench/data/financebench_open_source.jsonl"
    )

    sample = dataset.load()[0]

    retrieved = [

        RetrievedDocument(
            document_id=sample.source_document,
            score=0.98,
            rank=1,
        ),

        RetrievedDocument(
            document_id="OTHER_DOC",
            score=0.91,
            rank=2,
        ),
    ]

    result = EvaluationResult(

        question=sample.question,

        prediction=sample.reference_answer,

        retrieved_documents=retrieved,

        latency_ms=95,
    )

    evaluator = RetrievalEvaluator()

    metrics = evaluator.evaluate(
        sample,
        result,
    )

    assert metrics.hit_rate == 1
    assert metrics.recall_at_1 == 1
    assert metrics.mrr == 1