from pprint import pprint

from src.evaluation.dataset import FinanceBenchDataset
from src.evaluation.models import (
    RetrievedDocument,
    EvaluationResult,
)
from src.evaluation.retrieval import RetrievalEvaluator


dataset = FinanceBenchDataset(
    "./data/financebench/data/financebench_open_source.jsonl"
)

sample = dataset.load()[0]

result = EvaluationResult(
    question=sample.question,
    prediction="dummy answer",
    retrieved_documents=[
        RetrievedDocument(
            document_id=sample.source_document,
            score=0.98,
            rank=1,
        ),
        RetrievedDocument(
            document_id="UNRELATED_DOC",
            score=0.87,
            rank=2,
        ),
    ],
    latency_ms=110,
)

evaluator = RetrievalEvaluator()
metrics = evaluator.evaluate(sample, result)

pprint(metrics)