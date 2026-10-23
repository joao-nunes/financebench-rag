import pytest
from langchain_core.documents import Document

from src.evaluation.evidence import evidence_pages, score_ranking

DOC = "APPLE_2022_10K"


def chunk(*pages: int, doc: str = DOC) -> Document:
    return Document(
        page_content="", metadata={"document_id": doc, "pages": list(pages)}
    )


def test_recall_and_mrr_when_evidence_ranked_third():
    ranked = [chunk(1), chunk(2), chunk(10), chunk(11)]

    metrics = score_ranking(ranked, {(DOC, 10)}, [10] * 4, DOC)

    assert metrics["page_recall@1"] == 0.0
    assert metrics["page_recall@3"] == 1.0
    assert metrics["mrr"] == pytest.approx(1 / 3)
    assert metrics["doc_hit@5"] == 1.0


def test_partial_recall_with_two_evidence_pages():
    relevant = evidence_pages(
        {
            "evidence": [
                {"doc_name": DOC, "evidence_page_num": 10},
                {"doc_name": DOC, "evidence_page_num": 20},
            ]
        }
    )
    ranked = [chunk(10), chunk(5), chunk(6)]

    metrics = score_ranking(ranked, relevant, [10] * 3, DOC)

    assert metrics["page_recall@3"] == 0.5


def test_same_page_in_other_document_is_not_relevant():
    ranked = [chunk(10, doc="APPLE_2021_10K")]

    metrics = score_ranking(ranked, {(DOC, 10)}, [10], DOC)

    assert metrics["page_recall@1"] == 0.0
    assert metrics["mrr"] == 0.0
    assert metrics["doc_hit@5"] == 0.0


def test_token_budget_excludes_chunk_that_overflows():
    ranked = [chunk(1), chunk(10)]

    metrics = score_ranking(
        ranked, {(DOC, 10)}, [600, 500], DOC, token_budgets=(1000, 2000)
    )

    assert metrics["page_recall@1000tok"] == 0.0
    assert metrics["page_recall@2000tok"] == 1.0
