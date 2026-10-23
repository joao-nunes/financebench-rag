import pytest

from src.retrieval.context_selector import TokenBudgetSelector
from src.retrieval.models import RetrievalResult


def _chunk(text):
    return RetrievalResult(document_id="d", content=text, score=0.0, metadata={})


def test_keeps_rank_order_and_stops_at_first_chunk_over_budget():
    selector = TokenBudgetSelector(max_tokens=10)
    chunks = [_chunk("one two three"), _chunk("four five six"), _chunk("x " * 20)]

    selected = selector.select(chunks)

    assert selected == chunks[:2]


def test_always_keeps_top_chunk_even_if_over_budget():
    selector = TokenBudgetSelector(max_tokens=5)
    chunks = [_chunk("word " * 50), _chunk("short")]

    assert selector.select(chunks) == chunks[:1]


def test_rejects_non_positive_budget():
    with pytest.raises(ValueError):
        TokenBudgetSelector(max_tokens=0)
