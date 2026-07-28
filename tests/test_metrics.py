import pytest

from src.evaluation.metrics import (
    hit_rate,
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    ndcg_at_k,
)


def test_perfect_retrieval():

    relevant = {"APPLE_2022_10K"}

    retrieved = [
        "APPLE_2022_10K",
        "APPLE_2021_10K",
        "MSFT_2022_10K",
    ]

    assert hit_rate(relevant, retrieved) == 1
    assert recall_at_k(relevant, retrieved, 1) == 1
    assert recall_at_k(relevant, retrieved, 5) == 1
    assert precision_at_k(relevant, retrieved, 1) == 1
    assert reciprocal_rank(relevant, retrieved) == 1
    assert ndcg_at_k(relevant, retrieved, 10) == pytest.approx(1.0)


def test_second_position():

    relevant = {"APPLE_2022_10K"}

    retrieved = [
        "APPLE_2021_10K",
        "APPLE_2022_10K",
        "MSFT_2022_10K",
    ]

    assert hit_rate(relevant, retrieved) == 1
    assert recall_at_k(relevant, retrieved, 1) == 0
    assert recall_at_k(relevant, retrieved, 5) == 1
    assert reciprocal_rank(relevant, retrieved) == pytest.approx(0.5)


def test_document_not_found():

    relevant = {"APPLE_2022_10K"}

    retrieved = [
        "TESLA_2023_10K",
        "GOOGLE_2023_10K",
    ]

    assert hit_rate(relevant, retrieved) == 0
    assert recall_at_k(relevant, retrieved, 5) == 0
    assert reciprocal_rank(relevant, retrieved) == 0
    assert ndcg_at_k(relevant, retrieved, 10) == 0