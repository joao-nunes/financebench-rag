from src.retrieval.bm25 import BM25Index, reciprocal_rank_fusion


def test_bm25_ranks_documents_containing_rare_query_terms_first():
    texts = [
        "Apple reported net sales of 394,328 million",
        "Microsoft reported revenue growth in cloud",
        "Apple and Microsoft both reported results",
    ]

    ids, scores = BM25Index().build(texts).search("Apple net sales", k=3)

    assert ids[0] == 0
    assert list(scores) == sorted(scores, reverse=True)


def test_bm25_search_round_trips_through_save_and_load(tmp_path):
    index = BM25Index().build(["capital expenditure", "dividends paid"])
    index.save(tmp_path / "bm25.pkl")

    ids, _ = BM25Index.load(tmp_path / "bm25.pkl").search("dividends", k=1)

    assert ids.tolist() == [1]


def test_rrf_favours_items_ranked_high_in_both_lists():
    fused = reciprocal_rank_fusion([[1, 2, 3], [2, 4, 1]])

    assert fused[0] == 2
    assert set(fused) == {1, 2, 3, 4}
