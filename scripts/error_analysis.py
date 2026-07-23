from pathlib import Path

import pandas as pd


def save_error_analysis(
    benchmark_results,
    output_dir: str | Path = "results",
):
    """
    Save retrieval benchmark results for later analysis.

    Creates

    - retrieval_results.csv
    - retrieval_errors.csv
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    rows = []
    error_rows = []

    for result in benchmark_results:

        sample = result.sample
        retrieved = result.result.retrieved_documents

        retrieved_ids = [
            doc.document_id
            for doc in retrieved
        ]

        retrieved_scores = [
            doc.score
            for doc in retrieved
        ]

        row = {

            "question": sample.question,

            "expected_document": sample.source_document,

            "unbounded_recall": result.retrieval_metrics.unbounded_recall,

            "recall_at_1": result.retrieval_metrics.recall_at_1,

            "recall_at_5": result.retrieval_metrics.recall_at_5,

            "mrr": result.retrieval_metrics.mrr,

            "ndcg_at_5": result.retrieval_metrics.ndcg_at_5,

            "retrieved_documents": " | ".join(
                retrieved_ids
            ),

            "retrieval_scores": " | ".join(
                [
                    f"{score:.4f}"
                    if score is not None
                    else "None"
                    for score in retrieved_scores
                ]
            ),
        }

        rows.append(row)

        #
        # Save failures separately
        #

        if not result.retrieval_metrics.unbounded_recall:

            error_rows.append(row)

    pd.DataFrame(rows).to_csv(
        output_dir / "retrieval_results.csv",
        index=False,
    )

    pd.DataFrame(error_rows).to_csv(
        output_dir / "retrieval_errors.csv",
        index=False,
    )

    print(
        f"Saved {len(rows)} benchmark results."
    )

    print(
        f"Saved {len(error_rows)} retrieval failures."
    )