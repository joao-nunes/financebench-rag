#!/usr/bin/env python3
"""
Create a lightweight FinanceBench sample dataset.

The script:
1. Selects a small set of annual reports.
2. Copies their PDFs into data/sample/documents/.
3. Filters the FinanceBench metadata.
4. Filters the FinanceBench questions.

Example
-------
python scripts/create_sample_dataset.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE_DIR = ROOT / "data" / "financebench"
DEST_DIR = ROOT / "data" / "sample"

SOURCE_DOCS = SOURCE_DIR / "pdfs"
DEST_DOCS = DEST_DIR / "pdfs"

DOCUMENT_INFO_FILE = SOURCE_DIR / "data/financebench_document_information.jsonl"
QUESTIONS_FILE = SOURCE_DIR / "data/financebench_open_source.jsonl"

# Representative reports for the sample dataset
SAMPLE_DOCUMENTS = {
    "APPLE_2022_10K",
    "MICROSOFT_2023_10K",
    "AMAZON_2022_10K",
    "NVIDIA_2024_10K",
    "TESLA_2023_10K",
}

# =============================================================================


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")


def copy_documents(document_names: set[str]) -> int:
    """Copy selected PDFs to the sample directory."""

    DEST_DOCS.mkdir(parents=True, exist_ok=True)

    copied = 0

    for doc_name in sorted(document_names):
        pdf_name = f"{doc_name}.pdf"

        src = SOURCE_DOCS / pdf_name
        dst = DEST_DOCS / pdf_name

        if not src.exists():
            print(f"Warning: {pdf_name} not found.")
            continue

        shutil.copy2(src, dst)
        copied += 1

    return copied


def main() -> None:
    print("Loading FinanceBench...")

    document_info = load_jsonl(DOCUMENT_INFO_FILE)
    questions = load_jsonl(QUESTIONS_FILE)

    # -------------------------------------------------------------------------
    # Filter document metadata
    # -------------------------------------------------------------------------

    selected_documents = [
        doc for doc in document_info if doc["doc_name"] in SAMPLE_DOCUMENTS
    ]

    selected_doc_names = {doc["doc_name"] for doc in selected_documents}

    # -------------------------------------------------------------------------
    # Filter questions
    # -------------------------------------------------------------------------

    filtered_questions = [q for q in questions if q["doc_name"] in selected_doc_names]

    # -------------------------------------------------------------------------
    # Write files
    # -------------------------------------------------------------------------

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    save_jsonl(
        DEST_DIR / "data/financebench_document_information.jsonl",
        selected_documents,
    )

    save_jsonl(
        DEST_DIR / "data/financebench_open_source.jsonl",
        filtered_questions,
    )

    copied = copy_documents(selected_doc_names)

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print("\nDone!")
    print(f"Documents selected : {len(selected_documents)}")
    print(f"Questions selected : {len(filtered_questions)}")
    print(f"PDFs copied        : {copied}")
    print(f"Output directory   : {DEST_DIR}")

    print(SOURCE_DOCS)
    print(SOURCE_DOCS.exists())


if __name__ == "__main__":
    main()
