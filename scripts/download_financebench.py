#!/usr/bin/env python3

from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://raw.githubusercontent.com/patronus-ai/financebench/main"

DATA_FILES = [
    "financebench_open_source.jsonl",
    "financebench_document_information.jsonl",
]

DATA_DIR = Path("data")
PDF_DIR = DATA_DIR / "pdfs"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        print(f"✓ {destination} already exists")
        return

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with (
        open(destination, "wb") as f,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)

    print("Downloading metadata...")

    for filename in DATA_FILES:
        download(
            f"{BASE_URL}/data/{filename}",
            DATA_DIR / filename,
        )

    print("Loading document list...")

    import pandas as pd

    metadata = pd.read_json(
        DATA_DIR / "financebench_document_information.jsonl",
        lines=True,
    )

    print(f"Downloading {len(metadata)} PDFs...")

    for _, row in metadata.iterrows():
        download(
            row["doc_link"],
            PDF_DIR / f"{row['doc_name']}.pdf",
        )

    print("Done!")


if __name__ == "__main__":
    main()
