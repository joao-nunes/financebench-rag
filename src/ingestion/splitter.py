from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL

PAGE_SEPARATOR = "\n"


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """
    Describes how documents are split into chunks.

    Attributes:
        unit: "chars" measures chunk_size/chunk_overlap in characters,
            "tokens" in tokens of the embedding model's tokenizer.
        scope: "page" splits each page independently, "document"
            concatenates all pages of a document first so chunks can
            span page breaks (e.g. tables continued on the next page).
        whole_page: Emit every page as a single chunk, ignoring sizes.
    """

    chunk_size: int = CHUNK_SIZE
    chunk_overlap: int = CHUNK_OVERLAP
    unit: Literal["chars", "tokens"] = "chars"
    scope: Literal["page", "document"] = "page"
    whole_page: bool = False


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(EMBEDDING_MODEL)


def _build_splitter(config: ChunkingConfig) -> RecursiveCharacterTextSplitter:
    if config.unit == "tokens":
        return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            _tokenizer(),
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
    )


def _merge_pages(pages: list[Document]) -> tuple[Document, list[tuple[int, int, int]]]:
    """
    Concatenate the pages of one document.

    Returns the merged document and a list of (start, end, page) character
    spans so chunks can be mapped back to the pages they cover.
    """

    pages = sorted(pages, key=lambda d: d.metadata.get("page", 0))

    parts = []
    spans = []
    offset = 0

    for page in pages:
        text = page.page_content
        spans.append((offset, offset + len(text), page.metadata.get("page", 0)))
        parts.append(text)
        offset += len(text) + len(PAGE_SEPARATOR)

    metadata = {
        k: v for k, v in pages[0].metadata.items() if k not in ("page", "page_label")
    }

    return Document(page_content=PAGE_SEPARATOR.join(parts), metadata=metadata), spans


def _pages_in_range(
    spans: list[tuple[int, int, int]], start: int, end: int
) -> list[int]:
    return [page for s, e, page in spans if s < end and e > start]


def split_documents(
    documents: list[Document],
    config: ChunkingConfig | None = None,
) -> list[Document]:
    """
    Split a list of LangChain Documents (one per PDF page) into chunks.

    Every chunk carries a "pages" metadata entry listing the page numbers
    it covers, and "page" set to the first of them.

    Args:
        documents: List of input documents.
        config: Chunking strategy. Defaults to character-based splitting
            using CHUNK_SIZE / CHUNK_OVERLAP from the config.

    Returns:
        List of chunked LangChain Documents.
    """

    config = config or ChunkingConfig()

    if config.whole_page:
        chunks = [
            Document(page_content=doc.page_content, metadata=dict(doc.metadata))
            for doc in documents
            if doc.page_content.strip()
        ]
        for chunk in chunks:
            chunk.metadata["pages"] = [chunk.metadata.get("page", 0)]
        return chunks

    text_splitter = _build_splitter(config)

    if config.scope == "page":
        chunks = text_splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["pages"] = [chunk.metadata.get("page", 0)]
        return chunks

    by_document: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        by_document[doc.metadata["document_id"]].append(doc)

    chunks = []
    for pages in by_document.values():
        merged, spans = _merge_pages(pages)

        for chunk in text_splitter.split_documents([merged]):
            start = chunk.metadata["start_index"]
            covered = _pages_in_range(spans, start, start + len(chunk.page_content))
            chunk.metadata["pages"] = covered
            chunk.metadata["page"] = covered[0] if covered else 0
            chunks.append(chunk)

    return chunks
