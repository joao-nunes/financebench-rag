import pytest
from langchain_core.documents import Document

from src.config import CHUNK_SIZE
from src.ingestion.splitter import ChunkingConfig, split_documents

DOC_ID = "APPLE_2022_10K"


def make_page(page: int, text: str) -> Document:
    return Document(page_content=text, metadata={"document_id": DOC_ID, "page": page})


def sentences(prefix: str, n: int) -> str:
    # PyPDF output is line-structured, which is what lets chunks cross pages.
    return "\n".join(f"{prefix} sentence number {i}." for i in range(n))


@pytest.fixture
def pages():
    return [make_page(3, sentences("alpha", 40)), make_page(4, sentences("beta", 40))]


@pytest.mark.parametrize(
    "config",
    [
        ChunkingConfig(),
        ChunkingConfig(64, 8, "tokens"),
        ChunkingConfig(200, 20, "chars", scope="document"),
        ChunkingConfig(whole_page=True),
    ],
    ids=["chars", "tokens", "crosspage", "whole_page"],
)
def test_every_chunk_records_its_pages(pages, config):
    chunks = split_documents(pages, config)

    assert chunks
    for chunk in chunks:
        assert chunk.metadata["pages"]
        assert chunk.metadata["page"] == chunk.metadata["pages"][0]
        assert chunk.metadata["document_id"] == DOC_ID


def test_default_config_keeps_character_chunks(pages):
    chunks = split_documents(pages)

    assert len(chunks) > 2
    assert all(len(c.page_content) <= CHUNK_SIZE for c in chunks)
    assert all(len(c.metadata["pages"]) == 1 for c in chunks)


def test_crosspage_chunks_map_back_to_correct_pages(pages):
    chunks = split_documents(pages, ChunkingConfig(200, 0, "chars", scope="document"))

    for chunk in chunks:
        expected = set()
        if "alpha" in chunk.page_content:
            expected.add(3)
        if "beta" in chunk.page_content:
            expected.add(4)
        assert set(chunk.metadata["pages"]) == expected

    assert any(c.metadata["pages"] == [3, 4] for c in chunks)


def test_token_chunks_respect_token_limit(pages):
    from transformers import AutoTokenizer

    from src.config import EMBEDDING_MODEL

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    chunks = split_documents(pages, ChunkingConfig(64, 8, "tokens"))

    assert all(len(tokenizer.tokenize(c.page_content)) <= 64 for c in chunks)


def test_whole_page_skips_empty_pages(pages):
    chunks = split_documents(
        pages + [make_page(5, "   \n ")], ChunkingConfig(whole_page=True)
    )

    assert [c.metadata["pages"] for c in chunks] == [[3], [4]]
