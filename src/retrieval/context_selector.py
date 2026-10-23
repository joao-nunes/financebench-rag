from __future__ import annotations

import logging

import tiktoken

from src.retrieval.models import RetrievalResult

logger = logging.getLogger(__name__)


class TokenBudgetSelector:
    """
    Selects the top-ranked chunks that fit within a token budget.

    Chunks are taken in rank order until the next one would exceed the
    budget. The top chunk is always kept, even if it alone is over budget,
    so the generator never receives an empty context.
    """

    def __init__(
        self,
        max_tokens: int,
        encoding: str = "o200k_base",
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")

        self.max_tokens = max_tokens
        self._encoding = tiktoken.get_encoding(encoding)

    def count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text, disallowed_special=()))

    def select(self, chunks: list[RetrievalResult]) -> list[RetrievalResult]:
        selected = []
        used = 0

        for chunk in chunks:
            tokens = self.count_tokens(chunk.content)

            if selected and used + tokens > self.max_tokens:
                break

            selected.append(chunk)
            used += tokens

        logger.debug(
            "Selected %d/%d chunks (%d/%d tokens)",
            len(selected),
            len(chunks),
            used,
            self.max_tokens,
        )

        return selected
