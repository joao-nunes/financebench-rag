from langchain_openai import ChatOpenAI

from src.config import (
    OPENAI_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
)


def get_llm() -> ChatOpenAI:
    """
    Instantiate and return the language model used by the RAG pipeline.

    Returns
    -------
    ChatOpenAI
        Configured ChatOpenAI model.
    """

    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )