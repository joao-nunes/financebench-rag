from langchain_openai import ChatOpenAI

from src.config import (
    MAX_TOKENS,
    OPENAI_MODEL,
    TEMPERATURE,
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
