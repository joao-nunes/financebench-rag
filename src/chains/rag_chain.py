from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.evaluation.models import (
    RetrievedChunk,
)


def format_context(chunks: list[RetrievedChunk]) -> str:
    """
    Convert a list of retrieved chunks into a single context string.

    Parameters
    ----------
    docs : list[Document]
        Documents returned by the retriever.

    Returns
    -------
    str
        Concatenated document contents.
    """

    MAX_CONTEXT_CHARS = 5000

    context = "\n\n".join(chunk.page_content for chunk in chunks)

    context = context[:MAX_CONTEXT_CHARS]

    return context


def create_generation_chain(
    prompt: ChatPromptTemplate,
    llm: BaseChatModel,
):

    generation_chain = (
        {
            "question": RunnableLambda(lambda x: x["question"]),
            "context": RunnableLambda(lambda x: format_context(x["context"])),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return generation_chain
