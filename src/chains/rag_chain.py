from __future__ import annotations

import time

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough

from src.evaluation.models import (
    EvaluationResult,
    RetrievedDocument,
)
from src.evaluation.pipeline import RAGPipeline


def format_docs(docs: list[Document]) -> str:
    """
    Convert a list of retrieved Documents into a single context string.

    Parameters
    ----------
    docs : list[Document]
        Documents returned by the retriever.

    Returns
    -------
    str
        Concatenated document contents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_generation_chain(
    prompt: ChatPromptTemplate,
    llm: BaseChatModel,
):
    return (
        prompt
        | llm
        | StrOutputParser()
    )

def create_rag_chain(
    retriever,
    prompt: ChatPromptTemplate,
    llm: BaseChatModel,
):
    """
    Create the baseline LCEL Retrieval-Augmented Generation (RAG) pipeline.

    Pipeline:
        Question
            ↓
        Retriever
            ↓
        Format retrieved documents
            ↓
        Prompt template
            ↓
        Language model
            ↓
        String output parser

    Parameters
    ----------
    retriever
        LangChain retriever.

    prompt : ChatPromptTemplate
        Prompt template used to instruct the LLM.

    llm : BaseChatModel
        Chat language model.

    Returns
    -------
    Runnable
        Executable LCEL RAG chain.
    """

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


class LangChainRAGPipeline(RAGPipeline):
    """
    Concrete implementation of a Retrieval-Augmented Generation pipeline.

    Responsibilities
    ----------------
    1. Retrieve relevant documents.
    2. Generate an answer.
    3. Convert LangChain objects into EvaluationResult.
    """

    def __init__(
        self,
        retriever,
        reranker,
        prompt: ChatPromptTemplate,
        llm: BaseChatModel,
    ) -> None:

        self.retriever = retriever
        self.reranker = reranker

        self.chain = create_generation_chain(
            prompt=prompt,
            llm=llm,
        )

    def invoke(
        self,
        question: str,
    ) -> EvaluationResult:

        start = time.perf_counter()

        retrieved_docs = self.retriever.invoke(question)

        reranked_docs = self.reranker.rerank(
            question,
            retrieved_docs,
        )

        context = format_docs(reranked_docs)

        prediction = self.chain.invoke(
            {
                "question": question,
                "context": context,
            }
        )


        latency_ms = (time.perf_counter() - start) * 1000

        retrieved_documents = []

        for rank, doc in enumerate(reranked_docs, start=1):

            retrieved_documents.append(
                RetrievedDocument(
                    document_id=doc.metadata["document_id"],
                    text=doc.page_content,
                    score=doc.metadata.get("score", 0.0),
                    rank=rank,
                    metadata=doc.metadata,
                )
            )

        return EvaluationResult(
            question=question,
            prediction=prediction,
            retrieved_documents=retrieved_documents,
            latency_ms=latency_ms,
        )
