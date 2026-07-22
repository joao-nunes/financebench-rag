from __future__ import annotations

from typing import Protocol

from .models import EvaluationResult

import time

from src.evaluation.models import (
    EvaluationResult,
    RetrievedDocument,
)

class RAGPipeline(Protocol):
    """
    Interface for any Retrieval-Augmented Generation pipeline.

    Implementations may use any retriever, reranker, LLM or prompting
    strategy, but they must expose a single method that receives a
    question and returns an EvaluationResult.
    """

    def invoke(self, question: str) -> EvaluationResult:
        """
        Execute the complete RAG pipeline.

        Parameters
        ----------
        question : str
            User question.

        Returns
        -------
        EvaluationResult
            Prediction together with the retrieved documents and
            evaluation metadata.
        """
        ...



class FinanceBenchRAGPipeline:
    """
    Concrete implementation of a RAG pipeline.
    """

    def __init__(
        self,
        retriever,
        generator,
    ):
        self.retriever = retriever
        self.generator = generator

    def invoke(
        self,
        question: str,
    ) -> EvaluationResult:

        start = time.perf_counter()

        retrieved_documents = self.retriever.retrieve(question)

        prediction = self.generator.generate(
            question=question,
            documents=retrieved_documents,
        )

        latency_ms = (time.perf_counter() - start) * 1000

        return EvaluationResult(
            question=question,
            prediction=prediction,
            retrieved_documents=retrieved_documents,
            latency_ms=latency_ms,
        )
    

class RetrievalPipeline(RAGPipeline):
    """
    Pipeline that performs retrieval only.

    Useful for benchmarking retrieval metrics without calling an LLM.
    """

    def __init__(self, retriever):
        self.retriever = retriever

    def invoke(
        self,
        question: str,
    ) -> EvaluationResult:

        start = time.perf_counter()

        docs = self.retriever.invoke(question)

        latency_ms = (time.perf_counter() - start) * 1000

        seen = set()
        retrieved_documents = []

        for doc in docs:

            document_id = doc.metadata["document_id"]

            if document_id in seen:
                continue

            seen.add(document_id)

            retrieved_documents.append(
                RetrievedDocument(
                    document_id=document_id,
                    score=doc.metadata.get("score"),
                    rank=len(retrieved_documents) + 1,
                    metadata=doc.metadata,
                )
            )

        return EvaluationResult(
            question=question,
            prediction="",          # No generation
            retrieved_documents=retrieved_documents,
            latency_ms=latency_ms,
        )