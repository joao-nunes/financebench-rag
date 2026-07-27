from __future__ import annotations

from typing import Protocol

from .models import EvaluationResult

import time

from src.evaluation.models import (
    EvaluationResult,
    RetrievedDocument,
    RetrievedChunk,
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
        reranker,
        generator,
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def invoke(
        self,
        question: str,
    ) -> EvaluationResult:

        start = time.perf_counter()

        retrieved_chunks = self.retriever.invoke(question)
        
        retrieved_chunks = [
                RetrievedChunk(
                    document_id=doc.metadata["document_id"],
                    page_content=doc.page_content,
                    score=doc.metadata.get("score"),
                    metadata=doc.metadata,
                )
                for doc in retrieved_chunks
        ]

        reranked_chunks = self.reranker.rerank(question, retrieved_chunks)

        reranked_chunks = [
            RetrievedChunk(
                document_id=doc.metadata["document_id"],
                page_content=doc.page_content,
                score=doc.score,      # or doc.metadata["score"], depending on your reranker
                metadata=doc.metadata,
            )
            for doc in reranked_chunks
            ]

        latency_ms = (time.perf_counter() - start) * 1000

        # Generation
        prediction = self.generator.invoke(
            {
                "question": question,
                "context": reranked_chunks,
            }
        )

        return EvaluationResult(
            question=question,
            prediction=prediction,
            retrieved_chunks=retrieved_chunks,
            reranked_chunks=reranked_chunks,
            latency_ms=latency_ms,
        )
    

class RetrievalPipeline(RAGPipeline):
    """
    Pipeline that performs retrieval only.

    Useful for benchmarking retrieval metrics without calling an LLM.
    """

    def __init__(self, retriever, reranker):
        self.retriever = retriever
        self.reranker = reranker

    def invoke(
        self,
        question: str,
    ) -> EvaluationResult:

        start = time.perf_counter()

        retrieved_chunks = self.retriever.invoke(question)
        
        retrieved_chunks = [
                RetrievedChunk(
                    document_id=doc.metadata["document_id"],
                    page_content=doc.page_content,
                    score=doc.metadata.get("score"),
                    metadata=doc.metadata,
                )
                for doc in retrieved_chunks
        ]

        reranked_chunks = self.reranker.rerank(question, retrieved_chunks)

        reranked_chunks = [
            RetrievedChunk(
                document_id=doc.metadata["document_id"],
                page_content=doc.page_content,
                score=doc.score,      # or doc.metadata["score"], depending on your reranker
                metadata=doc.metadata,
            )
            for doc in reranked_chunks
            ]

        latency_ms = (time.perf_counter() - start) * 1000


        return EvaluationResult(
            question=question,
            prediction="",          # No generation
            retrieved_chunks=retrieved_chunks,
            reranked_chunks=reranked_chunks,
            latency_ms=latency_ms,
        )