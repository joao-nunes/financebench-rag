import pytest

from src.exceptions import PromptBuildError, RerankingError, RetrievalError
from src.generation.generator import Generator
from src.generation.prompt_builder import PromptBuilder
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.models import RetrievalResult
from src.retrieval.reranker import Reranker
from src.retrieval.retriever import Retriever


class FakeRetriever(Retriever):
    def __init__(self):
        self.received_query = None

    def retrieve(self, query: str) -> list[RetrievalResult]:
        self.received_query = query
        return [
            RetrievalResult(
                document_id="doc_1",
                content="Apple reported record quarterly revenue.",
                score=0.92,
                metadata={"source": "10-K"},
            ),
            RetrievalResult(
                document_id="doc_2",
                content="Revenue increased by 8% year-over-year.",
                score=0.87,
                metadata={"source": "10-Q"},
            ),
        ]


class FailingRetriever(Retriever):
    def retrieve(self, query: str) -> list[RetrievalResult]:
        raise RetrievalError("Failed to retrieve relevant chunks.")


class FakeReranker(Reranker):
    def __init__(self):
        self.received_query = None
        self.received_chunks = None

    def rerank(
        self,
        query: str,
        chunks: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        self.received_query = query
        self.received_chunks = chunks
        return chunks


class FailingReranker(Reranker):
    def rerank(
        self, query: str, chunks: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        raise RerankingError("Failed to rerank relevant chunks.")


class FakePromptBuilder(PromptBuilder):
    def __init__(self):
        self.received_question = None
        self.received_context = None

    def build(self, question, context):
        self.received_question = question
        self.received_context = context
        return "PROMPT"


class FailingPromptBuilder(PromptBuilder):
    def build(self, question, context):
        raise PromptBuildError("Failed to build prompt.")


class FakeGenerator(Generator):
    def __init__(self):
        self.received_prompt = None

    def generate(self, prompt: str) -> str:
        self.received_prompt = prompt
        return "Generated answer"


def test_pipeline_returns_generator_answer():
    fake_retriever = FakeRetriever()
    fake_reranker = FakeReranker()
    fake_prompt_builder = FakePromptBuilder()
    fake_generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=fake_retriever,
        reranker=fake_reranker,
        prompt_builder=fake_prompt_builder,
        generator=fake_generator,
    )

    question = "What is FinanceBench?"
    response = pipeline.answer(question)
    assert response.answer == "Generated answer"

    assert fake_retriever.received_query == question
    assert fake_reranker.received_query == question
    assert fake_prompt_builder.received_question == question
    assert fake_generator.received_prompt == "PROMPT"

    assert response.answer == "Generated answer"

    assert response.metrics.retrieved_documents == 2
    assert response.metrics.reranked_documents == 2

    assert response.metrics.pipeline_time > 0


def test_pipeline_raises_retrieval_error():

    failing_retriever = FailingRetriever()
    fake_reranker = FakeReranker()
    fake_prompt_builder = FakePromptBuilder()
    fake_generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=failing_retriever,
        reranker=fake_reranker,
        prompt_builder=fake_prompt_builder,
        generator=fake_generator,
    )

    with pytest.raises(RetrievalError):
        pipeline.answer("What is FinanceBench?")


def test_pipeline_raises_reranking_error():
    retriever = FakeRetriever()
    failing_reranker = FailingReranker()
    fake_prompt_builder = FakePromptBuilder()
    fake_generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=failing_reranker,
        prompt_builder=fake_prompt_builder,
        generator=fake_generator,
    )

    with pytest.raises(RerankingError):
        pipeline.answer("What is FinanceBench?")


def test_pipeline_raises_prompt_build_error():
    retriever = FakeRetriever()
    fake_reranker = FakeReranker()
    failing_prompt_builder = FailingPromptBuilder()
    fake_generator = FakeGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=fake_reranker,
        prompt_builder=failing_prompt_builder,
        generator=fake_generator,
    )

    with pytest.raises(PromptBuildError):
        pipeline.answer("What is FinanceBench?")


class FakeSelector:
    def select(self, chunks):
        return chunks[:1]


def test_pipeline_passes_only_selected_chunks_to_prompt():
    prompt_builder = FakePromptBuilder()

    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        prompt_builder=prompt_builder,
        generator=FakeGenerator(),
        context_selector=FakeSelector(),
    )

    response = pipeline.answer("What is FinanceBench?")

    assert [c.document_id for c in prompt_builder.received_context] == ["doc_1"]
    assert [c.document_id for c in response.retrieved_documents] == ["doc_1"]
    assert response.metrics.context_documents == 1


def test_pipeline_records_langfuse_trace(monkeypatch):
    from langfuse import Langfuse
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from src import observability

    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    exporter = InMemorySpanExporter()
    Langfuse(span_exporter=exporter)
    observability._client.cache_clear()

    pipeline = RAGPipeline(
        retriever=FakeRetriever(),
        reranker=FakeReranker(),
        prompt_builder=FakePromptBuilder(),
        generator=FakeGenerator(),
    )
    response = pipeline.answer("What is FinanceBench?")
    observability.flush()
    observability._client.cache_clear()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    root = spans["rag-pipeline"]

    assert response.trace_id == format(root.context.trace_id, "032x")
    assert spans["retrieval"].parent.span_id == root.context.span_id
    assert spans["reranking"].parent.span_id == root.context.span_id
