from contextlib import asynccontextmanager

import api.dependencies as dependencies
from src.config import (
    CONTEXT_TOKEN_BUDGET,
    DOCUMENT_INFO_FILE,
    RERANKER_MODEL,
    VECTORSTORE_DIR,
)
from src.generation.openai_generator import OpenAIGenerator
from src.generation.simple_prompt_builder import SimplePromptBuilder
from src.indexing.embeddings import get_embedding_model
from src.indexing.faiss_store import FAISSStore
from src.logging_config import configure_logging
from src.observability import flush as flush_traces
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.service import RAGService
from src.retrieval.context_selector import TokenBudgetSelector
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.filing_filter import FilingFilter


@asynccontextmanager
async def lifespan(app):

    configure_logging()

    embedding_model = get_embedding_model()

    vectorstore = FAISSStore()

    vectorstore.load(
        VECTORSTORE_DIR,
        embedding_model,
    )

    filing_filter = FilingFilter.from_document_information(
        vectorstore.document_ids(),
        DOCUMENT_INFO_FILE,
    )
    retriever = FAISSRetriever(
        vectorstore=vectorstore.store,
        filing_filter=filing_filter,
    )
    # Keep the full ranking; the token budget decides how much reaches the LLM.
    reranker = CrossEncoderReranker(model_name=RERANKER_MODEL, top_n=None)
    context_selector = TokenBudgetSelector(CONTEXT_TOKEN_BUDGET)
    prompt_builder = SimplePromptBuilder()
    generator = OpenAIGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        prompt_builder=prompt_builder,
        generator=generator,
        context_selector=context_selector,
    )

    dependencies.rag_service = RAGService(pipeline)

    yield

    dependencies.rag_service = None
    flush_traces()
