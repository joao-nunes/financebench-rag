from contextlib import asynccontextmanager

import api.dependencies as dependencies
from src.config import VECTORSTORE_DIR
from src.generation.openai_generator import OpenAIGenerator
from src.generation.simple_prompt_builder import SimplePromptBuilder
from src.indexing.embeddings import get_embedding_model
from src.indexing.faiss_store import FAISSStore
from src.logging_config import configure_logging
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.service import RAGService
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.faiss_retriever import FAISSRetriever


@asynccontextmanager
async def lifespan(app):

    configure_logging()

    embedding_model = get_embedding_model()

    vectorstore = FAISSStore()

    vectorstore.load(
        VECTORSTORE_DIR,
        embedding_model,
    )

    retriever = FAISSRetriever(vectorstore=vectorstore.store)
    reranker = CrossEncoderReranker()
    prompt_builder = SimplePromptBuilder()
    generator = OpenAIGenerator()

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        prompt_builder=prompt_builder,
        generator=generator,
    )

    dependencies.rag_service = RAGService(pipeline)

    yield

    dependencies.rag_service = None
