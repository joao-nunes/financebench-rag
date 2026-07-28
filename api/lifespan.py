from contextlib import asynccontextmanager

import api.dependencies as dependencies
from src.indexing.faiss_store import FAISSStore
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.service import RAGService
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.reranking import CrossEncoderReranker
from src.generation.prompt_builder import SimplePromptBuilder
from src.generation.openai_generator import OpenAIGenerator

from pathlib import Path
from src.indexing.embeddings import get_embedding_model

@asynccontextmanager
async def lifespan(app):

    VECTORSTORE_PATH = Path("./data/vectorstore")
    embedding_model = get_embedding_model()

    vectorstore = FAISSStore()

    vectorstore.load(
        VECTORSTORE_PATH,
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