from contextlib import asynccontextmanager

from api.dependencies import rag_service
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.service import RAGService


@asynccontextmanager
async def lifespan(app):

    pipeline = RAGPipeline(...)

    global rag_service
    rag_service = RAGService(pipeline)

    yield

    rag_service = None