import logging

from fastapi import APIRouter, Depends

from api.dependencies import get_pipeline
from api.schemas import ChatRequest, ChatResponse, Source
from src.config import MAX_SOURCES
from src.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    response = pipeline.answer(request.question)

    return ChatResponse(
        answer=response.answer,
        sources=[
            Source(
                document_id=doc.document_id,
                score=doc.score,
                content=doc.content,
            )
            for doc in response.retrieved_documents[:MAX_SOURCES]
        ],
        trace_id=response.trace_id,
    )
