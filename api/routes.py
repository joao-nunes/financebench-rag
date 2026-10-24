import logging

from fastapi import APIRouter, Depends

from api.dependencies import get_pipeline
from api.schemas import ChatRequest, ChatResponse, Source
from src.config import MAX_SOURCES
from src.pipeline.rag_pipeline import RAGPipeline

logger = logging.getLogger(__name__)

router = APIRouter()


def _page_number(page: int | None) -> int | None:
    # Chunk metadata stores the 0-based page index from the PDF loader.
    return None if page is None else page + 1


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
                page=_page_number(doc.metadata.get("page")),
                score=doc.score,
                content=doc.content,
            )
            for doc in response.retrieved_documents[:MAX_SOURCES]
        ],
        trace_id=response.trace_id,
    )
