from fastapi import APIRouter, Depends

from api.dependencies import get_rag_service
from api.schemas import ChatRequest, ChatResponse
from src.pipeline.service import RAGService

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    rag: RAGService = Depends(get_rag_service),
):
    return rag.answer(request.question)