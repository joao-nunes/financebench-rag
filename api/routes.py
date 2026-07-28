from fastapi import APIRouter, Depends

from api.dependencies import get_rag_service
from api.schemas import ChatRequest, ChatResponse
from src.pipeline.service import RAGService
from api.schemas import Source

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    rag: RAGService = Depends(get_rag_service),
):
    response = rag.answer(request.question)

    return ChatResponse(
        answer=response.answer,
        sources=[
            Source(
                document_id=doc.document_id,
                score=doc.score,
                content=doc.content,
            )
            for doc in response.retrieved_documents
        ],
)