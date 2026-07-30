from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    document_id: str
    score: float
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
