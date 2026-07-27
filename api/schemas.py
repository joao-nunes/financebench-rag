from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    document_id: str
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]