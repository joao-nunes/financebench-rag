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
    # Langfuse trace of this answer; None when tracing is off.
    trace_id: str | None = None
