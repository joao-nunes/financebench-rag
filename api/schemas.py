from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class Source(BaseModel):
    # Filing the chunk comes from, e.g. APPLE_2022_10K.
    document_id: str
    # 1-based PDF page, as shown in a PDF viewer.
    page: int | None
    score: float
    content: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]
    # Langfuse trace of this answer; None when tracing is off.
    trace_id: str | None = None
