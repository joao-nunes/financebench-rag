from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.exceptions import (
    GenerationError,
    PromptBuildError,
    RAGException,
    RerankingError,
    RetrievalError,
)

# Status code and message returned for each pipeline failure.
ERRORS: dict[type[RAGException], tuple[int, str]] = {
    RetrievalError: (500, "Unable to retrieve supporting chunks."),
    RerankingError: (500, "Unable to rerank retrieved chunks."),
    PromptBuildError: (500, "Unable to build the prompt."),
    GenerationError: (502, "The language model failed to generate a response."),
}


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(RAGException)
    async def rag_exception_handler(request: Request, exc: RAGException):
        status_code, detail = ERRORS.get(type(exc), (500, str(exc)))
        return JSONResponse(status_code=status_code, content={"detail": detail})
