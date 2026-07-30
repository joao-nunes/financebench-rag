from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.exceptions import (
    GenerationError,
    PromptBuildError,
    RAGException,
    RerankingError,
    RetrievalError,
)


def register_exception_handlers(app: FastAPI) -> None:

    @app.exception_handler(RetrievalError)
    async def retrieval_error_handler(
        request: Request,
        exc: RetrievalError,
    ):
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Unable to retrieve supporting chunks.",
            },
        )

    @app.exception_handler(RerankingError)
    async def reranking_error_handler(
        request: Request,
        exc: RerankingError,
    ):
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Unable to rerank retrieved chunks.",
            },
        )

    @app.exception_handler(PromptBuildError)
    async def prompt_error_handler(
        request: Request,
        exc: PromptBuildError,
    ):
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Unable to build the prompt.",
            },
        )

    @app.exception_handler(GenerationError)
    async def generation_error_handler(
        request: Request,
        exc: GenerationError,
    ):
        return JSONResponse(
            status_code=502,
            content={
                "detail": "The language model failed to generate a response.",
            },
        )

    @app.exception_handler(RAGException)
    async def rag_exception_handler(
        request: Request,
        exc: RAGException,
    ):
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc),
            },
        )
