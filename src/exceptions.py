class RAGException(Exception):
    """Base exception for the RAG application."""


class RetrievalError(RAGException):
    """Raised when document retrieval fails."""


class RerankingError(RAGException):
    """Raised when reranking fails."""


class PromptBuildError(RAGException):
    """Raised when prompt construction fails."""


class GenerationError(RAGException):
    """Raised when the LLM generation fails."""
