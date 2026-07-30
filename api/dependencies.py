from src.pipeline.service import RAGService

rag_service: RAGService | None = None


def get_rag_service() -> RAGService:
    if rag_service is None:
        raise RuntimeError("Pipeline not initialized")
    return rag_service
