from pathlib import Path

from src.config import (
    CONTEXT_TOKEN_BUDGET,
    DOCUMENT_INFO_FILE,
    RERANKER_MODEL,
    VECTORSTORE_DIR,
)
from src.generation.openai_generator import OpenAIGenerator
from src.generation.simple_prompt_builder import SimplePromptBuilder
from src.indexing.embeddings import get_embedding_model
from src.indexing.faiss_store import FAISSStore
from src.pipeline.rag_pipeline import RAGPipeline
from src.retrieval.context_selector import TokenBudgetSelector
from src.retrieval.cross_encoder_reranker import CrossEncoderReranker
from src.retrieval.faiss_retriever import FAISSRetriever
from src.retrieval.filing_filter import FilingFilter


def build_pipeline(
    vectorstore_dir: Path = VECTORSTORE_DIR,
    document_info_file: Path = DOCUMENT_INFO_FILE,
) -> RAGPipeline:
    """
    The production pipeline: filing-filtered dense retrieval, cross-encoder
    reranking, a token-budgeted context and an OpenAI model.
    """

    vectorstore = FAISSStore()
    vectorstore.load(vectorstore_dir, get_embedding_model())

    filing_filter = FilingFilter.from_document_information(
        vectorstore.document_ids(),
        document_info_file,
    )

    return RAGPipeline(
        retriever=FAISSRetriever(vectorstore.store, filing_filter=filing_filter),
        # Keep the full ranking; the token budget decides what reaches the LLM.
        reranker=CrossEncoderReranker(model_name=RERANKER_MODEL, top_n=None),
        prompt_builder=SimplePromptBuilder(),
        generator=OpenAIGenerator(),
        context_selector=TokenBudgetSelector(CONTEXT_TOKEN_BUDGET),
    )
