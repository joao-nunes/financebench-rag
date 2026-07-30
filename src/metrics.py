from dataclasses import dataclass


@dataclass(slots=True)
class PipelineMetrics:
    retrieval_time: float = 0.0
    reranking_time: float = 0.0
    prompt_build_time: float = 0.0
    generation_time: float = 0.0
    pipeline_time: float = 0.0

    retrieved_documents: int = 0
    reranked_documents: int = 0

    prompt_length: int = 0
