from src.pipeline.rag_pipeline import RAGPipeline

# Set by api.lifespan when the app starts.
pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    if pipeline is None:
        raise RuntimeError("Pipeline not initialized")
    return pipeline
