from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_pipeline
from src.pipeline.metrics import PipelineMetrics
from src.pipeline.rag_pipeline import PipelineResponse
from src.retrieval.models import RetrievalResult


class FakePipeline:
    def answer(self, question):
        chunk = RetrievalResult(
            document_id="APPLE_2022_10K",
            content="Total net sales were $394,328 million.",
            score=0.9,
            metadata={"document_id": "APPLE_2022_10K", "page": 9},
        )
        return PipelineResponse("$394,328 million", [chunk], PipelineMetrics())


def test_chat_returns_filing_and_page_of_sources():
    app.dependency_overrides[get_pipeline] = FakePipeline
    try:
        # No `with`: the lifespan, which loads the real models, does not run.
        response = TestClient(app).post("/chat", json={"question": "Sales?"})
    finally:
        app.dependency_overrides.clear()

    source = response.json()["sources"][0]
    assert response.status_code == 200
    assert source["document_id"] == "APPLE_2022_10K"
    assert source["page"] == 10
