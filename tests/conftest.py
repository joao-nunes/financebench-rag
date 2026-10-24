import pytest


@pytest.fixture(autouse=True)
def no_langfuse(monkeypatch):
    """Keep tests from sending traces when .env holds Langfuse keys."""
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
