from typing import Any
from dataclasses import dataclass


@dataclass(slots=True)
class RetrievalResult:
    document_id: str
    page_content: str
    score: float
    metadata: dict[str, Any]