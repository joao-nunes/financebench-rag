from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RetrievalResult:
    document_id: str
    content: str
    score: float
    metadata: dict[str, Any]
