from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime


@dataclass
class CheckpointMetadata:
    """
    Metadata required to resume indexing.
    """

    current_batch: int
    indexed_documents: int
    total_documents: int

    batch_size: int

    embedding_model: str

    created_at: str

    @classmethod
    def create(
        cls,
        current_batch: int,
        indexed_documents: int,
        total_documents: int,
        batch_size: int,
        embedding_model: str,
    ) -> "CheckpointMetadata":

        return cls(
            current_batch=current_batch,
            indexed_documents=indexed_documents,
            total_documents=total_documents,
            batch_size=batch_size,
            embedding_model=embedding_model,
            created_at=datetime.now().isoformat(),
        )

    def to_dict(self):

        return asdict(self)