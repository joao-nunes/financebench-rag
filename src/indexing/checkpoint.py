from __future__ import annotations

import json
import shutil
from pathlib import Path

from src.indexing.faiss_store import FAISSStore
from src.utils.logger import get_logger
from src.indexing.metadata import CheckpointMetadata


logger = get_logger(__name__)


class CheckpointManager:

    def __init__(
        self,
        checkpoint_dir: str | Path,
    ) -> None:

        self.checkpoint_dir = Path(checkpoint_dir)

        self.faiss_dir = self.checkpoint_dir / "faiss"

        self.metadata_file = (
            self.checkpoint_dir / "metadata.json"
        )

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
    
    def save(
        self,
        store: FAISSStore,
        metadata: CheckpointMetadata,
    ) -> None:

        logger.info(
            "Saving checkpoint..."
        )

        self.faiss_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        store.save(
            self.faiss_dir
        )

        with open(
            self.metadata_file,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata.to_dict(),
                f,
                indent=4,
            )

        logger.info(
            "Checkpoint saved."
        )

    def load_metadata(
        self,
    ) -> CheckpointMetadata | None:

        if not self.metadata_file.exists():
            return None

        with open(
            self.metadata_file,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        return CheckpointMetadata(**data)
    
    def load_store(
        self,
        store: FAISSStore,
        embedding_model,
    ) -> FAISSStore:

        store.load(
            self.faiss_dir,
            embedding_model,
        )

        return store
    
    def exists(self) -> bool:

        return (
            self.metadata_file.exists()
            and
            (self.faiss_dir / "index.faiss").exists()
            and
            (self.faiss_dir / "index.pkl").exists()
        )
    
    def clear(self):

        if self.checkpoint_dir.exists():

            shutil.rmtree(
                self.checkpoint_dir
            )

            logger.info(
                "Checkpoint removed."
            )