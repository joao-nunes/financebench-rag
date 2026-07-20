import pickle
from pathlib import Path
from typing import Any


def save_cache(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_cache(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def cache_exists(path: Path) -> bool:
    return path.exists()