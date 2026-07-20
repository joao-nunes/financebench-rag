from __future__ import annotations

import logging
from pathlib import Path


LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    log_dir: str | Path = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create (or retrieve) a configured logger.

    Parameters
    ----------
    name
        Logger name (typically __name__).
    log_dir
        Directory where log files are stored.
    level
        Logging level.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger
    # has already been configured.
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    #
    # Console
    #
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    #
    # File
    #
    file_handler = logging.FileHandler(
        log_dir / "indexer.log",
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger