"""
Langfuse tracing.

Tracing is on when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set
(LANGFUSE_BASE_URL selects the server; Langfuse Cloud by default). Without
them every helper here is a no-op, so the pipeline runs unchanged in tests
and in local runs without a Langfuse project.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager
from functools import cache
from typing import Any

logger = logging.getLogger(__name__)


class _NoopObservation:
    def update(self, **kwargs: Any) -> _NoopObservation:
        return self


def tracing_enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


@cache
def _client():
    from langfuse import get_client

    logger.info("Langfuse tracing enabled")
    return get_client()


@contextmanager
def observation(name: str, as_type: str = "span", **kwargs: Any) -> Iterator[Any]:
    """
    Record the enclosed block as a Langfuse observation.

    Nested observations become its children; the outermost one starts a
    trace. Yields an object whose update(output=..., metadata=...) adds to
    the observation.
    """

    if not tracing_enabled():
        yield _NoopObservation()
        return

    with _client().start_as_current_observation(
        name=name, as_type=as_type, **kwargs
    ) as obs:
        yield obs


def update_current(**kwargs: Any) -> None:
    """Add output or metadata to the innermost open span."""

    if tracing_enabled():
        _client().update_current_span(**kwargs)


def current_trace_id() -> str | None:
    return _client().get_current_trace_id() if tracing_enabled() else None


def flush() -> None:
    """Send buffered observations; call before the process exits."""

    if tracing_enabled():
        _client().flush()
