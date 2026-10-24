from contextlib import asynccontextmanager

import api.dependencies as dependencies
from src.logging_config import configure_logging
from src.observability import flush as flush_traces
from src.pipeline.factory import build_pipeline


@asynccontextmanager
async def lifespan(app):
    configure_logging()
    dependencies.pipeline = build_pipeline()

    yield

    dependencies.pipeline = None
    flush_traces()
