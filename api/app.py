from fastapi import FastAPI

from api.exception_handlers import register_exception_handlers
from api.lifespan import lifespan
from api.routes import router

app = FastAPI(
    title="Simple RAG API",
    lifespan=lifespan,
)
register_exception_handlers(app)
app.include_router(router)
