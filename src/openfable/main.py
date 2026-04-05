from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastmcp import FastMCP
from sqlalchemy import text

from openfable.db import engine
from openfable.routers import documents, health, retrieval


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup: verify DB connection (migrations already ran via entrypoint)
    with engine.begin() as conn:
        conn.execute(text("SELECT 1"))
    yield
    # Shutdown: dispose connection pool
    engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="OpenFable",
        description="FABLE retrieval engine - structured, hierarchy-aware RAG",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(health.router, prefix="/v1/api", tags=["health"])
    app.include_router(documents.router, prefix="/v1/api", tags=["documents"])
    app.include_router(retrieval.router, prefix="/v1/api", tags=["retrieval"])

    # Expose all API routes as MCP tools
    mcp = FastMCP.from_fastapi(app=app)
    app.mount("/v1/mcp", mcp.sse_app())

    return app


app = create_app()
