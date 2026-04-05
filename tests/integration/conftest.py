"""Integration test infrastructure for OpenFable.

Spins up a real pgvector testcontainer and patches the module-level engine
so that both direct service calls AND the FastAPI app (via TestClient) use
the testcontainer database instead of the Docker Compose 'db' host.

Run integration tests with: uv run pytest -m integration
Skip integration tests with: uv run pytest -m "not integration"
"""

import os
import subprocess
import uuid
from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer

import openfable.db as db_mod
import openfable.main as main_mod
import openfable.services.ingestion.pipeline as pipeline_mod
from openfable.models.chunk import Chunk
from openfable.models.document import Document, IngestionJob
from openfable.models.node import Node
from openfable.repositories.document_repo import (
    DocumentRepository,
    compute_content_hash,
    count_tokens,
)
from openfable.repositories.node_repo import NodeInsert, NodeRepository
from openfable.schemas.chunking import ChunkResult
from openfable.services.embedding_service import EmbeddingService
from openfable.services.llm_service import LLMService
from openfable.services.retrieval_service import RetrievalService
from tests.integration.fixtures import (
    CHUNK_VECTORS_SHORT,
    GOLDEN_DOC_SHORT,
    MOCK_CHUNKS_SHORT,
    ROOT_VECTOR,
)

PROJECT_ROOT = Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Database: testcontainer + engine + sessions
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def postgres_url() -> str:  # type: ignore[return]
    """Start pgvector testcontainer, run Alembic migrations, yield connection URL."""
    with PostgresContainer("pgvector/pgvector:pg17") as container:
        db_url = container.get_connection_url()
        subprocess.run(
            ["uv", "run", "alembic", "upgrade", "head"],
            env={**os.environ, "OPENFABLE_DATABASE_URL": db_url},
            check=True,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
        )
        yield db_url


@pytest.fixture(scope="session")
def session_engine(postgres_url: str):  # type: ignore[return]
    engine = create_engine(postgres_url, echo=False)
    yield engine
    engine.dispose()


@pytest.fixture
def TestSessionLocal(session_engine):
    return sessionmaker(session_engine, expire_on_commit=False)


@pytest.fixture
def clean_db(session_engine) -> Generator[None, None, None]:  # type: ignore[return]
    truncate = text(
        "TRUNCATE TABLE chunks, nodes, ingestion_jobs, documents RESTART IDENTITY CASCADE"
    )
    with session_engine.begin() as conn:
        conn.execute(truncate)
    yield
    with session_engine.begin() as conn:
        conn.execute(truncate)


# ---------------------------------------------------------------------------
# Engine patching: redirect openfable.db.engine and openfable.main.engine
# so that lifespan startup and SessionLocal both hit the testcontainer.
# ---------------------------------------------------------------------------


@pytest.fixture
def patch_engines(session_engine, monkeypatch):
    """Patch module-level engine references so the FastAPI app uses the testcontainer DB."""
    test_session_local = sessionmaker(session_engine, expire_on_commit=False)
    monkeypatch.setattr(db_mod, "engine", session_engine)
    monkeypatch.setattr(db_mod, "SessionLocal", test_session_local)
    monkeypatch.setattr(main_mod, "engine", session_engine)
    monkeypatch.setattr(pipeline_mod, "SessionLocal", test_session_local)


# ---------------------------------------------------------------------------
# Mock services
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_service() -> LLMService:
    llm = MagicMock(spec=LLMService)
    llm.complete_structured = MagicMock()
    llm.model = "test-model"
    llm.health_probe = MagicMock(return_value=None)
    return llm  # type: ignore[return-value]


@pytest.fixture
def mock_embedding_service() -> EmbeddingService:
    embed = MagicMock(spec=EmbeddingService)
    embed.embed_batch = MagicMock()
    embed.embed_nodes = MagicMock()
    return embed  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Pipeline helpers: build mocks from fixture data, ingest a document
# ---------------------------------------------------------------------------


def make_mock_chunking_service(chunks_dicts: list[dict]) -> MagicMock:
    svc = MagicMock()
    svc.segment = MagicMock(return_value=[
        ChunkResult(chunk_text=c["chunk_text"], start_idx=c["start_idx"], end_idx=c["end_idx"])
        for c in chunks_dicts
    ])
    return svc


def make_mock_tree_builder(chunk_count: int, chunk_vectors: list[list[float]]) -> MagicMock:
    root_id = uuid.uuid4()
    leaf_ids = [uuid.uuid4() for _ in range(chunk_count)]

    root_ni = NodeInsert(
        id=root_id, node_type="root", depth=1, position=0,
        title="Test Root", summary="Test document summary",
        toc_path="Test_Root", content=None, token_count=None,
        parent_id=None, path="Test_Root",
    )
    leaf_nis = [
        NodeInsert(
            id=leaf_ids[i], node_type="leaf", depth=2, position=i,
            title=None, summary=None, toc_path=None,
            content=f"Leaf content {i}", token_count=50,
            parent_id=root_id, path=f"Test_Root.chunk_{i}", chunk_id=None,
        )
        for i in range(chunk_count)
    ]

    builder = MagicMock()
    builder.build = MagicMock(return_value=[root_ni] + leaf_nis)
    builder._leaf_ids = leaf_ids
    builder._root_id = root_id
    return builder


def make_mock_embedding_service(mock_tree_builder: MagicMock, chunk_vectors: list[list[float]]) -> MagicMock:
    root_id = mock_tree_builder._root_id
    leaf_ids = mock_tree_builder._leaf_ids

    def _embed_nodes(node_texts, batch_size=64):
        result = []
        for node_id, _text in node_texts:
            if node_id == root_id:
                result.append((node_id, ROOT_VECTOR))
            elif node_id in leaf_ids:
                idx = leaf_ids.index(node_id)
                result.append((node_id, chunk_vectors[idx] if idx < len(chunk_vectors) else ROOT_VECTOR))
            else:
                result.append((node_id, ROOT_VECTOR))
        return result

    embed_svc = MagicMock()
    embed_svc.embed_nodes = _embed_nodes
    return embed_svc


def ingest_document(
    text: str,
    TestSessionLocal,
    monkeypatch,
    mock_llm_service,
    chunks_dicts: list[dict],
    chunk_vectors: list[list[float]],
) -> uuid.UUID:
    """Ingest a document through the full pipeline with mocked LLM/embeddings. Returns document_id."""
    mock_chunking = make_mock_chunking_service(chunks_dicts)
    mock_tree = make_mock_tree_builder(len(chunks_dicts), chunk_vectors)
    mock_embed = make_mock_embedding_service(mock_tree, chunk_vectors)

    monkeypatch.setattr("openfable.services.ingestion.pipeline.LLMService", lambda *a, **kw: mock_llm_service)
    monkeypatch.setattr("openfable.services.ingestion.pipeline.ChunkingService", lambda llm: mock_chunking)
    monkeypatch.setattr("openfable.services.ingestion.pipeline.TreeBuilder", lambda llm: mock_tree)
    monkeypatch.setattr("openfable.services.ingestion.pipeline.EmbeddingService", lambda *a, **kw: mock_embed)

    repo = DocumentRepository()
    with TestSessionLocal() as session:
        with session.begin():
            content_hash = compute_content_hash(text)
            token_count = count_tokens(text)
            doc, job = repo.create_with_job(session, text, content_hash, token_count)
            doc_id = doc.id
            job_id = job.id

    from openfable.services.ingestion.pipeline import IngestionPipeline
    pipeline = IngestionPipeline()
    t = pipeline.start(doc_id, job_id)
    t.join(timeout=30)
    return doc_id
