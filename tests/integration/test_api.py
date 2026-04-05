"""API contract tests via TestClient against a real pgvector testcontainer.

Every test hits the actual FastAPI app with a real database. LLM and TEI
are mocked — these tests verify HTTP contracts, not algorithm correctness.

Run with: uv run pytest tests/integration/test_api.py -m integration -v
"""

import uuid
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from openfable.db import get_session
from openfable.main import create_app
from openfable.services.ingestion.pipeline import IngestionPipeline, get_ingestion_pipeline
from openfable.services.llm_service import get_llm_service
from openfable.services.retrieval_service import RetrievalService, get_retrieval_service
from openfable.repositories.document_repo import DocumentRepository
from openfable.repositories.node_repo import NodeRepository


@pytest.fixture
def api_client(clean_db, patch_engines, session_engine, mock_llm_service, mock_embedding_service):
    """TestClient wired to real pgvector DB with mocked LLM/TEI."""
    from sqlalchemy.orm import sessionmaker

    app = create_app()
    TestSessionLocal = sessionmaker(session_engine, expire_on_commit=False)

    def override_get_session():
        with TestSessionLocal() as session:
            yield session

    mock_pipeline = MagicMock(spec=IngestionPipeline)
    mock_pipeline.start = MagicMock(return_value=MagicMock())
    mock_pipeline.cancel = MagicMock(return_value=False)
    mock_pipeline.is_running = MagicMock(return_value=False)

    def override_retrieval() -> RetrievalService:
        return RetrievalService(
            llm_service=mock_llm_service,
            embedding_service=mock_embedding_service,
            node_repo=NodeRepository(),
            doc_repo=DocumentRepository(),
        )

    app.dependency_overrides[get_session] = override_get_session
    app.dependency_overrides[get_ingestion_pipeline] = lambda: mock_pipeline
    app.dependency_overrides[get_retrieval_service] = override_retrieval
    app.dependency_overrides[get_llm_service] = lambda: mock_llm_service

    with TestClient(app=app, raise_server_exceptions=True) as client:
        yield client

    app.dependency_overrides.clear()


DOC_TEXT = (
    "Integration test document about software testing. This text covers "
    "how automated tests verify system behavior and catch regressions "
    "before they reach production environments."
)


@pytest.mark.integration
class TestDocumentEndpoints:

    def test_create_document_returns_202(self, api_client):
        resp = api_client.post("/v1/api/documents", json={"text": DOC_TEXT, "metadata": {}})
        assert resp.status_code == 202
        data = resp.json()
        assert "document_id" in data
        assert "job_id" in data
        assert data["status"] == "pending"

    def test_get_document(self, api_client):
        create_resp = api_client.post("/v1/api/documents", json={"text": DOC_TEXT, "metadata": {}})
        doc_id = create_resp.json()["document_id"]

        resp = api_client.get(f"/v1/api/documents/{doc_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"] == doc_id
        assert "index_status" in data

    def test_list_documents(self, api_client):
        api_client.post("/v1/api/documents", json={"text": DOC_TEXT, "metadata": {}})

        resp = api_client.get("/v1/api/documents")
        assert resp.status_code == 200
        data = resp.json()
        assert "documents" in data
        assert "total" in data
        assert data["total"] >= 1

    def test_delete_document(self, api_client):
        create_resp = api_client.post("/v1/api/documents", json={"text": DOC_TEXT, "metadata": {}})
        doc_id = create_resp.json()["document_id"]

        resp = api_client.delete(f"/v1/api/documents/{doc_id}")
        assert resp.status_code == 204

        get_resp = api_client.get(f"/v1/api/documents/{doc_id}")
        assert get_resp.status_code == 404

    def test_get_nonexistent_returns_404(self, api_client):
        fake_id = str(uuid.uuid4())
        resp = api_client.get(f"/v1/api/documents/{fake_id}")
        assert resp.status_code == 404

    def test_delete_nonexistent_returns_404(self, api_client):
        fake_id = str(uuid.uuid4())
        resp = api_client.delete(f"/v1/api/documents/{fake_id}")
        assert resp.status_code == 404

    def test_create_empty_text_rejected(self, api_client):
        resp = api_client.post("/v1/api/documents", json={"text": "", "metadata": {}})
        assert resp.status_code == 422


@pytest.mark.integration
class TestQueryEndpoint:

    def test_query_rejects_missing_budget(self, api_client):
        resp = api_client.post("/v1/api/query", json={"query": "test"})
        assert resp.status_code == 422

    def test_query_rejects_budget_below_minimum(self, api_client):
        resp = api_client.post("/v1/api/query", json={"query": "test", "token_budget": 50})
        assert resp.status_code == 422

    def test_query_rejects_budget_above_maximum(self, api_client):
        resp = api_client.post("/v1/api/query", json={"query": "test", "token_budget": 50000})
        assert resp.status_code == 422


@pytest.mark.integration
class TestHealthEndpoint:

    def test_health_returns_status(self, api_client):
        resp = api_client.get("/v1/api/health")
        assert resp.status_code in (200, 503)
        data = resp.json()
        assert "status" in data
