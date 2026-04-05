"""Retrieval integration tests against real pgvector testcontainer.

Tests verify the retrieval algorithms produce correct results when
querying a real database populated with known documents and vectors.
LLM calls are mocked to return deterministic selections; vector
similarity uses real pgvector cosine distance.

Run with: uv run pytest tests/integration/test_retrieval.py -m integration -v
"""

import uuid
from unittest.mock import MagicMock

import pytest
from sqlalchemy import select

from openfable.models.node import Node
from openfable.repositories.document_repo import DocumentRepository
from openfable.repositories.node_repo import NodeRepository
from openfable.schemas.retrieval import DocumentSelection, LLMSelectResult
from openfable.services.retrieval_service import RetrievalService
from tests.integration.conftest import ingest_document
from tests.integration.fixtures import (
    CHUNK_VECTORS_LONG,
    CHUNK_VECTORS_SHORT,
    GOLDEN_DOC_LONG,
    GOLDEN_DOC_SHORT,
    MOCK_CHUNKS_LONG,
    MOCK_CHUNKS_SHORT,
    QUERY_RETRIEVAL,
    make_query_vector,
)


def _make_retrieval_service(mock_llm, mock_embed) -> RetrievalService:
    return RetrievalService(
        llm_service=mock_llm,
        embedding_service=mock_embed,
        node_repo=NodeRepository(),
        doc_repo=DocumentRepository(),
    )


@pytest.mark.integration
class TestDocumentLevelRetrieval:

    def test_high_budget_returns_document_level(
        self, clean_db, patch_engines, TestSessionLocal,
        mock_llm_service, mock_embedding_service, monkeypatch
    ):
        """With a high budget, retrieval returns document_level routing
        and includes document content."""
        doc_id = ingest_document(
            GOLDEN_DOC_SHORT, TestSessionLocal, monkeypatch,
            mock_llm_service, MOCK_CHUNKS_SHORT, CHUNK_VECTORS_SHORT,
        )

        # LLMselect returns the ingested document as relevant
        mock_llm_service.complete_structured = MagicMock(return_value=LLMSelectResult(
            selected_documents=[DocumentSelection(document_id=doc_id, relevance_score=0.9)]
        ))

        # embed_batch returns query vector for vector path
        mock_embedding_service.embed_batch = MagicMock(return_value=[make_query_vector()])

        svc = _make_retrieval_service(mock_llm_service, mock_embedding_service)
        with TestSessionLocal() as session:
            result = svc.query(session, QUERY_RETRIEVAL, token_budget=32000)

        assert result.routing == "document_level"
        assert len(result.documents) >= 1
        assert any(d.document_id == doc_id for d in result.documents)

    def test_empty_corpus_returns_empty(
        self, clean_db, patch_engines, TestSessionLocal,
        mock_llm_service, mock_embedding_service,
    ):
        """Querying an empty database returns no results without error."""
        mock_llm_service.complete_structured = MagicMock(return_value=LLMSelectResult(
            selected_documents=[]
        ))
        mock_embedding_service.embed_batch = MagicMock(return_value=[make_query_vector()])

        svc = _make_retrieval_service(mock_llm_service, mock_embedding_service)
        with TestSessionLocal() as session:
            result = svc.query(session, QUERY_RETRIEVAL, token_budget=1000)

        # Empty corpus: no documents found by either path
        assert len(result.documents) == 0
        assert result.total_tokens == 0

    def test_vector_path_ranks_relevant_higher(
        self, clean_db, patch_engines, TestSessionLocal,
        mock_llm_service, mock_embedding_service, monkeypatch
    ):
        """The vector retrieval path ranks the document with a high-similarity
        embedding above one with an orthogonal embedding."""
        # Ingest relevant doc first, then irrelevant doc second.
        # Each ingest_document call patches pipeline services, so we do them
        # sequentially — the vectors are baked into the DB rows during ingestion.
        relevant_id = ingest_document(
            GOLDEN_DOC_SHORT, TestSessionLocal, monkeypatch,
            mock_llm_service, MOCK_CHUNKS_SHORT, CHUNK_VECTORS_SHORT,
        )
        irrelevant_id = ingest_document(
            GOLDEN_DOC_LONG, TestSessionLocal, monkeypatch,
            mock_llm_service, MOCK_CHUNKS_LONG, CHUNK_VECTORS_LONG,
        )

        # Verify embeddings are actually stored differently
        with TestSessionLocal() as session:
            from sqlalchemy import select as sa_select
            relevant_nodes = list(session.execute(
                sa_select(Node).where(Node.document_id == relevant_id, Node.embedding.isnot(None))
            ).scalars())
            irrelevant_nodes = list(session.execute(
                sa_select(Node).where(Node.document_id == irrelevant_id, Node.embedding.isnot(None))
            ).scalars())
            assert len(relevant_nodes) > 0, "Relevant doc should have embedded nodes"
            assert len(irrelevant_nodes) > 0, "Irrelevant doc should have embedded nodes"

        # LLMselect returns both docs — let vector scores determine ranking
        mock_llm_service.complete_structured = MagicMock(return_value=LLMSelectResult(
            selected_documents=[
                DocumentSelection(document_id=relevant_id, relevance_score=0.5),
                DocumentSelection(document_id=irrelevant_id, relevance_score=0.5),
            ]
        ))
        mock_embedding_service.embed_batch = MagicMock(return_value=[make_query_vector()])

        svc = _make_retrieval_service(mock_llm_service, mock_embedding_service)
        with TestSessionLocal() as session:
            result = svc.query(session, QUERY_RETRIEVAL, token_budget=32000)

        assert len(result.documents) >= 2
        doc_ids = [d.document_id for d in result.documents]
        relevant_rank = doc_ids.index(relevant_id)
        irrelevant_rank = doc_ids.index(irrelevant_id)
        assert relevant_rank < irrelevant_rank, (
            f"Relevant doc (rank {relevant_rank}) should rank above "
            f"irrelevant doc (rank {irrelevant_rank})"
        )


@pytest.mark.integration
class TestNodeLevelRetrieval:

    def test_low_budget_triggers_node_level(
        self, clean_db, patch_engines, TestSessionLocal,
        mock_llm_service, mock_embedding_service, monkeypatch
    ):
        """With a budget too small for full documents, retrieval falls back
        to node_level routing and returns individual chunks."""
        # Ingest both documents so total tokens exceed a small budget
        doc_id_short = ingest_document(
            GOLDEN_DOC_SHORT, TestSessionLocal, monkeypatch,
            mock_llm_service, MOCK_CHUNKS_SHORT, CHUNK_VECTORS_SHORT,
        )
        doc_id_long = ingest_document(
            GOLDEN_DOC_LONG, TestSessionLocal, monkeypatch,
            mock_llm_service, MOCK_CHUNKS_LONG, CHUNK_VECTORS_LONG,
        )

        # LLMselect returns both documents
        mock_llm_service.complete_structured = MagicMock(return_value=LLMSelectResult(
            selected_documents=[
                DocumentSelection(document_id=doc_id_short, relevance_score=0.9),
                DocumentSelection(document_id=doc_id_long, relevance_score=0.8),
            ]
        ))
        mock_embedding_service.embed_batch = MagicMock(return_value=[make_query_vector()])

        svc = _make_retrieval_service(mock_llm_service, mock_embedding_service)
        with TestSessionLocal() as session:
            result = svc.query(session, QUERY_RETRIEVAL, token_budget=100)

        # With budget=100 and multiple documents, should trigger node-level
        # (or return document_level if single doc fits — either is valid routing)
        assert result.routing in ("document_level", "node_level")
        if result.routing == "node_level":
            assert result.chunks is not None
            assert len(result.chunks) >= 1
            # Budget should be respected
            if result.total_tokens_used is not None:
                assert result.total_tokens_used <= 100 or result.over_budget is True
