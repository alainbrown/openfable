"""Ingestion pipeline integration tests against real pgvector testcontainer.

Tests the full pipeline flow: document creation → chunking → tree building →
embedding → status update. LLM and TEI are mocked with deterministic fixtures;
the database is real.

Run with: uv run pytest tests/integration/test_pipeline.py -m integration -v
"""

import pytest
from sqlalchemy import select

from openfable.models.chunk import Chunk
from openfable.models.document import Document
from openfable.models.node import Node
from openfable.repositories.document_repo import (
    DocumentRepository,
    compute_content_hash,
    count_tokens,
)
from openfable.services.ingestion.pipeline import IngestionPipeline
from tests.integration.conftest import (
    ingest_document,
)
from tests.integration.fixtures import (
    CHUNK_VECTORS_LONG,
    CHUNK_VECTORS_SHORT,
    GOLDEN_DOC_LONG,
    GOLDEN_DOC_SHORT,
    MOCK_CHUNKS_LONG,
    MOCK_CHUNKS_SHORT,
)


@pytest.mark.integration
class TestIngestionPipeline:
    def test_full_pipeline_completes(
        self, clean_db, patch_engines, TestSessionLocal, mock_llm_service, monkeypatch
    ):
        """Ingest a document end-to-end. Verify: status=complete, chunks exist,
        tree nodes exist with correct parent/child relationships, embeddings stored."""
        doc_id = ingest_document(
            GOLDEN_DOC_SHORT,
            TestSessionLocal,
            monkeypatch,
            mock_llm_service,
            MOCK_CHUNKS_SHORT,
            CHUNK_VECTORS_SHORT,
        )

        with TestSessionLocal() as session:
            chunks = list(
                session.execute(select(Chunk).where(Chunk.document_id == doc_id)).scalars()
            )
            assert len(chunks) == len(MOCK_CHUNKS_SHORT)

            nodes = list(
                session.execute(
                    select(Node)
                    .where(Node.document_id == doc_id)
                    .order_by(Node.depth, Node.position)
                ).scalars()
            )
            assert len(nodes) == 1 + len(MOCK_CHUNKS_SHORT)  # root + leaves

            root = [n for n in nodes if n.node_type == "root"]
            assert len(root) == 1
            assert root[0].depth == 1

            leaves = [n for n in nodes if n.node_type == "leaf"]
            assert len(leaves) == len(MOCK_CHUNKS_SHORT)
            for leaf in leaves:
                assert leaf.parent_id == root[0].id
                assert leaf.depth == 2

            nodes_with_embedding = [n for n in nodes if n.embedding is not None]
            assert len(nodes_with_embedding) > 0

    def test_longer_document_creates_more_nodes(
        self, clean_db, patch_engines, TestSessionLocal, mock_llm_service, monkeypatch
    ):
        """A 3-chunk document produces 4 nodes (1 root + 3 leaves)."""
        doc_id = ingest_document(
            GOLDEN_DOC_LONG,
            TestSessionLocal,
            monkeypatch,
            mock_llm_service,
            MOCK_CHUNKS_LONG,
            CHUNK_VECTORS_LONG,
        )

        with TestSessionLocal() as session:
            nodes = list(session.execute(select(Node).where(Node.document_id == doc_id)).scalars())
            assert len(nodes) == 1 + len(MOCK_CHUNKS_LONG)

    def test_chunking_failure_raises(self, clean_db, patch_engines, TestSessionLocal, monkeypatch):
        """When chunking raises an error, the exception propagates."""
        from unittest.mock import MagicMock

        from openfable.exceptions import ChunkingError

        failing_chunking = MagicMock()
        failing_chunking.segment = MagicMock(side_effect=ChunkingError("bad chunks"))

        monkeypatch.setattr(
            "openfable.services.ingestion.pipeline.LLMService", lambda *a, **kw: MagicMock()
        )
        monkeypatch.setattr(
            "openfable.services.ingestion.pipeline.ChunkingService", lambda llm: failing_chunking
        )
        monkeypatch.setattr(
            "openfable.services.ingestion.pipeline.TreeBuilder", lambda llm: MagicMock()
        )
        monkeypatch.setattr(
            "openfable.services.ingestion.pipeline.EmbeddingService", lambda *a, **kw: MagicMock()
        )

        repo = DocumentRepository()
        with TestSessionLocal() as session:
            with session.begin():
                content_hash = compute_content_hash(GOLDEN_DOC_SHORT)
                token_count = count_tokens(GOLDEN_DOC_SHORT)
                doc = repo.create(session, GOLDEN_DOC_SHORT, content_hash, token_count)
                doc_id = doc.id

        pipeline = IngestionPipeline()
        with pytest.raises(ChunkingError):
            with TestSessionLocal() as session:
                pipeline.run(session, doc_id)

    def test_delete_cascades_to_nodes_and_chunks(
        self, clean_db, patch_engines, TestSessionLocal, mock_llm_service, monkeypatch
    ):
        """Deleting a document removes all its chunks and nodes from the DB."""
        doc_id = ingest_document(
            GOLDEN_DOC_SHORT,
            TestSessionLocal,
            monkeypatch,
            mock_llm_service,
            MOCK_CHUNKS_SHORT,
            CHUNK_VECTORS_SHORT,
        )

        # Verify data exists before delete
        with TestSessionLocal() as session:
            assert (
                session.execute(select(Node).where(Node.document_id == doc_id)).scalars().first()
                is not None
            )
            assert (
                session.execute(select(Chunk).where(Chunk.document_id == doc_id)).scalars().first()
                is not None
            )

        # Delete via repository
        repo = DocumentRepository()
        with TestSessionLocal() as session:
            with session.begin():
                repo.delete(session, doc_id)

        # Verify everything is gone
        with TestSessionLocal() as session:
            assert (
                session.execute(select(Document).where(Document.id == doc_id)).scalar_one_or_none()
                is None
            )
            assert (
                list(session.execute(select(Node).where(Node.document_id == doc_id)).scalars())
                == []
            )
            assert (
                list(session.execute(select(Chunk).where(Chunk.document_id == doc_id)).scalars())
                == []
            )
