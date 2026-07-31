"""IngestionPipeline: chunk -> tree -> embed.

The two LLM-dependent stages (chunking, tree structuring) are injected rather
than constructed inline, so the pipeline can be driven either by a live LLM
(LiteLLM, the default) or by pre-computed structure supplied from outside the
process -- see openfable.services.ingestion.precomputed.

When nothing is injected the defaults are built from the module-level
LLMService/ChunkingService/TreeBuilder names, preserving the original behaviour.
"""

import logging
import uuid
from typing import Protocol

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from openfable.config import settings
from openfable.exceptions import ChunkingError
from openfable.models.chunk import Chunk as ChunkModel
from openfable.models.node import Node
from openfable.repositories.chunk_repo import ChunkRepository
from openfable.repositories.document_repo import DocumentRepository
from openfable.repositories.node_repo import NodeInsert, NodeRepository
from openfable.schemas.chunking import ChunkResult
from openfable.services.embedding_service import EmbeddingService, _build_embedding_text
from openfable.services.ingestion.chunking import ChunkingService
from openfable.services.ingestion.tree_builder import TreeBuilder
from openfable.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class Chunker(Protocol):
    """Splits raw document text into semantically coherent chunks."""

    def segment(self, text: str) -> list[ChunkResult]: ...


class TreeStructurer(Protocol):
    """Organises persisted chunks into a hierarchical node tree."""

    def build(self, chunks: list[ChunkModel]) -> list[NodeInsert]: ...


class IngestionPipeline:
    def __init__(
        self,
        chunker: Chunker | None = None,
        tree_structurer: TreeStructurer | None = None,
    ) -> None:
        self._chunker = chunker
        self._tree_structurer = tree_structurer

    def _resolve(self) -> tuple[Chunker, TreeStructurer]:
        """Return the injected implementations, falling back to the LiteLLM ones.

        LLMService is constructed only when a default is actually needed, so a
        fully-injected pipeline never touches LiteLLM and needs no API key.
        """
        chunker = self._chunker
        structurer = self._tree_structurer
        if chunker is None or structurer is None:
            llm = LLMService()
            if chunker is None:
                chunker = ChunkingService(llm)
            if structurer is None:
                structurer = TreeBuilder(llm)
        return chunker, structurer

    # --- Stages -----------------------------------------------------------
    # Each stage commits, so they run either back-to-back in one process
    # (run()) or across separate CLI invocations with structure supplied
    # in between.

    def chunk_stage(self, session: Session, document_id: uuid.UUID, chunker: Chunker) -> None:
        """Segment the document and persist the resulting chunks."""
        repo = DocumentRepository()
        doc = repo.get_by_id(session, document_id)
        if doc is None or doc.content is None:
            raise ChunkingError(f"Document {document_id} not found or has no content")

        chunks = chunker.segment(doc.content)
        ChunkRepository().insert_chunks(session, document_id, chunks)
        session.commit()

    def load_chunks(self, session: Session, document_id: uuid.UUID) -> list[ChunkModel]:
        """Fetch a document's chunks in document order."""
        result = session.execute(
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .order_by(ChunkModel.position)
        )
        return list(result.scalars().all())

    def tree_stage(
        self,
        session: Session,
        document_id: uuid.UUID,
        structurer: TreeStructurer,
    ) -> list[ChunkModel]:
        """Build the node tree from persisted chunks and link leaves to chunks."""
        db_chunks = self.load_chunks(session, document_id)

        node_inserts = structurer.build(db_chunks)

        node_repo = NodeRepository()
        node_repo.insert_tree(session, document_id, node_inserts)

        chunk_links = [
            (ni.id, ni.chunk_id)
            for ni in node_inserts
            if ni.node_type == "leaf" and ni.chunk_id is not None
        ]
        node_repo.link_chunks_to_leaves(session, chunk_links)
        session.commit()
        return db_chunks

    def embed_stage(self, session: Session, document_id: uuid.UUID) -> int:
        """Embed every node of the document and persist the vectors."""
        node_result = session.execute(
            select(Node).where(Node.document_id == document_id).order_by(Node.depth, Node.position)
        )
        all_nodes = list(node_result.scalars().all())

        node_texts = [
            (n.id, _build_embedding_text(n.node_type, n.toc_path, n.summary, n.content))
            for n in all_nodes
        ]

        embed_svc = EmbeddingService()
        node_embeddings = embed_svc.embed_nodes(
            node_texts,
            batch_size=settings.embedding_batch_size,
        )

        for node_id, vector in node_embeddings:
            session.execute(update(Node).where(Node.id == node_id).values(embedding=vector))
        session.commit()
        return len(node_embeddings)

    def run(self, session: Session, document_id: uuid.UUID) -> None:
        """Run the full ingestion pipeline synchronously."""
        chunker, structurer = self._resolve()

        self.chunk_stage(session, document_id, chunker)
        db_chunks = self.tree_stage(session, document_id, structurer)
        embedded = self.embed_stage(session, document_id)

        logger.info(
            "Ingestion complete for document %s (%d chunks, %d embeddings)",
            document_id,
            len(db_chunks),
            embedded,
        )


def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline()
