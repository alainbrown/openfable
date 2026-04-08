import logging
import uuid

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from openfable.config import settings
from openfable.exceptions import ChunkingError
from openfable.models.chunk import Chunk as ChunkModel
from openfable.models.node import Node
from openfable.repositories.chunk_repo import ChunkRepository
from openfable.repositories.document_repo import DocumentRepository
from openfable.repositories.node_repo import NodeRepository
from openfable.services.embedding_service import EmbeddingService, _build_embedding_text
from openfable.services.ingestion.chunking import ChunkingService
from openfable.services.ingestion.tree_builder import TreeBuilder
from openfable.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class IngestionPipeline:
    def run(self, session: Session, document_id: uuid.UUID) -> None:
        """Run the full ingestion pipeline synchronously."""
        repo = DocumentRepository()
        llm = LLMService()
        chunking_svc = ChunkingService(llm)
        chunk_repo = ChunkRepository()

        # --- Stage: chunking ---
        doc = repo.get_by_id(session, document_id)
        if doc is None or doc.content is None:
            raise ChunkingError(f"Document {document_id} not found or has no content")

        chunks = chunking_svc.segment(doc.content)
        chunk_repo.insert_chunks(session, document_id, chunks)
        session.commit()

        # --- Stage: tree_building ---
        chunk_result = session.execute(
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .order_by(ChunkModel.position)
        )
        db_chunks = list(chunk_result.scalars().all())

        tree_builder = TreeBuilder(llm)
        node_inserts = tree_builder.build(db_chunks)

        node_repo = NodeRepository()
        inserted_nodes = node_repo.insert_tree(session, document_id, node_inserts)

        chunk_links = [
            (ni.id, ni.chunk_id)
            for ni in node_inserts
            if ni.node_type == "leaf" and ni.chunk_id is not None
        ]
        node_repo.link_chunks_to_leaves(session, chunk_links)
        session.commit()

        # --- Stage: embedding ---
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

        logger.info(
            "Ingestion complete for document %s (%d chunks, %d nodes, %d embeddings)",
            document_id,
            len(db_chunks),
            len(inserted_nodes),
            len(node_embeddings),
        )


def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline()
