"""NodeRepository: bulk tree insert and chunk linkage.

Design notes:
- NodeInsert is a dataclass (not Pydantic) because it's a pure in-memory
  intermediate before DB persistence — no validation needed at this layer.
- Pre-generated UUIDs in NodeInsert.id allow parent_id references to be
  resolved in-memory before the flush, so all Node rows can be inserted
  in a single session.add_all() + flush(). PostgreSQL defers self-referential
  FK constraint checks within the same transaction.
- link_chunks_to_leaves() uses individual UPDATE statements per chunk_id;
  bulk update via IN clause is a valid future optimization if needed.
"""

import uuid
from dataclasses import dataclass, field

from sqlalchemy import select, text, update
from sqlalchemy.orm import Session

from openfable.models.chunk import Chunk
from openfable.models.node import Node


@dataclass
class NodeInsert:
    """Intermediate representation of a node before DB persistence.

    All fields map directly to Node ORM columns. Pre-generated id allows
    parent_id cross-references to be resolved before the database flush.
    """

    node_type: str
    depth: int
    position: int
    title: str | None
    summary: str | None
    toc_path: str | None
    content: str | None
    token_count: int | None
    parent_id: uuid.UUID | None = None
    id: uuid.UUID = field(default_factory=uuid.uuid4)
    path: str = ""
    chunk_id: uuid.UUID | None = None  # set only for leaf nodes, used for link_chunks_to_leaves


class NodeRepository:
    """Repository for persisting tree nodes and linking chunks to leaf nodes."""

    def insert_tree(
        self,
        session: Session,
        document_id: uuid.UUID,
        nodes: list[NodeInsert],
    ) -> list[Node]:
        """Persist a flat list of NodeInsert objects as Node ORM rows.

        Pre-generated UUIDs in NodeInsert allow self-referential parent_id
        references to resolve within a single session.add_all() + flush() call.
        PostgreSQL resolves FK constraints at commit time, not at flush.

        Returns the list of Node ORM objects (with DB-populated created_at).
        """
        node_models = [
            Node(
                id=n.id,
                document_id=document_id,
                parent_id=n.parent_id,
                path=n.path,
                node_type=n.node_type,
                depth=n.depth,
                position=n.position,
                title=n.title,
                summary=n.summary,
                toc_path=n.toc_path,
                content=n.content,
                token_count=n.token_count,
            )
            for n in nodes
        ]
        session.add_all(node_models)
        session.flush()
        return node_models

    def link_chunks_to_leaves(
        self,
        session: Session,
        chunk_links: list[tuple[uuid.UUID, uuid.UUID]],
    ) -> None:
        """Update Chunk.node_id for each (node_id, chunk_id) pair.

        Caller is responsible for flush/commit after this call.
        """
        for node_id, chunk_id in chunk_links:
            session.execute(update(Chunk).where(Chunk.id == chunk_id).values(node_id=node_id))

    def find_similar_internal_nodes(
        self,
        session: Session,
        query_vector: list[float],
        top_k: int,
    ) -> list[tuple[uuid.UUID, uuid.UUID, float]]:
        """Top-K internal nodes by cosine similarity using HNSW partial index.

        Returns list of (node_id, document_id, similarity_score) tuples.
        similarity = 1 - cosine_distance; range [0, 1].
        WHERE clause matches partial index predicate idx_nodes_embedding_internal.
        """
        result = session.execute(
            text("""
                SELECT id, document_id,
                       1 - (embedding <=> (:query_vec)::vector) AS similarity
                FROM nodes
                WHERE node_type IN ('root', 'section', 'subsection')
                  AND embedding IS NOT NULL
                ORDER BY embedding <=> (:query_vec)::vector
                LIMIT :top_k
            """),
            {"query_vec": str(query_vector), "top_k": top_k},
        )
        return [(row.id, row.document_id, float(row.similarity)) for row in result]

    def find_similar_nodes(
        self,
        session: Session,
        query_vector: list[float],
        top_k: int,
    ) -> list[tuple[uuid.UUID, uuid.UUID, float]]:
        """Top-K nodes (all types) by cosine similarity.

        Searches over all embedded nodes (internal + leaf) per FABLE paper
        Algorithm 1, line 7. Returns list of (node_id, document_id, similarity_score).
        """
        result = session.execute(
            text("""
                SELECT id, document_id,
                       1 - (embedding <=> (:query_vec)::vector) AS similarity
                FROM nodes
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> (:query_vec)::vector
                LIMIT :top_k
            """),
            {"query_vec": str(query_vector), "top_k": top_k},
        )
        return [(row.id, row.document_id, float(row.similarity)) for row in result]

    def find_internal_nodes_by_depth(
        self,
        session: Session,
        max_depth: int,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[Node]:
        """Fetch internal nodes at depth <= max_depth for LLMselect.

        If document_ids is None, fetches across all documents.
        Filters to node_type IN ('root', 'section', 'subsection') -- never leaf nodes.
        """
        stmt = (
            select(Node)
            .where(Node.depth <= max_depth)
            .where(Node.node_type.in_(["root", "section", "subsection"]))
        )
        if document_ids is not None:
            stmt = stmt.where(Node.document_id.in_(document_ids))
        stmt = stmt.order_by(Node.document_id, Node.depth, Node.position)
        result = session.execute(stmt)
        return list(result.scalars().all())

    def find_all_nodes_for_documents(
        self,
        session: Session,
        document_ids: list[uuid.UUID],
    ) -> list[Node]:
        """Fetch ALL nodes (all types, all depths) for given document IDs.

        Used by TreeExpansion for in-memory tree walking — returns every node
        regardless of type so the caller can traverse parent/child relationships.
        Ordered by document_id, depth, position for stable iteration.
        """
        stmt = (
            select(Node)
            .where(Node.document_id.in_(document_ids))
            .order_by(Node.document_id, Node.depth, Node.position)
        )
        result = session.execute(stmt)
        return list(result.scalars().all())

    def find_leaf_similarities_for_documents(
        self,
        session: Session,
        query_vector: list[float],
        document_ids: list[uuid.UUID],
    ) -> list[tuple[uuid.UUID, float]]:
        """Compute cosine similarity for all leaf nodes in the given documents.

        Uses the idx_nodes_embedding_leaf partial HNSW index (predicate: node_type = 'leaf').
        Returns ALL leaf nodes (not top-K) for the documents per D-12.
        Ordered by document_id, depth, position for deterministic iteration.

        Returns list of (node_id, similarity) tuples.
        """
        result = session.execute(
            text("""
                SELECT id, 1 - (embedding <=> (:query_vec)::vector) AS similarity
                FROM nodes
                WHERE node_type = 'leaf'
                  AND embedding IS NOT NULL
                  AND document_id = ANY(:doc_ids)
                ORDER BY document_id, depth, position
            """),
            {"query_vec": str(query_vector), "doc_ids": document_ids},
        )
        return [(row.id, float(row.similarity)) for row in result]


def get_node_repo() -> NodeRepository:
    """Factory function for NodeRepository."""
    return NodeRepository()
