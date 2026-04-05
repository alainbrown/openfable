"""Initial schema: documents, ingestion_jobs, nodes, chunks with pgvector and ltree extensions.

Revision ID: 0001
Revises:
Create Date: 2026-04-04
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

# revision identifiers
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # --- Extensions (Pitfall 2: must come before any table using vector or ltree types) ---
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("CREATE EXTENSION IF NOT EXISTS ltree")

    # --- documents table ---
    op.create_table(
        "documents",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("schema_version", sa.Integer(), nullable=False, server_default=sa.text("1")),
        sa.Column("llm_model", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("index_status", sa.String(20), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint(
            "index_status IN ('pending', 'in_progress', 'complete', 'failed')",
            name="ck_documents_index_status",
        ),
    )
    op.create_index("idx_documents_status", "documents", ["index_status"])

    # --- ingestion_jobs table ---
    op.create_table(
        "ingestion_jobs",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("error_detail", sa.Text(), nullable=True),
        sa.Column("last_heartbeat", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint(
            "status IN ('pending', 'chunking', 'tree_building', 'embedding', 'indexing', 'complete', 'failed')",
            name="ck_ingestion_jobs_status",
        ),
    )
    op.create_index("idx_ingestion_jobs_document", "ingestion_jobs", ["document_id"])

    # --- nodes table (adjacency list + ltree + pgvector) ---
    op.create_table(
        "nodes",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("parent_id", UUID(as_uuid=True), sa.ForeignKey("nodes.id", ondelete="CASCADE"), nullable=True),
        sa.Column("node_type", sa.Text(), nullable=False),
        sa.Column("depth", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("toc_path", sa.Text(), nullable=True),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("embedding", Vector(1024), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
        sa.CheckConstraint(
            "node_type IN ('root', 'section', 'subsection', 'leaf')",
            name="ck_nodes_node_type",
        ),
    )
    # ltree path column — added via raw SQL because SQLAlchemy does not have a native ltree type
    op.execute("ALTER TABLE nodes ADD COLUMN path ltree NOT NULL DEFAULT ''")

    # Tree traversal indexes
    op.create_index("idx_nodes_document", "nodes", ["document_id"])
    op.create_index("idx_nodes_parent", "nodes", ["parent_id"])
    op.execute("CREATE INDEX idx_nodes_path_gist ON nodes USING GIST (path)")
    op.create_index("idx_nodes_type_depth", "nodes", ["node_type", "depth"])

    # HNSW partial vector indexes (Pitfall 4: separate indexes prevent internal/leaf contamination)
    # Using non-concurrent CREATE INDEX since table is empty in initial migration (Pitfall 3: CONCURRENTLY not allowed in transaction)
    op.execute("""
        CREATE INDEX idx_nodes_embedding_internal ON nodes
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE node_type IN ('root', 'section', 'subsection')
    """)
    op.execute("""
        CREATE INDEX idx_nodes_embedding_leaf ON nodes
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        WHERE node_type = 'leaf'
    """)

    # --- chunks table ---
    op.create_table(
        "chunks",
        sa.Column("id", UUID(as_uuid=True), server_default=sa.text("gen_random_uuid()"), primary_key=True),
        sa.Column("document_id", UUID(as_uuid=True), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("node_id", UUID(as_uuid=True), sa.ForeignKey("nodes.id", ondelete="CASCADE"), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("NOW()")),
    )
    op.create_index("idx_chunks_document", "chunks", ["document_id"])
    op.create_index("idx_chunks_node", "chunks", ["node_id"])


def downgrade() -> None:
    op.drop_table("chunks")
    op.drop_table("nodes")
    op.drop_table("ingestion_jobs")
    op.drop_table("documents")
    op.execute("DROP EXTENSION IF EXISTS ltree")
    op.execute("DROP EXTENSION IF EXISTS vector")
