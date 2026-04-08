import uuid
from collections.abc import Callable
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, DateTime, Index, Integer, Text, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import UserDefinedType

from openfable.db import Base


class LTreeType(UserDefinedType[str]):
    """SQLAlchemy type for PostgreSQL ltree column type.

    asyncpg sends Python strings as $1::VARCHAR which PostgreSQL rejects for ltree.
    This UserDefinedType renders as 'ltree' so asyncpg uses the correct type annotation.
    Bind processor ensures the value is a string (ltree labels must be plain text).
    """

    cache_ok = True

    def get_col_spec(self, **kw: object) -> str:
        return "ltree"

    def bind_processor(self, dialect: Dialect) -> Callable[[str | None], str | None] | None:
        def process(value: str | None) -> str | None:
            return value

        return process

    def result_processor(
        self, dialect: Dialect, coltype: object
    ) -> Callable[[object], str | None] | None:
        def process(value: object) -> str | None:
            return str(value) if value is not None else None

        return process


class Node(Base):
    __tablename__ = "nodes"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    document_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    parent_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    path: Mapped[str] = mapped_column(
        LTreeType, nullable=False
    )  # ltree column type via UserDefinedType
    node_type: Mapped[str] = mapped_column(Text, nullable=False)
    depth: Mapped[int] = mapped_column(Integer, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    title: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    toc_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding = mapped_column(Vector(1024), nullable=True)  # BGE-M3 dense, 1024-dim
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("NOW()")
    )

    __table_args__ = (
        CheckConstraint(
            "node_type IN ('root', 'section', 'subsection', 'leaf')",
            name="ck_nodes_node_type",
        ),
        Index("idx_nodes_document", "document_id"),
        Index("idx_nodes_parent", "parent_id"),
        Index("idx_nodes_type_depth", "node_type", "depth"),
    )
