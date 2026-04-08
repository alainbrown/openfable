import hashlib
import uuid

import tiktoken
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from openfable.config import settings
from openfable.models.chunk import Chunk
from openfable.models.document import Document
from openfable.models.node import Node


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


class DocumentRepository:
    def create(
        self,
        session: Session,
        text: str,
        content_hash: str,
        token_count: int,
    ) -> Document:
        doc = Document(
            content=text,
            content_hash=content_hash,
            llm_model=settings.litellm_model,
            token_count=token_count,
        )
        session.add(doc)
        session.flush()
        return doc

    def get_by_id(self, session: Session, document_id: uuid.UUID) -> Document | None:
        result = session.execute(select(Document).where(Document.id == document_id))
        return result.scalar_one_or_none()

    def get_by_content_hash(self, session: Session, content_hash: str) -> Document | None:
        result = session.execute(select(Document).where(Document.content_hash == content_hash))
        return result.scalar_one_or_none()

    def list_all(self, session: Session) -> list[Document]:
        result = session.execute(select(Document).order_by(Document.created_at.desc()))
        return list(result.scalars().all())

    def delete(self, session: Session, document_id: uuid.UUID) -> None:
        session.execute(delete(Document).where(Document.id == document_id))

    def delete_associated_data(self, session: Session, document_id: uuid.UUID) -> None:
        """Delete nodes, chunks for a document (for idempotent re-ingest per D-01).
        Does NOT delete the document record itself — reuses the same document_id."""
        session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        session.execute(delete(Node).where(Node.document_id == document_id))

    def reset_document_for_reingest(
        self,
        session: Session,
        document_id: uuid.UUID,
        text: str,
        content_hash: str,
        token_count: int,
    ) -> None:
        """Reset document for idempotent re-ingest (D-01): clear associated data,
        update document fields."""
        self.delete_associated_data(session, document_id)
        session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                content=text,
                content_hash=content_hash,
                token_count=token_count,
                llm_model=settings.litellm_model,
                updated_at=func.now(),
            )
        )


def get_document_repo() -> DocumentRepository:
    return DocumentRepository()
