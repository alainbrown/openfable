import hashlib
import uuid

import tiktoken
from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session

from openfable.config import settings
from openfable.models.chunk import Chunk
from openfable.models.document import Document, IngestionJob
from openfable.models.node import Node


def compute_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_tokens(text: str) -> int:
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


class DocumentRepository:
    def create_with_job(
        self,
        session: Session,
        text: str,
        content_hash: str,
        token_count: int,
    ) -> tuple[Document, IngestionJob]:
        doc = Document(
            content=text,
            content_hash=content_hash,
            llm_model=settings.litellm_model,
            token_count=token_count,
            index_status="pending",
        )
        session.add(doc)
        session.flush()  # flush to get doc.id for FK (Pitfall 5)

        job = IngestionJob(document_id=doc.id, status="pending")
        session.add(job)
        session.flush()
        return doc, job

    def get_by_id(self, session: Session, document_id: uuid.UUID) -> Document | None:
        result = session.execute(select(Document).where(Document.id == document_id))
        return result.scalar_one_or_none()

    def get_by_content_hash(
        self, session: Session, content_hash: str
    ) -> Document | None:
        result = session.execute(
            select(Document).where(Document.content_hash == content_hash)
        )
        return result.scalar_one_or_none()

    def list_all(self, session: Session) -> list[Document]:
        result = session.execute(select(Document).order_by(Document.created_at.desc()))
        return list(result.scalars().all())

    def delete(self, session: Session, document_id: uuid.UUID) -> None:
        session.execute(delete(Document).where(Document.id == document_id))
        # CASCADE FKs handle ingestion_jobs, nodes, chunks automatically

    def delete_associated_data(self, session: Session, document_id: uuid.UUID) -> None:
        """Delete nodes, chunks for a document (for idempotent re-ingest per D-01).
        Does NOT delete the document record itself — reuses the same document_id."""
        session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        session.execute(delete(Node).where(Node.document_id == document_id))
        session.execute(delete(IngestionJob).where(IngestionJob.document_id == document_id))

    def reset_document_for_reingest(
        self,
        session: Session,
        document_id: uuid.UUID,
        text: str,
        content_hash: str,
        token_count: int,
    ) -> IngestionJob:
        """Reset document for idempotent re-ingest (D-01): clear associated data,
        update document fields, create new job."""
        self.delete_associated_data(session, document_id)
        session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                content=text,
                content_hash=content_hash,
                token_count=token_count,
                llm_model=settings.litellm_model,
                index_status="pending",
                error_detail=None,
                updated_at=func.now(),
            )
        )
        job = IngestionJob(document_id=document_id, status="pending")
        session.add(job)
        session.flush()
        return job

    def get_latest_job(
        self, session: Session, document_id: uuid.UUID
    ) -> IngestionJob | None:
        result = session.execute(
            select(IngestionJob)
            .where(IngestionJob.document_id == document_id)
            .order_by(IngestionJob.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    def update_job_status(
        self,
        session: Session,
        job_id: uuid.UUID,
        status: str,
        error_detail: str | None = None,
    ) -> None:
        session.execute(
            update(IngestionJob)
            .where(IngestionJob.id == job_id)
            .values(
                status=status,
                error_detail=error_detail,
                updated_at=func.now(),
            )
        )

    def update_document_status(
        self,
        session: Session,
        document_id: uuid.UUID,
        status: str,
        error_detail: str | None = None,
    ) -> None:
        session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                index_status=status,
                error_detail=error_detail,
                updated_at=func.now(),
            )
        )


def get_document_repo() -> DocumentRepository:
    return DocumentRepository()
