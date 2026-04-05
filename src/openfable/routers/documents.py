import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from openfable.db import get_session
from openfable.repositories.document_repo import (
    DocumentRepository,
    compute_content_hash,
    count_tokens,
    get_document_repo,
)
from openfable.schemas.document import (
    DocumentCreate,
    DocumentIngestResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentStatusResponse,
    JobStatusEnum,
    JobStatusResponse,
)
from openfable.services.ingestion.pipeline import (
    IngestionPipeline,
    get_ingestion_pipeline,
)

router = APIRouter()


@router.post("/documents", response_model=DocumentIngestResponse)
def create_document(
    body: DocumentCreate,
    session: Session = Depends(get_session),
    repo: DocumentRepository = Depends(get_document_repo),
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),
) -> DocumentIngestResponse:
    """Submit a document for ingestion. Blocks until ingestion completes.

    If the same content is submitted again (same content_hash), the existing
    document_id is reused and re-ingested (D-01: idempotent re-ingest).
    """
    content_hash = compute_content_hash(body.text)
    token_count = count_tokens(body.text)

    existing = repo.get_by_content_hash(session, content_hash)
    if existing:
        job = repo.reset_document_for_reingest(
            session,
            existing.id,
            body.text,
            content_hash,
            token_count,
        )
        doc_id = existing.id
    else:
        doc, job = repo.create_with_job(
            session,
            body.text,
            content_hash,
            token_count,
        )
        doc_id = doc.id

    session.commit()
    pipeline.run(session, doc_id, job.id)

    return DocumentIngestResponse(
        document_id=doc_id,
        job_id=job.id,
        status=JobStatusEnum.complete,
        content_hash=content_hash,
    )


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    session: Session = Depends(get_session),
    repo: DocumentRepository = Depends(get_document_repo),
) -> DocumentListResponse:
    """List all documents with their index status.

    Content is never included in list responses (D-06: use GET /documents/{id}?include=content).
    """
    docs = repo.list_all(session)
    items = [
        DocumentListItem(
            document_id=doc.id,
            content_hash=doc.content_hash,
            index_status=doc.index_status,
            token_count=doc.token_count,
            created_at=doc.created_at,
        )
        for doc in docs
    ]
    return DocumentListResponse(documents=items, total=len(items))


@router.get("/documents/{document_id}", response_model=DocumentStatusResponse)
def get_document(
    document_id: uuid.UUID,
    meta_only: bool = Query(default=False),
    session: Session = Depends(get_session),
    repo: DocumentRepository = Depends(get_document_repo),
) -> DocumentStatusResponse:
    """Get document details with latest job status.

    Returns document content by default. Use ?meta_only=true for a lightweight
    response without the raw text.
    """
    doc = repo.get_by_id(session, document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")

    latest_job = repo.get_latest_job(session, document_id)
    job_response = None
    if latest_job is not None:
        job_response = JobStatusResponse(
            job_id=latest_job.id,
            status=JobStatusEnum(latest_job.status),
            error_detail=latest_job.error_detail,
            created_at=latest_job.created_at,
            updated_at=latest_job.updated_at,
        )

    return DocumentStatusResponse(
        document_id=doc.id,
        content_hash=doc.content_hash,
        index_status=doc.index_status,
        llm_model=doc.llm_model,
        token_count=doc.token_count,
        schema_version=doc.schema_version,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
        latest_job=job_response,
        content=None if meta_only else doc.content,
    )


