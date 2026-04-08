import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class DocumentCreate(BaseModel):
    text: str = Field(..., min_length=1, description="Raw document text to ingest")


class DocumentIngestResponse(BaseModel):
    document_id: uuid.UUID
    content_hash: str


class DocumentStatusResponse(BaseModel):
    document_id: uuid.UUID
    content_hash: str
    llm_model: str
    token_count: int | None = None
    schema_version: int
    created_at: datetime
    updated_at: datetime
    content: str | None = None


class DocumentListItem(BaseModel):
    document_id: uuid.UUID
    content_hash: str
    token_count: int | None = None
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    total: int
