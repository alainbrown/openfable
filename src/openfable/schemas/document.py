import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class JobStatusEnum(str, Enum):
    pending = "pending"
    chunking = "chunking"
    tree_building = "tree_building"
    embedding = "embedding"
    indexing = "indexing"
    complete = "complete"
    failed = "failed"


class DocumentCreate(BaseModel):
    text: str = Field(..., min_length=1, description="Raw document text to ingest")
    metadata: dict[str, str] = Field(default_factory=dict, description="Optional metadata tags")


class JobStatusResponse(BaseModel):
    job_id: uuid.UUID
    status: JobStatusEnum
    error_detail: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentIngestResponse(BaseModel):
    document_id: uuid.UUID
    job_id: uuid.UUID
    status: JobStatusEnum
    content_hash: str


class DocumentStatusResponse(BaseModel):
    document_id: uuid.UUID
    content_hash: str
    index_status: str
    llm_model: str
    token_count: int | None = None
    schema_version: int
    created_at: datetime
    updated_at: datetime
    latest_job: JobStatusResponse | None = None
    content: str | None = None


class DocumentListItem(BaseModel):
    document_id: uuid.UUID
    content_hash: str
    index_status: str
    token_count: int | None = None
    created_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentListItem]
    total: int
