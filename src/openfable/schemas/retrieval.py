import uuid
from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query")
    token_budget: int = Field(
        ...,
        ge=100,
        le=32000,
        description="Maximum tokens to return",
    )


class DocumentResult(BaseModel):
    document_id: uuid.UUID
    title: str | None
    score: float
    token_count: int | None
    content: str | None = None


class DocumentSelection(BaseModel):
    document_id: uuid.UUID
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score from 0.0 (not relevant) to 1.0 (highly relevant)",
    )


class LLMSelectResult(BaseModel):
    selected_documents: list[DocumentSelection] = Field(
        default_factory=list,
        description="Documents selected as relevant to the query, with scores",
    )


class NodeSelection(BaseModel):
    node_id: uuid.UUID
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score from 0.0 to 1.0",
    )


class LLMNavigateResult(BaseModel):
    selected_nodes: list[NodeSelection] = Field(
        default_factory=list,
        description="Non-leaf nodes selected as relevant subtree roots",
    )


class NodeResult(BaseModel):
    node_id: uuid.UUID
    document_id: uuid.UUID
    content: str | None
    token_count: int | None
    score: float
    depth: int
    position: int
    source: Literal["llm_guided", "tree_expansion"] | None = None


class ChunkResult(BaseModel):
    node_id: uuid.UUID
    document_id: uuid.UUID
    content: str | None
    token_count: int | None
    score: float
    position: int
    source: Literal["llm_guided", "tree_expansion"]


class QueryResponse(BaseModel):
    query: str
    routing: Literal["document_level", "node_level"]
    total_tokens: int
    documents: list[DocumentResult]
    node_results: list[NodeResult] | None = None
    chunks: list[ChunkResult] | None = None
    total_tokens_used: int | None = None
    over_budget: bool | None = None
