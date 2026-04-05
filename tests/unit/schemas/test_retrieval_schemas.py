"""Unit tests for retrieval Pydantic schemas.

Tests cover QueryRequest validation (budget bounds, empty query rejection),
QueryResponse routing literal, DocumentResult optional content, LLMSelectResult
defaults, and DocumentSelection score bounds.
"""

import uuid

import pydantic
import pytest

from openfable.schemas.retrieval import (
    DocumentResult,
    DocumentSelection,
    LLMSelectResult,
    QueryRequest,
    QueryResponse,
)


def test_query_request_valid() -> None:
    """QueryRequest accepts a valid query and token_budget."""
    req = QueryRequest(query="test question", token_budget=500)
    assert req.query == "test question"
    assert req.token_budget == 500


def test_query_request_min_budget() -> None:
    """QueryRequest accepts token_budget=100 (minimum allowed per D-01/Pitfall 6)."""
    req = QueryRequest(query="q", token_budget=100)
    assert req.token_budget == 100


def test_query_request_max_budget() -> None:
    """QueryRequest accepts token_budget=32000 (maximum allowed per D-01)."""
    req = QueryRequest(query="q", token_budget=32000)
    assert req.token_budget == 32000


def test_query_request_rejects_budget_below_100() -> None:
    """QueryRequest rejects token_budget=99 with ValidationError."""
    with pytest.raises(pydantic.ValidationError):
        QueryRequest(query="q", token_budget=99)


def test_query_request_rejects_budget_above_32000() -> None:
    """QueryRequest rejects token_budget=32001 with ValidationError."""
    with pytest.raises(pydantic.ValidationError):
        QueryRequest(query="q", token_budget=32001)


def test_query_request_rejects_empty_query() -> None:
    """QueryRequest rejects empty string query (min_length=1)."""
    with pytest.raises(pydantic.ValidationError):
        QueryRequest(query="", token_budget=500)


def test_document_result_content_optional() -> None:
    """DocumentResult content defaults to None and can be set."""
    doc_id = uuid.uuid4()
    result_no_content = DocumentResult(
        document_id=doc_id,
        title="Test",
        score=0.9,
        token_count=100,
    )
    assert result_no_content.content is None

    result_with_content = DocumentResult(
        document_id=doc_id,
        title="Test",
        score=0.9,
        token_count=100,
        content="hello",
    )
    assert result_with_content.content == "hello"


def test_query_response_routing_literal() -> None:
    """QueryResponse accepts 'document_level' and 'node_level' routing; rejects invalid."""
    resp_doc = QueryResponse(
        query="q",
        routing="document_level",
        total_tokens=100,
        documents=[],
    )
    assert resp_doc.routing == "document_level"

    resp_node = QueryResponse(
        query="q",
        routing="node_level",
        total_tokens=100,
        documents=[],
    )
    assert resp_node.routing == "node_level"

    with pytest.raises(pydantic.ValidationError):
        QueryResponse(
            query="q",
            routing="invalid",  # type: ignore[arg-type]
            total_tokens=100,
            documents=[],
        )


def test_llmselect_result_default_empty() -> None:
    """LLMSelectResult defaults to an empty selected_documents list."""
    result = LLMSelectResult()
    assert result.selected_documents == []


def test_document_selection_score_bounds() -> None:
    """DocumentSelection rejects scores outside [0.0, 1.0]; accepts boundary values."""
    doc_id = uuid.uuid4()

    with pytest.raises(pydantic.ValidationError):
        DocumentSelection(document_id=doc_id, relevance_score=1.5)

    with pytest.raises(pydantic.ValidationError):
        DocumentSelection(document_id=doc_id, relevance_score=-0.1)

    # Boundary values are valid
    sel_min = DocumentSelection(document_id=doc_id, relevance_score=0.0)
    assert sel_min.relevance_score == 0.0

    sel_max = DocumentSelection(document_id=doc_id, relevance_score=1.0)
    assert sel_max.relevance_score == 1.0
