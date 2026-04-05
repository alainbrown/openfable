"""Unit tests for node fusion and budget control (RET-07, RET-08, RET-09).

Tests _node_fusion() and _budget_select() on RetrievalService as pure synchronous
logic -- no async, no mocking of DB/LLM/embedding needed. All tests use plain
`def` (not `async def`) because the methods under test are not async.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from openfable.repositories.document_repo import DocumentRepository
from openfable.repositories.node_repo import NodeRepository
from openfable.schemas.retrieval import ChunkResult, NodeResult, QueryResponse
from openfable.services.embedding_service import EmbeddingService
from openfable.services.llm_service import LLMService
from openfable.services.retrieval_service import RetrievalService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service():
    """RetrievalService with mock deps (only needed for method access, not called)."""
    return RetrievalService(
        llm_service=MagicMock(spec=LLMService),
        embedding_service=MagicMock(spec=EmbeddingService),
        node_repo=MagicMock(spec=NodeRepository),
        doc_repo=MagicMock(spec=DocumentRepository),
    )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _nr(
    doc_id: uuid.UUID,
    position: int,
    score: float,
    source: str,
    token_count: int = 50,
    content: str = "chunk text",
) -> NodeResult:
    """Build a NodeResult for testing."""
    return NodeResult(
        node_id=uuid.uuid4(),
        document_id=doc_id,
        content=content,
        token_count=token_count,
        score=score,
        depth=1,
        position=position,
        source=source,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# RET-07: Node fusion tests
# ---------------------------------------------------------------------------


def test_fusion_llm_before_tree(service) -> None:
    """_node_fusion returns llm_guided before tree_expansion regardless of score."""
    doc_a = uuid.uuid4()
    llm_result = _nr(doc_a, position=0, score=0.5, source="llm_guided")
    tree_result = _nr(doc_a, position=1, score=0.9, source="tree_expansion")

    # Pass tree_result first (higher score) -- llm_result should still be first in output
    result = service._node_fusion([tree_result, llm_result], [doc_a])

    assert result[0].source == "llm_guided"
    assert result[1].source == "tree_expansion"


def test_fusion_position_order_within_partition(service) -> None:
    """_node_fusion sorts within a partition by position ascending."""
    doc_a = uuid.uuid4()
    nr_pos2 = _nr(doc_a, position=2, score=0.7, source="llm_guided")
    nr_pos0 = _nr(doc_a, position=0, score=0.5, source="llm_guided")
    nr_pos1 = _nr(doc_a, position=1, score=0.9, source="llm_guided")

    result = service._node_fusion([nr_pos2, nr_pos0, nr_pos1], [doc_a])

    assert [r.position for r in result] == [0, 1, 2]


def test_fusion_cross_document_order(service) -> None:
    """_node_fusion orders by document rank from doc_order, not by score."""
    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()
    doc_b_chunk = _nr(doc_b, position=0, score=0.9, source="llm_guided")
    doc_a_chunk = _nr(doc_a, position=0, score=0.5, source="llm_guided")

    # doc_a is ranked first in doc_order despite lower score
    result = service._node_fusion([doc_b_chunk, doc_a_chunk], [doc_a, doc_b])

    assert result[0].document_id == doc_a
    assert result[1].document_id == doc_b


def test_source_tagging(service) -> None:
    """_node_fusion preserves source values from input NodeResults."""
    doc_a = uuid.uuid4()
    llm = _nr(doc_a, position=0, score=0.8, source="llm_guided")
    tree = _nr(doc_a, position=1, score=0.6, source="tree_expansion")

    result = service._node_fusion([llm, tree], [doc_a])

    assert result[0].source == "llm_guided"
    assert result[1].source == "tree_expansion"


# ---------------------------------------------------------------------------
# RET-08: Budget selection tests
# ---------------------------------------------------------------------------


def test_budget_greedy_includes_fitting(service) -> None:
    """_budget_select includes all chunks when total fits within budget."""
    doc_a = uuid.uuid4()
    chunks = [
        _nr(doc_a, i, 0.9 - i * 0.1, "llm_guided", token_count=tc)
        for i, tc in enumerate([50, 30, 20])
    ]

    selected, total, over = service._budget_select(chunks, 100)

    assert len(selected) == 3
    assert total == 100
    assert over is False


def test_budget_skip_and_continue(service) -> None:
    """_budget_select skips a chunk that would exceed budget but continues for smaller chunks."""
    doc_a = uuid.uuid4()
    c1 = _nr(doc_a, 0, 0.9, "llm_guided", token_count=50)
    c2 = _nr(doc_a, 1, 0.8, "llm_guided", token_count=80)  # would exceed budget
    c3 = _nr(doc_a, 2, 0.7, "llm_guided", token_count=30)

    selected, total, over = service._budget_select([c1, c2, c3], 100)

    assert len(selected) == 2  # c1 and c3; c2 skipped
    assert total == 80  # 50 + 30
    assert over is False
    assert selected[0].position == 0
    assert selected[1].position == 2


def test_budget_over_budget_fallback(service) -> None:
    """_budget_select returns single best-scored chunk with over_budget=True when nothing fits."""
    doc_a = uuid.uuid4()
    c1 = _nr(doc_a, 0, 0.9, "llm_guided", token_count=500)
    c2 = _nr(doc_a, 1, 0.7, "llm_guided", token_count=300)

    selected, total, over = service._budget_select([c1, c2], 100)

    assert len(selected) == 1
    assert selected[0].score == pytest.approx(0.9)  # best-scored chunk
    assert total == 500
    assert over is True


def test_budget_empty_input(service) -> None:
    """_budget_select with empty input returns ([], 0, False) with no error."""
    selected, total, over = service._budget_select([], 100)

    assert selected == []
    assert total == 0
    assert over is False


# ---------------------------------------------------------------------------
# RET-09: QueryResponse enrichment tests
# ---------------------------------------------------------------------------


def test_query_response_enriched(service) -> None:
    """ChunkResult built from budget-selected NodeResult has all required fields."""
    doc_a = uuid.uuid4()
    nr = _nr(doc_a, 0, 0.8, "llm_guided", token_count=50, content="test content")

    fusion_ordered = service._node_fusion([nr], [doc_a])
    selected, total, over = service._budget_select(fusion_ordered, 200)

    chunk = ChunkResult(
        node_id=selected[0].node_id,
        document_id=selected[0].document_id,
        content=selected[0].content,
        token_count=selected[0].token_count,
        score=selected[0].score,
        position=selected[0].position,
        source=selected[0].source,  # type: ignore[arg-type]
    )

    assert chunk.content == "test content"
    assert chunk.score == pytest.approx(0.8)
    assert chunk.token_count == 50
    assert chunk.position == 0
    assert chunk.source == "llm_guided"


def test_query_response_document_level_no_chunks() -> None:
    """QueryResponse with routing=document_level has chunks=None and over_budget=None."""
    qr = QueryResponse(
        query="test",
        routing="document_level",
        total_tokens=100,
        documents=[],
        chunks=None,
    )

    assert qr.chunks is None
    assert qr.over_budget is None


def test_total_tokens_used_accuracy(service) -> None:
    """total_tokens_used returned by _budget_select equals sum of selected chunk token_counts."""
    doc_a = uuid.uuid4()
    chunks = [
        _nr(doc_a, i, 0.9 - i * 0.1, "llm_guided", token_count=tc)
        for i, tc in enumerate([42, 58, 17])
    ]

    selected, total, over = service._budget_select(chunks, 200)

    assert total == sum(r.token_count for r in selected)
    assert total == 117  # 42 + 58 + 17
