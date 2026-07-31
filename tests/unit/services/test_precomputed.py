"""Unit tests for the agent-driven ingestion path.

Covers:
- chunks_from_boundaries: marker resolution, whitespace tolerance, gap-free
  coverage, and the error messages a caller retries against
- PrecomputedTreeStructurer: reuses the LLM path's coverage validation
- IngestionPipeline injection: injected implementations are used, and no
  LLMService is constructed when both are supplied
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from openfable.exceptions import ChunkingError, TreeConstructionError
from openfable.schemas.tree import LLMInternalNode, LLMLeafNode
from openfable.services.ingestion.pipeline import IngestionPipeline
from openfable.services.ingestion.precomputed import (
    PrecomputedChunker,
    PrecomputedTreeStructurer,
    chunks_from_boundaries,
)

DOC = (
    "# Title\n\nIntro paragraph here.\n\n## Section A\n\nBody of A.\n\n## Section B\n\nBody of B.\n"
)


# ---------------------------------------------------------------------------
# chunks_from_boundaries
# ---------------------------------------------------------------------------


def test_boundaries_split_at_markers() -> None:
    """Each marker opens a chunk; offsets are computed, not supplied."""
    chunks = chunks_from_boundaries(DOC, ["# Title", "## Section A", "## Section B"])

    assert len(chunks) == 3
    assert chunks[0].chunk_text.startswith("# Title")
    assert chunks[1].chunk_text.startswith("## Section A")
    assert chunks[2].chunk_text.startswith("## Section B")


def test_boundaries_are_gap_free_and_exact() -> None:
    """Chunks tile the document exactly: every offset slice matches its text."""
    chunks = chunks_from_boundaries(DOC, ["# Title", "## Section A", "## Section B"])

    assert chunks[0].start_idx == 0
    assert chunks[-1].end_idx == len(DOC)
    for i, c in enumerate(chunks):
        assert DOC[c.start_idx : c.end_idx] == c.chunk_text
        if i > 0:
            assert c.start_idx == chunks[i - 1].end_idx


def test_boundaries_prepend_leading_text() -> None:
    """Text before the first marker becomes its own chunk rather than vanishing."""
    chunks = chunks_from_boundaries(DOC, ["## Section A", "## Section B"])

    assert len(chunks) == 3
    assert chunks[0].start_idx == 0
    assert chunks[0].chunk_text.startswith("# Title")


def test_boundaries_tolerate_whitespace_differences() -> None:
    """A marker whose internal whitespace differs still resolves."""
    chunks = chunks_from_boundaries(DOC, ["# Title", "##    Section A", "## Section B"])

    assert len(chunks) == 3
    assert chunks[1].chunk_text.startswith("## Section A")


def test_boundaries_repeated_phrase_advances() -> None:
    """A phrase appearing twice resolves to successive occurrences, not the same one."""
    text = "Alpha one.\nRepeat me.\nBeta two.\nRepeat me.\nGamma three.\n"
    chunks = chunks_from_boundaries(text, ["Repeat me.", "Repeat me."])

    assert len(chunks) == 3  # leading "Alpha one." + two repeats
    assert chunks[1].start_idx < chunks[2].start_idx


def test_boundaries_missing_marker_names_the_offender() -> None:
    """An unresolvable marker raises with its text, so the caller can fix and retry."""
    with pytest.raises(ChunkingError) as exc:
        chunks_from_boundaries(DOC, ["# Title", "## Section Q"])

    assert "## Section Q" in str(exc.value)


def test_boundaries_out_of_order_marker_rejected() -> None:
    """Markers must appear in document order; a backwards one cannot resolve."""
    with pytest.raises(ChunkingError):
        chunks_from_boundaries(DOC, ["## Section B", "## Section A"])


def test_boundaries_empty_list_rejected() -> None:
    with pytest.raises(ChunkingError, match="at least one marker"):
        chunks_from_boundaries(DOC, [])


def test_precomputed_chunker_satisfies_protocol() -> None:
    """PrecomputedChunker.segment() resolves against the text it is given."""
    chunker = PrecomputedChunker(["# Title", "## Section A", "## Section B"])
    assert len(chunker.segment(DOC)) == 3


# ---------------------------------------------------------------------------
# PrecomputedTreeStructurer
# ---------------------------------------------------------------------------


def _chunk(position: int) -> MagicMock:
    c = MagicMock()
    c.id = uuid.uuid4()
    c.content = f"chunk {position}"
    c.token_count = 5
    c.position = position
    return c


def _root_over(indexes: list[int]) -> LLMInternalNode:
    return LLMInternalNode(
        type="internal",
        node_type="root",
        title="Doc",
        summary="Summary",
        children=[LLMLeafNode(type="leaf", chunk_index=i) for i in indexes],
    )


def test_structurer_builds_nodes_for_full_coverage() -> None:
    """A tree covering every chunk yields root + one leaf per chunk."""
    chunks = [_chunk(i) for i in range(3)]
    nodes = PrecomputedTreeStructurer(_root_over([0, 1, 2])).build(chunks)

    assert len([n for n in nodes if n.node_type == "root"]) == 1
    assert len([n for n in nodes if n.node_type == "leaf"]) == 3


def test_structurer_rejects_missing_chunk() -> None:
    """Coverage validation from the LLM path applies to supplied trees too."""
    chunks = [_chunk(i) for i in range(3)]
    with pytest.raises(TreeConstructionError, match="Missing chunk indexes"):
        PrecomputedTreeStructurer(_root_over([0, 1])).build(chunks)


def test_structurer_rejects_duplicate_chunk() -> None:
    chunks = [_chunk(i) for i in range(2)]
    with pytest.raises(TreeConstructionError, match="Duplicate"):
        PrecomputedTreeStructurer(_root_over([0, 1, 1])).build(chunks)


def test_structurer_rejects_empty_chunks() -> None:
    with pytest.raises(TreeConstructionError, match="empty chunk list"):
        PrecomputedTreeStructurer(_root_over([0])).build([])


# ---------------------------------------------------------------------------
# Pipeline injection
# ---------------------------------------------------------------------------


@patch("openfable.services.ingestion.pipeline.LLMService")
def test_injected_implementations_skip_llm_service(mock_llm_cls: MagicMock) -> None:
    """A fully-injected pipeline never constructs LLMService -- so it needs no API key."""
    chunker, structurer = MagicMock(), MagicMock()
    resolved_chunker, resolved_structurer = IngestionPipeline(chunker, structurer)._resolve()

    assert resolved_chunker is chunker
    assert resolved_structurer is structurer
    mock_llm_cls.assert_not_called()


@patch("openfable.services.ingestion.pipeline.TreeBuilder")
@patch("openfable.services.ingestion.pipeline.ChunkingService")
@patch("openfable.services.ingestion.pipeline.LLMService")
def test_uninjected_pipeline_falls_back_to_litellm(
    mock_llm_cls: MagicMock,
    mock_chunking_cls: MagicMock,
    mock_tree_cls: MagicMock,
) -> None:
    """With nothing injected, the LiteLLM-backed defaults are constructed as before."""
    IngestionPipeline()._resolve()

    mock_llm_cls.assert_called_once()
    mock_chunking_cls.assert_called_once_with(mock_llm_cls.return_value)
    mock_tree_cls.assert_called_once_with(mock_llm_cls.return_value)


@patch("openfable.services.ingestion.pipeline.TreeBuilder")
@patch("openfable.services.ingestion.pipeline.ChunkingService")
@patch("openfable.services.ingestion.pipeline.LLMService")
def test_partial_injection_only_defaults_the_missing_side(
    mock_llm_cls: MagicMock,
    mock_chunking_cls: MagicMock,
    mock_tree_cls: MagicMock,
) -> None:
    """Injecting only a chunker still builds the default tree structurer."""
    chunker = MagicMock()
    resolved_chunker, _ = IngestionPipeline(chunker=chunker)._resolve()

    assert resolved_chunker is chunker
    mock_chunking_cls.assert_not_called()
    mock_tree_cls.assert_called_once()
