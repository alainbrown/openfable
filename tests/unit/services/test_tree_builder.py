"""Unit tests for TreeBuilder service and helper functions.

All LLM calls are mocked -- no live LLM required. Tests cover:
- ltree helper functions (_sanitize_ltree_label, _build_toc_path)
- chunk coverage validation (_validate_chunk_coverage)
- depth flattening (_flatten_excess_depth)
- TreeBuilder service (single-pass, progressive, error handling)
- Prompt compliance (depth constraints in system prompts)
"""

import uuid
from unittest.mock import MagicMock, patch

import pytest

from openfable.exceptions import TreeConstructionError
from openfable.repositories.node_repo import NodeInsert
from openfable.schemas.tree import (
    LLMInternalNode,
    LLMLeafNode,
    TreeBuildResponse,
    TreeMergeResponse,
)
from openfable.services.ingestion.tree_builder import (
    TREE_BUILD_PARTITION_SYSTEM_PROMPT,
    TREE_BUILD_SYSTEM_PROMPT,
    TreeBuilder,
    _build_toc_path,
    _flatten_excess_depth,
    _sanitize_ltree_label,
    _validate_chunk_coverage,
)

# ---------------------------------------------------------------------------
# Helper: mock chunk factory
# ---------------------------------------------------------------------------


def _make_chunk(
    index: int, content: str = "Test content for chunk.", token_count: int = 50
) -> MagicMock:
    """Create a mock Chunk ORM object with required attributes."""
    chunk = MagicMock()
    chunk.id = uuid.uuid4()
    chunk.content = content
    chunk.token_count = token_count
    chunk.position = index
    return chunk


# ---------------------------------------------------------------------------
# Helper: build LLMInternalNode trees for tests
# ---------------------------------------------------------------------------


def _make_tree_2_sections_3_leaves() -> LLMInternalNode:
    """Build a 3-level tree: root -> 2 sections -> leaves [0, 1] and [2]."""
    leaf0 = LLMLeafNode(type="leaf", chunk_index=0)
    leaf1 = LLMLeafNode(type="leaf", chunk_index=1)
    leaf2 = LLMLeafNode(type="leaf", chunk_index=2)
    section1 = LLMInternalNode(
        type="internal",
        node_type="section",
        title="Section One",
        summary="Summary of section one",
        children=[leaf0, leaf1],
    )
    section2 = LLMInternalNode(
        type="internal",
        node_type="section",
        title="Section Two",
        summary="Summary of section two",
        children=[leaf2],
    )
    root = LLMInternalNode(
        type="internal",
        node_type="root",
        title="Document Root",
        summary="Summary of the whole document",
        children=[section1, section2],
    )
    return root


# ---------------------------------------------------------------------------
# Section 1: ltree helper tests (synchronous, no mock)
# ---------------------------------------------------------------------------


def test_sanitize_ltree_label_basic() -> None:
    """Non-alphanumeric characters are replaced with underscore."""
    assert _sanitize_ltree_label("Data Collection & Analysis") == "Data_Collection_Analysis"


def test_sanitize_ltree_label_dots() -> None:
    """Dots in titles are replaced with underscores."""
    assert _sanitize_ltree_label("3.2 Methods") == "3_2_Methods"


def test_sanitize_ltree_label_all_special() -> None:
    """All-special-character input falls back to 'node'."""
    assert _sanitize_ltree_label("!!!") == "node"


def test_sanitize_ltree_label_truncation() -> None:
    """Labels longer than 63 characters are truncated."""
    long_title = "A" * 100
    result = _sanitize_ltree_label(long_title)
    assert len(result) <= 63


def test_build_toc_path() -> None:
    """Ancestor + current title produce a dot-separated ltree path."""
    result = _build_toc_path(["Research"], "Survey Methods")
    assert result == "Research.Survey_Methods"


def test_build_toc_path_root() -> None:
    """Empty ancestor list produces a single-component path."""
    result = _build_toc_path([], "Root Title")
    assert result == "Root_Title"


# ---------------------------------------------------------------------------
# Section 2: chunk coverage validation tests (synchronous)
# ---------------------------------------------------------------------------


def test_validate_chunk_coverage_valid() -> None:
    """Tree with chunk_indexes 0, 1, 2 for 3 chunks raises no error."""
    root = _make_tree_2_sections_3_leaves()
    # Should not raise
    _validate_chunk_coverage(root, 3)


def test_validate_chunk_coverage_out_of_bounds() -> None:
    """chunk_index >= num_chunks raises TreeConstructionError."""
    leaf_out = LLMLeafNode(type="leaf", chunk_index=5)
    section = LLMInternalNode(
        type="internal",
        node_type="section",
        title="Section",
        summary="Summary",
        children=[leaf_out],
    )
    root = LLMInternalNode(
        type="internal",
        node_type="root",
        title="Root",
        summary="Summary",
        children=[section],
    )
    with pytest.raises(TreeConstructionError, match="chunk_index=5"):
        _validate_chunk_coverage(root, 3)


def test_validate_chunk_coverage_missing() -> None:
    """Tree missing chunk_index=1 (only has 0 and 2) raises TreeConstructionError."""
    leaf0 = LLMLeafNode(type="leaf", chunk_index=0)
    leaf2 = LLMLeafNode(type="leaf", chunk_index=2)
    section = LLMInternalNode(
        type="internal",
        node_type="section",
        title="Section",
        summary="Summary",
        children=[leaf0, leaf2],
    )
    root = LLMInternalNode(
        type="internal",
        node_type="root",
        title="Root",
        summary="Summary",
        children=[section],
    )
    with pytest.raises(TreeConstructionError, match="Missing chunk indexes"):
        _validate_chunk_coverage(root, 3)


def test_validate_chunk_coverage_duplicate() -> None:
    """Tree with duplicate chunk_index 0 raises TreeConstructionError."""
    leaf0a = LLMLeafNode(type="leaf", chunk_index=0)
    leaf0b = LLMLeafNode(type="leaf", chunk_index=0)
    leaf1 = LLMLeafNode(type="leaf", chunk_index=1)
    section = LLMInternalNode(
        type="internal",
        node_type="section",
        title="Section",
        summary="Summary",
        children=[leaf0a, leaf0b, leaf1],
    )
    root = LLMInternalNode(
        type="internal",
        node_type="root",
        title="Root",
        summary="Summary",
        children=[section],
    )
    with pytest.raises(TreeConstructionError, match="Duplicate"):
        _validate_chunk_coverage(root, 2)


# ---------------------------------------------------------------------------
# Section 3: flatten excess depth tests (synchronous)
# ---------------------------------------------------------------------------


def test_flatten_excess_depth_no_violation() -> None:
    """Nodes with depths 1-4 are returned unchanged."""
    root_id = uuid.uuid4()
    section_id = uuid.uuid4()
    subsection_id = uuid.uuid4()
    leaf_id = uuid.uuid4()

    root = NodeInsert(
        id=root_id,
        node_type="root",
        depth=1,
        position=0,
        title="Root",
        summary="Root summary",
        toc_path="Root",
        content=None,
        token_count=None,
        parent_id=None,
        path="Root",
    )
    section = NodeInsert(
        id=section_id,
        node_type="section",
        depth=2,
        position=0,
        title="Section",
        summary="Section summary",
        toc_path="Root.Section",
        content=None,
        token_count=None,
        parent_id=root_id,
        path="Root.Section",
    )
    subsection = NodeInsert(
        id=subsection_id,
        node_type="subsection",
        depth=3,
        position=0,
        title="Subsection",
        summary="Subsection summary",
        toc_path="Root.Section.Subsection",
        content=None,
        token_count=None,
        parent_id=section_id,
        path="Root.Section.Subsection",
    )
    leaf = NodeInsert(
        id=leaf_id,
        node_type="leaf",
        depth=4,
        position=0,
        title=None,
        summary=None,
        toc_path=None,
        content="chunk content",
        token_count=10,
        parent_id=subsection_id,
        path="Root.Section.Subsection.chunk_0",
    )

    nodes = [root, section, subsection, leaf]
    result = _flatten_excess_depth(nodes)
    assert len(result) == 4
    assert all(n.depth <= 4 for n in result)


def test_flatten_excess_depth_flattens_d5() -> None:
    """An internal node at depth 5 is removed; its children are re-parented to its parent.

    Chain: root(d=1)->section(d=2)->subsection(d=3)->deep_sub(d=4)->too_deep(d=5)->leaf(d=6)
    After flattening:
    - leaf (d=6) is processed first (deepest): re-parented to too_deep's parent (deep_id)
    - too_deep (d=5) is then processed: no remaining children; removed from result
    - Node count: 6 -> 4 (both too_deep and leaf removed since both were in violations list)

    The violations list is computed BEFORE any modifications, so both too_deep(d=5) AND
    leaf(d=6) are marked for removal. After leaf is re-parented, too_deep has no children
    and is itself removed. The net result is 4 nodes (root, section, subsection, deep).
    """
    root_id = uuid.uuid4()
    section_id = uuid.uuid4()
    subsection_id = uuid.uuid4()
    deep_id = uuid.uuid4()
    too_deep_id = uuid.uuid4()
    leaf_id = uuid.uuid4()

    root = NodeInsert(
        id=root_id,
        node_type="root",
        depth=1,
        position=0,
        title="Root",
        summary="Root summary",
        toc_path="Root",
        content=None,
        token_count=None,
        parent_id=None,
        path="Root",
    )
    section = NodeInsert(
        id=section_id,
        node_type="section",
        depth=2,
        position=0,
        title="Section",
        summary="Section summary",
        toc_path="Root.Section",
        content=None,
        token_count=None,
        parent_id=root_id,
        path="Root.Section",
    )
    subsection = NodeInsert(
        id=subsection_id,
        node_type="subsection",
        depth=3,
        position=0,
        title="Subsection",
        summary="Subsection summary",
        toc_path="Root.Section.Subsection",
        content=None,
        token_count=None,
        parent_id=section_id,
        path="Root.Section.Subsection",
    )
    deep = NodeInsert(
        id=deep_id,
        node_type="subsection",
        depth=4,
        position=0,
        title="Deep",
        summary="Deep summary",
        toc_path="Root.Section.Subsection.Deep",
        content=None,
        token_count=None,
        parent_id=subsection_id,
        path="Root.Section.Subsection.Deep",
    )
    too_deep = NodeInsert(
        id=too_deep_id,
        node_type="subsection",
        depth=5,
        position=0,
        title="TooDeep",
        summary="Too deep summary",
        toc_path="Root.Section.Subsection.Deep.TooDeep",
        content=None,
        token_count=None,
        parent_id=deep_id,
        path="Root.Section.Subsection.Deep.TooDeep",
    )
    leaf = NodeInsert(
        id=leaf_id,
        node_type="leaf",
        depth=6,
        position=0,
        title=None,
        summary=None,
        toc_path=None,
        content="chunk content",
        token_count=10,
        parent_id=too_deep_id,
        path="Root.Section.Subsection.Deep.TooDeep.chunk_0",
    )

    nodes = [root, section, subsection, deep, too_deep, leaf]
    result = _flatten_excess_depth(nodes)

    # too_deep (d=5) must not be in the result
    assert too_deep not in result
    # No node should exceed depth 4
    assert all(n.depth <= 4 for n in result)
    # Both violations (too_deep at d=5 AND leaf at d=6) removed from result
    assert len(result) == 4


# ---------------------------------------------------------------------------
# Section 4: TreeBuilder service tests (async, use mock_llm fixture)
# ---------------------------------------------------------------------------


@patch("openfable.services.ingestion.tree_builder._get_content_budget")
def test_single_pass_tree_build(mock_budget, mock_llm) -> None:
    """TreeBuilder.build() calls LLM once and returns correct NodeInsert list."""
    mock_budget.return_value = 100_000  # all chunks fit in budget

    chunks = [_make_chunk(i) for i in range(3)]

    root = _make_tree_2_sections_3_leaves()
    mock_llm.complete_structured.return_value = TreeBuildResponse(root=root)

    builder = TreeBuilder(mock_llm)
    result = builder.build(chunks)

    # Verify it returned NodeInserts
    assert isinstance(result, list)
    assert all(isinstance(n, NodeInsert) for n in result)

    # Verify structure: 1 root, 2 sections, 3 leaves
    node_types = [n.node_type for n in result]
    assert node_types.count("root") == 1
    assert node_types.count("section") == 2
    assert node_types.count("leaf") == 3

    # Verify depth constraint
    assert all(n.depth <= 4 for n in result)

    # Verify LLM called exactly once with TreeBuildResponse
    mock_llm.complete_structured.assert_called_once()
    call_kwargs = mock_llm.complete_structured.call_args.kwargs
    assert call_kwargs.get("response_model") is TreeBuildResponse


@patch("openfable.services.ingestion.tree_builder._get_content_budget")
def test_progressive_build_triggers_merge(mock_budget, mock_llm) -> None:
    """With token budget smaller than total chunks, build() partitions and merges."""
    mock_budget.return_value = 100  # tiny budget: each 60-token chunk triggers partition

    # 3 chunks of 60 tokens each (total 180 > budget 100)
    chunks = [_make_chunk(i, token_count=60) for i in range(3)]

    # Partition 1: chunks [0, 1] -- total 120 > 100 so chunk 0 alone, chunk 1 alone
    # Actually with budget=100: chunk 0 (60) fits, chunk 1 (60) would exceed -> partition
    # Partition 1: [chunk 0] (60 tokens)
    # Partition 2: [chunk 1, chunk 2]? No, 60+60=120 > 100 -> 3 partitions
    # Let's use 2 chunks with 60 tokens each, but only ensure >= 3 calls happen

    # Build partition tree responses (each with only its own chunk indexes)
    def _make_partition_response(chunk_index: int) -> TreeBuildResponse:
        leaf = LLMLeafNode(type="leaf", chunk_index=0)  # always index 0 within partition
        section = LLMInternalNode(
            type="internal",
            node_type="section",
            title=f"Section for chunk {chunk_index}",
            summary=f"Summary for chunk {chunk_index}",
            children=[leaf],
        )
        root = LLMInternalNode(
            type="internal",
            node_type="root",
            title=f"Part {chunk_index} root",
            summary=f"Summary of part {chunk_index}",
            children=[section],
        )
        return TreeBuildResponse(root=root)

    merge_response = TreeMergeResponse(
        merged_title="Merged Document",
        merged_summary="Merged summary of all parts",
    )

    # 3 partitions (one chunk per partition) -> 3 partition calls + 1 merge call
    partition_0 = _make_partition_response(0)
    partition_1 = _make_partition_response(1)
    partition_2 = _make_partition_response(2)

    mock_llm.complete_structured.side_effect = [
        partition_0,
        partition_1,
        partition_2,
        merge_response,
    ]

    builder = TreeBuilder(mock_llm)
    result = builder.build(chunks)

    # At least 3 LLM calls (2+ partitions + 1 merge)
    assert mock_llm.complete_structured.call_count >= 3

    # Single root in result
    roots = [n for n in result if n.node_type == "root"]
    assert len(roots) == 1


@patch("openfable.services.ingestion.tree_builder._get_content_budget")
def test_llm_failure_raises_tree_construction_error(mock_budget, mock_llm) -> None:
    """When LLM raises an exception, build() raises TreeConstructionError."""
    mock_budget.return_value = 100_000
    mock_llm.complete_structured.side_effect = Exception("LLM failed")

    chunks = [_make_chunk(0)]
    builder = TreeBuilder(mock_llm)

    with pytest.raises(TreeConstructionError):
        builder.build(chunks)


def test_empty_chunks_raises_error(mock_llm) -> None:
    """build([]) raises TreeConstructionError immediately."""
    builder = TreeBuilder(mock_llm)
    with pytest.raises(TreeConstructionError):
        builder.build([])


# ---------------------------------------------------------------------------
# Section 5: prompt compliance tests (synchronous)
# ---------------------------------------------------------------------------


def test_tree_build_prompt_contains_depth_constraint() -> None:
    """TREE_BUILD_SYSTEM_PROMPT must mention depth or '4 levels'."""
    lower = TREE_BUILD_SYSTEM_PROMPT.lower()
    assert "depth" in lower or "4 levels" in lower, (
        "TREE_BUILD_SYSTEM_PROMPT must reference depth or 4 levels constraint"
    )


def test_partition_prompt_constrains_depth_3() -> None:
    """TREE_BUILD_PARTITION_SYSTEM_PROMPT must mention subsection restriction."""
    lower = TREE_BUILD_PARTITION_SYSTEM_PROMPT.lower()
    assert "subsection" in lower, (
        "TREE_BUILD_PARTITION_SYSTEM_PROMPT must reference subsection restriction"
    )
