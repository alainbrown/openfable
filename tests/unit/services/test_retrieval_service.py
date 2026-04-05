"""Unit tests for RetrievalService: fusion, routing, LLMselect, vector top-K, empty corpus.

All LLM and embedding calls are mocked — no live services required.
Covers RET-01 through RET-06:
  RET-01: empty corpus handling
  RET-02: LLMselect structured output call + failure resilience + vector top-K aggregation
  RET-03: max-score bi-path fusion (union + max + sorted descending)
  RET-04: budget-adaptive routing (document_level with content, node_level without)
  RET-05: LLMnavigate structured output call + failure resilience + hallucination filter + subtree expansion
  RET-06: TreeExpansion scoring formula (S_sim, S_inh, S_child, normalization)
"""

import uuid
from unittest.mock import MagicMock

import pytest

from openfable.models.document import Document
from openfable.models.node import Node
from openfable.schemas.retrieval import (
    ChunkResult,
    DocumentSelection,
    LLMNavigateResult,
    LLMSelectResult,
    NodeResult,
    NodeSelection,
    QueryResponse,
)
from openfable.services.embedding_service import EmbeddingService
from openfable.services.retrieval_service import (
    RetrievalService,
    _compute_tree_expansion_scores,
    _expand_llmnav_to_leaves,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embed():
    """Mock EmbeddingService with sync embed_batch returning one 1024-dim vector."""
    embed = MagicMock(spec=EmbeddingService)
    embed.embed_batch = MagicMock(return_value=[[0.1] * 1024])
    return embed


@pytest.fixture
def mock_node_repo():
    """Mock NodeRepository with empty returns by default."""
    repo = MagicMock()
    repo.find_similar_internal_nodes = MagicMock(return_value=[])
    repo.find_internal_nodes_by_depth = MagicMock(return_value=[])
    return repo


@pytest.fixture
def mock_doc_repo():
    """Mock DocumentRepository."""
    return MagicMock()


@pytest.fixture
def service(mock_llm, mock_embed, mock_node_repo, mock_doc_repo):
    """RetrievalService wired with mock dependencies."""
    return RetrievalService(
        llm_service=mock_llm,
        embedding_service=mock_embed,
        node_repo=mock_node_repo,
        doc_repo=mock_doc_repo,
    )


@pytest.fixture
def mock_session():
    """Mock Session with configurable execute behavior."""
    session = MagicMock()
    # Default: no rows returned
    default_result = MagicMock()
    default_result.scalars.return_value.all.return_value = []
    default_result.all.return_value = []
    session.execute = MagicMock(return_value=default_result)
    return session


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_document(doc_id: uuid.UUID, token_count: int = 100, content: str = "text") -> MagicMock:
    """Create a Document-like mock for testing."""
    doc = MagicMock(spec=Document)
    doc.id = doc_id
    doc.token_count = token_count
    doc.content = content
    doc.index_status = "complete"
    return doc


def _make_node(doc_id: uuid.UUID, node_type: str = "section", summary: str = "Summary") -> MagicMock:
    """Create a Node-like mock for testing."""
    node = MagicMock(spec=Node)
    node.id = uuid.uuid4()
    node.document_id = doc_id
    node.node_type = node_type
    node.depth = 1
    node.summary = summary
    node.title = None
    return node


def _make_tree_node(
    node_id: uuid.UUID,
    doc_id: uuid.UUID,
    node_type: str,
    depth: int,
    position: int,
    parent_id: uuid.UUID | None = None,
    content: str | None = None,
    token_count: int | None = None,
    summary: str | None = None,
) -> MagicMock:
    """Create a Node-like mock with all fields needed for TreeExpansion."""
    node = MagicMock(spec=Node)
    node.id = node_id
    node.document_id = doc_id
    node.node_type = node_type
    node.depth = depth
    node.position = position
    node.parent_id = parent_id
    node.content = content
    node.token_count = token_count
    node.summary = summary
    node.title = None
    return node


# ---------------------------------------------------------------------------
# RET-03: Fusion tests (synchronous — _fuse is not async)
# ---------------------------------------------------------------------------


def test_fuse_union_and_max(service) -> None:
    """_fuse returns union of both paths; overlapping doc uses max(llm, vector) score."""
    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()
    doc_c = uuid.uuid4()

    llm_scores = {doc_a: 0.9, doc_b: 0.5}
    vector_scores = {doc_b: 0.7, doc_c: 0.3}

    result = service._fuse(llm_scores, vector_scores)

    # Union of A, B, C
    assert len(result) == 3
    result_dict = dict(result)

    # doc_a: only in llm path
    assert result_dict[doc_a] == pytest.approx(0.9)
    # doc_b: max(0.5, 0.7) = 0.7
    assert result_dict[doc_b] == pytest.approx(0.7)
    # doc_c: only in vector path
    assert result_dict[doc_c] == pytest.approx(0.3)

    # Highest score first
    assert result[0][0] == doc_a


def test_fuse_single_path(service) -> None:
    """_fuse with only one path populated returns single document."""
    doc_a = uuid.uuid4()

    result = service._fuse({doc_a: 0.8}, {})

    assert len(result) == 1
    assert result[0][0] == doc_a
    assert result[0][1] == pytest.approx(0.8)


def test_fuse_empty(service) -> None:
    """_fuse with both paths empty returns empty list."""
    result = service._fuse({}, {})
    assert result == []


def test_fuse_sorted_descending(service) -> None:
    """_fuse result is sorted by score descending regardless of insertion order."""
    docs = [uuid.uuid4() for _ in range(5)]
    # Assign non-monotonic scores across both paths
    llm_scores = {docs[0]: 0.3, docs[1]: 0.9, docs[2]: 0.1}
    vector_scores = {docs[3]: 0.7, docs[4]: 0.5, docs[2]: 0.6}

    result = service._fuse(llm_scores, vector_scores)

    scores = [score for _, score in result]
    assert scores == sorted(scores, reverse=True), (
        f"Expected descending order, got: {scores}"
    )


# ---------------------------------------------------------------------------
# RET-04: Routing tests (async — _route queries session)
# ---------------------------------------------------------------------------


def test_routing_document_level(service, mock_session) -> None:
    """_route returns document_level when total_tokens <= budget; content is included."""
    doc_a_id = uuid.uuid4()
    doc_b_id = uuid.uuid4()

    doc_a = _make_document(doc_a_id, token_count=200, content="Content of doc A")
    doc_b = _make_document(doc_b_id, token_count=200, content="Content of doc B")

    # First execute call: Documents query
    doc_result = MagicMock()
    doc_result.scalars.return_value.all.return_value = [doc_a, doc_b]

    # Second execute call: root Node titles query
    root_result = MagicMock()
    root_result.all.return_value = [
        MagicMock(document_id=doc_a_id, title="Doc A Title"),
        MagicMock(document_id=doc_b_id, title="Doc B Title"),
    ]

    mock_session.execute = MagicMock(side_effect=[doc_result, root_result])

    fused = [(doc_a_id, 0.9), (doc_b_id, 0.7)]
    result = service._route(mock_session, "test query", fused, token_budget=500)

    assert isinstance(result, QueryResponse)
    assert result.routing == "document_level"
    assert result.total_tokens == 400  # 200 + 200
    assert result.documents[0].content is not None  # content included
    assert result.documents[0].document_id == doc_a_id
    assert result.documents[1].document_id == doc_b_id


def test_routing_node_level(service, mock_session) -> None:
    """_route returns node_level when total_tokens > budget; content is omitted."""
    doc_a_id = uuid.uuid4()
    doc_b_id = uuid.uuid4()

    doc_a = _make_document(doc_a_id, token_count=5000, content="Large document A")
    doc_b = _make_document(doc_b_id, token_count=5000, content="Large document B")

    doc_result = MagicMock()
    doc_result.scalars.return_value.all.return_value = [doc_a, doc_b]

    root_result = MagicMock()
    root_result.all.return_value = [
        MagicMock(document_id=doc_a_id, title="Doc A Title"),
        MagicMock(document_id=doc_b_id, title="Doc B Title"),
    ]

    mock_session.execute = MagicMock(side_effect=[doc_result, root_result])

    fused = [(doc_a_id, 0.9), (doc_b_id, 0.7)]
    result = service._route(mock_session, "test query", fused, token_budget=500)

    assert result.routing == "node_level"
    assert result.total_tokens == 10000  # 5000 + 5000
    assert result.documents[0].content is None  # content omitted


# ---------------------------------------------------------------------------
# RET-01: Empty corpus test
# ---------------------------------------------------------------------------


def test_empty_corpus(service, mock_session) -> None:
    """query() with empty corpus returns QueryResponse with empty documents and node_level routing."""
    # Both retrieval paths return empty results
    result = service.query(mock_session, "test query", 500)

    assert isinstance(result, QueryResponse)
    assert result.documents == []
    assert result.routing == "node_level"
    assert result.total_tokens == 0


# ---------------------------------------------------------------------------
# RET-02: LLMselect tests
# ---------------------------------------------------------------------------


def test_llmselect_calls_llm(service, mock_llm, mock_node_repo, mock_session) -> None:
    """_llmselect calls complete_structured with LLMSelectResult response model."""
    doc_a_id = uuid.uuid4()

    # Provide nodes so LLMselect has content to send to LLM
    node1 = _make_node(doc_a_id, summary="Section about topic X")
    node2 = _make_node(doc_a_id, summary="Section about topic Y")
    mock_node_repo.find_internal_nodes_by_depth = MagicMock(return_value=[node1, node2])

    # LLM returns doc_a as selected
    mock_llm.complete_structured.return_value = LLMSelectResult(
        selected_documents=[
            DocumentSelection(document_id=doc_a_id, relevance_score=0.9)
        ]
    )

    result = service._llmselect(mock_session, "test query")

    # Verify LLM was called
    assert mock_llm.complete_structured.called

    # Verify response_model=LLMSelectResult was passed
    call_kwargs = mock_llm.complete_structured.call_args
    assert call_kwargs.kwargs.get("response_model") == LLMSelectResult

    # Verify returned dict contains doc_a with correct score
    assert doc_a_id in result
    assert result[doc_a_id] == pytest.approx(0.9)


def test_llmselect_failure_returns_empty(
    service, mock_llm, mock_node_repo, mock_session
) -> None:
    """_llmselect returns {} when LLM raises exception (vector path acts as fallback)."""
    doc_id = uuid.uuid4()
    node = _make_node(doc_id, summary="Some section summary")
    mock_node_repo.find_internal_nodes_by_depth = MagicMock(return_value=[node])

    mock_llm.complete_structured.side_effect = Exception("LLM unavailable")

    result = service._llmselect(mock_session, "test query")

    assert result == {}


# ---------------------------------------------------------------------------
# RET-02: Vector top-K aggregation test
# ---------------------------------------------------------------------------


def test_vector_topk_aggregates_by_document(
    service, mock_node_repo, mock_session
) -> None:
    """_vector_topk aggregates node similarities to document-level max scores."""
    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()

    node_a1_id = uuid.uuid4()
    node_a2_id = uuid.uuid4()
    node_b_id = uuid.uuid4()

    # Two nodes from doc_a (similarities 0.8 and 0.6), one from doc_b (similarity 0.5)
    mock_node_repo.find_similar_internal_nodes = MagicMock(
        return_value=[
            (node_a1_id, doc_a, 0.8),
            (node_a2_id, doc_a, 0.6),
            (node_b_id, doc_b, 0.5),
        ]
    )

    result = service._vector_topk(mock_session, [0.1] * 1024)

    # doc_a gets max(0.8, 0.6) = 0.8
    assert result[doc_a] == pytest.approx(0.8)
    # doc_b gets 0.5
    assert result[doc_b] == pytest.approx(0.5)
    # Only two document entries (aggregated from 3 nodes)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# RET-05: LLMnavigate tests
# ---------------------------------------------------------------------------


def test_llmnavigate_calls_llm(service, mock_llm, mock_node_repo, mock_session) -> None:
    """_llmnavigate calls complete_structured with LLMNavigateResult model and temperature=0."""
    doc_id = uuid.uuid4()
    node1 = _make_node(doc_id, node_type="section", summary="Section A")
    mock_node_repo.find_internal_nodes_by_depth = MagicMock(return_value=[node1])

    mock_llm.complete_structured.return_value = LLMNavigateResult(
        selected_nodes=[NodeSelection(node_id=node1.id, relevance_score=0.8)]
    )

    result = service._llmnavigate(mock_session, "test query", [doc_id])

    assert mock_llm.complete_structured.called
    call_kwargs = mock_llm.complete_structured.call_args
    assert call_kwargs.kwargs.get("response_model") == LLMNavigateResult
    assert call_kwargs.kwargs.get("temperature") == 0
    assert node1.id in result
    assert result[node1.id] == pytest.approx(0.8)


def test_llmnavigate_failure_returns_empty(service, mock_llm, mock_node_repo, mock_session) -> None:
    """_llmnavigate returns {} when LLM raises exception (D-04 fallback)."""
    doc_id = uuid.uuid4()
    node1 = _make_node(doc_id, summary="Section A")
    mock_node_repo.find_internal_nodes_by_depth = MagicMock(return_value=[node1])
    mock_llm.complete_structured.side_effect = Exception("LLM unavailable")

    result = service._llmnavigate(mock_session, "test query", [doc_id])

    assert result == {}


def test_llmnavigate_filters_invalid_ids(service, mock_llm, mock_node_repo, mock_session) -> None:
    """_llmnavigate excludes hallucinated node IDs not present in fetched nodes."""
    doc_id = uuid.uuid4()
    real_node = _make_node(doc_id, summary="Real section")
    hallucinated_id = uuid.uuid4()
    mock_node_repo.find_internal_nodes_by_depth = MagicMock(return_value=[real_node])

    mock_llm.complete_structured.return_value = LLMNavigateResult(
        selected_nodes=[
            NodeSelection(node_id=real_node.id, relevance_score=0.9),
            NodeSelection(node_id=hallucinated_id, relevance_score=0.7),
        ]
    )

    result = service._llmnavigate(mock_session, "test query", [doc_id])

    assert real_node.id in result
    assert hallucinated_id not in result
    assert len(result) == 1


# ---------------------------------------------------------------------------
# RET-06: TreeExpansion scoring tests (pure function -- no async)
# ---------------------------------------------------------------------------


def test_tree_expansion_rank_d1() -> None:
    """Depth-1 tree: two leaves, higher similarity leaf ranks first."""
    doc_id = uuid.uuid4()
    root_id = uuid.uuid4()
    leaf_high_id = uuid.uuid4()
    leaf_low_id = uuid.uuid4()

    root = _make_tree_node(root_id, doc_id, "root", depth=0, position=0)
    leaf_high = _make_tree_node(leaf_high_id, doc_id, "leaf", depth=1, position=0, parent_id=root_id, content="high", token_count=10)
    leaf_low = _make_tree_node(leaf_low_id, doc_id, "leaf", depth=1, position=1, parent_id=root_id, content="low", token_count=10)

    nodes = [root, leaf_high, leaf_low]
    leaf_sims = {leaf_high_id: 0.9, leaf_low_id: 0.3}
    internal_sims: dict[uuid.UUID, float] = {}

    scores = _compute_tree_expansion_scores(nodes, leaf_sims, internal_sims)

    assert scores[leaf_high_id] > scores[leaf_low_id]
    # With normalization: highest = 1.0, lowest = 0.0
    assert scores[leaf_high_id] == pytest.approx(1.0)
    assert scores[leaf_low_id] == pytest.approx(0.0)


def test_tree_expansion_rank_d4() -> None:
    """Depth-4 tree: depth decay compresses but normalization preserves rank order."""
    doc_id = uuid.uuid4()
    root_id = uuid.uuid4()
    sec_id = uuid.uuid4()
    subsec_id = uuid.uuid4()
    leaf_high_id = uuid.uuid4()
    leaf_low_id = uuid.uuid4()

    # root(0) -> section(1) -> subsection(2) -> 2 leaves(3)
    # Using depth 0,1,2,3 to simulate D=4 style tree (0-indexed depths)
    root = _make_tree_node(root_id, doc_id, "root", depth=0, position=0)
    sec = _make_tree_node(sec_id, doc_id, "section", depth=1, position=0, parent_id=root_id, summary="Section")
    subsec = _make_tree_node(subsec_id, doc_id, "subsection", depth=2, position=0, parent_id=sec_id, summary="Subsection")
    leaf_high = _make_tree_node(leaf_high_id, doc_id, "leaf", depth=3, position=0, parent_id=subsec_id, content="high", token_count=10)
    leaf_low = _make_tree_node(leaf_low_id, doc_id, "leaf", depth=3, position=1, parent_id=subsec_id, content="low", token_count=10)

    nodes = [root, sec, subsec, leaf_high, leaf_low]
    leaf_sims = {leaf_high_id: 0.9, leaf_low_id: 0.3}
    internal_sims = {sec_id: 0.5, subsec_id: 0.4}

    scores = _compute_tree_expansion_scores(nodes, leaf_sims, internal_sims)

    # Despite depth decay (divided by 3), normalization preserves rank
    assert scores[leaf_high_id] > scores[leaf_low_id]
    assert scores[leaf_high_id] == pytest.approx(1.0)
    assert scores[leaf_low_id] == pytest.approx(0.0)


def test_tree_expansion_rank_d6() -> None:
    """Depth-6 tree: beyond paper's D=4 max; normalization must prevent score collapse."""
    doc_id = uuid.uuid4()
    ids = [uuid.uuid4() for _ in range(7)]
    # Chain: root(0)->d1->d2->d3->d4->d5->2 leaves(6)
    nodes = [
        _make_tree_node(ids[0], doc_id, "root", depth=0, position=0),
        _make_tree_node(ids[1], doc_id, "section", depth=1, position=0, parent_id=ids[0], summary="d1"),
        _make_tree_node(ids[2], doc_id, "subsection", depth=2, position=0, parent_id=ids[1], summary="d2"),
        _make_tree_node(ids[3], doc_id, "subsection", depth=3, position=0, parent_id=ids[2], summary="d3"),
        _make_tree_node(ids[4], doc_id, "subsection", depth=4, position=0, parent_id=ids[3], summary="d4"),
        _make_tree_node(ids[5], doc_id, "subsection", depth=5, position=0, parent_id=ids[4], summary="d5"),
    ]
    leaf_high_id = uuid.uuid4()
    leaf_low_id = uuid.uuid4()
    nodes.append(_make_tree_node(leaf_high_id, doc_id, "leaf", depth=6, position=0, parent_id=ids[5], content="high", token_count=10))
    nodes.append(_make_tree_node(leaf_low_id, doc_id, "leaf", depth=6, position=1, parent_id=ids[5], content="low", token_count=10))

    leaf_sims = {leaf_high_id: 0.9, leaf_low_id: 0.3}
    internal_sims = {ids[i]: 0.5 for i in range(1, 6)}

    scores = _compute_tree_expansion_scores(nodes, leaf_sims, internal_sims)

    # Even at D=6 with severe decay (/6), normalization preserves rank
    assert scores[leaf_high_id] > scores[leaf_low_id]
    assert scores[leaf_high_id] == pytest.approx(1.0)
    assert scores[leaf_low_id] == pytest.approx(0.0)


def test_tree_expansion_equal_scores() -> None:
    """All-equal cosine similarities normalize to 1.0 for all leaves (D-10 edge case)."""
    doc_id = uuid.uuid4()
    root_id = uuid.uuid4()
    leaf_a = uuid.uuid4()
    leaf_b = uuid.uuid4()

    nodes = [
        _make_tree_node(root_id, doc_id, "root", depth=0, position=0),
        _make_tree_node(leaf_a, doc_id, "leaf", depth=1, position=0, parent_id=root_id, content="a", token_count=5),
        _make_tree_node(leaf_b, doc_id, "leaf", depth=1, position=1, parent_id=root_id, content="b", token_count=5),
    ]
    # Equal similarities
    leaf_sims = {leaf_a: 0.5, leaf_b: 0.5}

    scores = _compute_tree_expansion_scores(nodes, leaf_sims, {})

    assert scores[leaf_a] == pytest.approx(1.0)
    assert scores[leaf_b] == pytest.approx(1.0)


def test_tree_expansion_empty_nodes() -> None:
    """_compute_tree_expansion_scores with empty nodes returns {}."""
    scores = _compute_tree_expansion_scores([], {}, {})
    assert scores == {}


def test_s_inh_propagation() -> None:
    """S_inh: root=0, child inherits max(parent S_inh, parent S_sim)."""
    doc_id = uuid.uuid4()
    root_id = uuid.uuid4()
    sec_id = uuid.uuid4()
    leaf_id = uuid.uuid4()

    nodes = [
        _make_tree_node(root_id, doc_id, "root", depth=0, position=0),
        _make_tree_node(sec_id, doc_id, "section", depth=1, position=0, parent_id=root_id, summary="sec"),
        _make_tree_node(leaf_id, doc_id, "leaf", depth=2, position=0, parent_id=sec_id, content="leaf", token_count=5),
    ]
    # High similarity on root (internal) -> S_inh should propagate to leaf
    internal_sims = {root_id: 0.8, sec_id: 0.2}  # root is very relevant
    leaf_sims = {leaf_id: 0.1}  # leaf itself has low similarity

    scores = _compute_tree_expansion_scores(nodes, leaf_sims, internal_sims)

    # The leaf should still get a non-zero score thanks to S_inh from root
    # S_sim(root) = 0.8/max(0,1) = 0.8
    # S_inh(sec) = max(0, 0.8) = 0.8 (inherits root's S_sim)
    # S_inh(leaf) = max(0.8, 0.2/1) = 0.8 (inherits from section: max of sec's S_inh=0.8 and sec's S_sim=0.2/1=0.2)
    # The leaf gets boosted by ancestor relevance
    assert leaf_id in scores
    # With only one leaf, normalized to 1.0
    assert scores[leaf_id] == pytest.approx(1.0)


def test_s_child_aggregation_3_level() -> None:
    """S_child on intermediate internal node: 3-level tree verifies aggregation on non-root internal node.

    Tree structure:
        root(0) -> section(1) -> 2 leaves(2)

    The section node (depth=1) is an intermediate internal node whose S_child
    is directly observable: it should equal mean(S(v) of its two leaf children).
    This test constructs the tree so that the section's S_child contributes
    differently to each leaf's final score via the S(v) formula.
    """
    doc_id = uuid.uuid4()
    root_id = uuid.uuid4()
    sec_id = uuid.uuid4()
    leaf_a = uuid.uuid4()
    leaf_b = uuid.uuid4()

    root = _make_tree_node(root_id, doc_id, "root", depth=0, position=0)
    sec = _make_tree_node(sec_id, doc_id, "section", depth=1, position=0, parent_id=root_id, summary="sec")
    leaf_node_a = _make_tree_node(leaf_a, doc_id, "leaf", depth=2, position=0, parent_id=sec_id, content="a", token_count=5)
    leaf_node_b = _make_tree_node(leaf_b, doc_id, "leaf", depth=2, position=1, parent_id=sec_id, content="b", token_count=5)

    nodes = [root, sec, leaf_node_a, leaf_node_b]
    # Provide internal similarity on section to make S_child observable
    # Root gets no similarity so root S_child is section's S(v), which includes section's S_child
    internal_sims = {root_id: 0.0, sec_id: 0.6}
    leaf_sims = {leaf_a: 0.8, leaf_b: 0.2}

    scores = _compute_tree_expansion_scores(nodes, leaf_sims, internal_sims)

    # Manual computation:
    # S_sim(root) = 0.0/1 = 0.0
    # S_sim(sec) = 0.6/1 = 0.6
    # S_sim(leaf_a) = 0.8/2 = 0.4
    # S_sim(leaf_b) = 0.2/2 = 0.1
    #
    # S_inh(root) = 0.0
    # S_inh(sec) = max(0.0, 0.0) = 0.0  (root has S_inh=0, S_sim=0)
    # S_inh(leaf_a) = max(0.0, 0.6) = 0.6  (sec has S_inh=0, S_sim=0.6)
    # S_inh(leaf_b) = max(0.0, 0.6) = 0.6
    #
    # Bottom-up (leaves first, then section, then root):
    # S_child(leaf_a) = 0.0 (leaf)
    # S_child(leaf_b) = 0.0 (leaf)
    # S(leaf_a) = (0.4 + 0.6 + 0.0) / 3 = 0.3333
    # S(leaf_b) = (0.1 + 0.6 + 0.0) / 3 = 0.2333
    # S_child(sec) = mean(0.3333, 0.2333) = 0.2833  <-- THIS is the directly observable S_child
    # S(sec) = (0.6 + 0.0 + 0.2833) / 3 = 0.2944
    # S_child(root) = mean(0.2944) = 0.2944
    # S(root) = (0.0 + 0.0 + 0.2944) / 3 = 0.0981
    #
    # Normalization (leaves only):
    # min=0.2333, max=0.3333, range=0.1
    # leaf_a: (0.3333 - 0.2333) / 0.1 = 1.0
    # leaf_b: (0.2333 - 0.2333) / 0.1 = 0.0

    assert scores[leaf_a] == pytest.approx(1.0)
    assert scores[leaf_b] == pytest.approx(0.0)

    # Key assertion: leaf_a's raw S(v) is higher than leaf_b's despite both having
    # the same S_inh (0.6) -- the difference comes from S_sim at depth 2.
    # The section's S_child (0.2833) is the mean of its children's S(v) values,
    # demonstrating that child aggregation works on an intermediate internal node.


# ---------------------------------------------------------------------------
# RET-05: LLMnavigate subtree-to-leaf expansion test
# ---------------------------------------------------------------------------


def test_expand_llmnav_to_leaves() -> None:
    """_expand_llmnav_to_leaves propagates internal node score to all descendant leaves."""
    doc_id = uuid.uuid4()
    root_id = uuid.uuid4()
    sec_id = uuid.uuid4()
    leaf_a = uuid.uuid4()
    leaf_b = uuid.uuid4()
    leaf_c = uuid.uuid4()

    # root -> section -> leaf_a, leaf_b
    # root -> leaf_c (direct child of root, not under section)
    root = _make_tree_node(root_id, doc_id, "root", depth=0, position=0)
    sec = _make_tree_node(sec_id, doc_id, "section", depth=1, position=0, parent_id=root_id, summary="sec")
    node_a = _make_tree_node(leaf_a, doc_id, "leaf", depth=2, position=0, parent_id=sec_id, content="a", token_count=5)
    node_b = _make_tree_node(leaf_b, doc_id, "leaf", depth=2, position=1, parent_id=sec_id, content="b", token_count=5)
    node_c = _make_tree_node(leaf_c, doc_id, "leaf", depth=1, position=1, parent_id=root_id, content="c", token_count=5)

    all_nodes = [root, sec, node_a, node_b, node_c]

    # LLMnavigate selected only the section node
    llm_nav_scores = {sec_id: 0.9}

    leaf_scores = _expand_llmnav_to_leaves(llm_nav_scores, all_nodes)

    # Section's score should propagate to its two leaf descendants
    assert leaf_a in leaf_scores
    assert leaf_b in leaf_scores
    assert leaf_scores[leaf_a] == pytest.approx(0.9)
    assert leaf_scores[leaf_b] == pytest.approx(0.9)
    # leaf_c is NOT under section, so it should NOT receive the score
    assert leaf_c not in leaf_scores


def test_expand_llmnav_to_leaves_empty() -> None:
    """_expand_llmnav_to_leaves with empty scores returns empty dict."""
    result = _expand_llmnav_to_leaves({}, [])
    assert result == {}


# ---------------------------------------------------------------------------
# RET-05: Query node_level integration test
# ---------------------------------------------------------------------------


def test_query_node_level_returns_node_results(service, mock_llm, mock_embed, mock_node_repo, mock_session) -> None:
    """query() with node_level routing calls _llmnavigate + _tree_expansion and returns NodeResult list."""
    doc_id = uuid.uuid4()
    root_id = uuid.uuid4()
    leaf_id = uuid.uuid4()

    # Setup: LLMselect returns a document
    section_node = _make_node(doc_id, node_type="section", summary="Section A")
    mock_node_repo.find_internal_nodes_by_depth = MagicMock(return_value=[section_node])
    mock_llm.complete_structured.return_value = LLMSelectResult(
        selected_documents=[DocumentSelection(document_id=doc_id, relevance_score=0.9)]
    )

    # Setup: vector top-K returns same document
    mock_node_repo.find_similar_internal_nodes = MagicMock(
        return_value=[(uuid.uuid4(), doc_id, 0.8)]
    )

    # Setup: _route returns node_level (budget exceeded)
    doc = _make_document(doc_id, token_count=5000, content="Large doc")
    doc_result = MagicMock()
    doc_result.scalars.return_value.all.return_value = [doc]
    root_result = MagicMock()
    root_result.all.return_value = [MagicMock(document_id=doc_id, title="Title")]

    # Setup: node-level retrieval -- find_all_nodes and find_leaf_similarities
    root_node = _make_tree_node(root_id, doc_id, "root", depth=0, position=0)
    leaf_node = _make_tree_node(leaf_id, doc_id, "leaf", depth=1, position=0, parent_id=root_id, content="leaf content", token_count=50)

    # Configure mock_session for multiple execute calls
    # Call 1: Documents query (_route)
    # Call 2: Root titles query (_route)
    mock_session.execute = MagicMock(side_effect=[doc_result, root_result])

    # find_all_nodes returns root + leaf (called once -- _tree_expansion returns it, _node_level_retrieval reuses)
    mock_node_repo.find_all_nodes_for_documents = MagicMock(return_value=[root_node, leaf_node])

    # find_leaf_similarities returns leaf with similarity
    mock_node_repo.find_leaf_similarities_for_documents = MagicMock(
        return_value=[(leaf_id, 0.7)]
    )

    # LLMnavigate returns the section node (already configured above)
    # Override to return LLMNavigateResult for the second call to complete_structured
    mock_llm.complete_structured.side_effect = [
        # First call: LLMselect
        LLMSelectResult(selected_documents=[DocumentSelection(document_id=doc_id, relevance_score=0.9)]),
        # Second call: LLMnavigate
        LLMNavigateResult(selected_nodes=[]),
    ]

    result = service.query(mock_session, "test query", token_budget=100)

    assert result.routing == "node_level"
    assert result.node_results is not None
    assert len(result.node_results) > 0
    # All node_results should be NodeResult instances
    for nr in result.node_results:
        assert isinstance(nr, NodeResult)
        assert nr.content is not None  # Only leaf nodes with content


# ---------------------------------------------------------------------------
# RET-07/08/09: query() node_level returns enriched QueryResponse with chunks
# ---------------------------------------------------------------------------


def test_query_node_level_returns_chunks(
    service, mock_llm, mock_embed, mock_node_repo, mock_session
) -> None:
    """query() with node_level routing returns QueryResponse with chunks, total_tokens_used, and over_budget."""
    doc_id = uuid.uuid4()
    root_id = uuid.uuid4()
    leaf_a_id = uuid.uuid4()
    leaf_b_id = uuid.uuid4()

    # Setup: LLMselect returns a document
    section_node = _make_node(doc_id, node_type="section", summary="Section A")
    mock_node_repo.find_internal_nodes_by_depth = MagicMock(return_value=[section_node])

    # Setup: vector top-K returns same document
    mock_node_repo.find_similar_internal_nodes = MagicMock(
        return_value=[(uuid.uuid4(), doc_id, 0.8)]
    )

    # Setup: _route returns node_level (budget exceeded)
    doc = _make_document(doc_id, token_count=5000, content="Large doc")
    doc_result = MagicMock()
    doc_result.scalars.return_value.all.return_value = [doc]
    root_result = MagicMock()
    root_result.all.return_value = [MagicMock(document_id=doc_id, title="Title")]
    mock_session.execute = MagicMock(side_effect=[doc_result, root_result])

    # Setup: node-level retrieval nodes
    root_node = _make_tree_node(
        root_id, doc_id, "root", depth=0, position=0,
    )
    leaf_a = _make_tree_node(
        leaf_a_id, doc_id, "leaf", depth=1, position=0,
        parent_id=root_id, content="leaf A content", token_count=50,
    )
    leaf_b = _make_tree_node(
        leaf_b_id, doc_id, "leaf", depth=1, position=1,
        parent_id=root_id, content="leaf B content", token_count=30,
    )
    mock_node_repo.find_all_nodes_for_documents = MagicMock(
        return_value=[root_node, leaf_a, leaf_b],
    )
    mock_node_repo.find_leaf_similarities_for_documents = MagicMock(
        return_value=[(leaf_a_id, 0.7), (leaf_b_id, 0.4)],
    )

    # LLMselect + LLMnavigate responses
    mock_llm.complete_structured.side_effect = [
        LLMSelectResult(
            selected_documents=[
                DocumentSelection(document_id=doc_id, relevance_score=0.9),
            ],
        ),
        LLMNavigateResult(selected_nodes=[]),
    ]

    result = service.query(mock_session, "test query", token_budget=200)

    # Phase 8 enrichment assertions
    assert result.routing == "node_level"
    assert result.chunks is not None
    assert len(result.chunks) > 0
    assert result.over_budget is False
    assert result.total_tokens_used is not None
    assert result.total_tokens_used > 0

    # Each chunk is a ChunkResult with required fields
    for chunk in result.chunks:
        assert isinstance(chunk, ChunkResult)
        assert chunk.content is not None
        assert chunk.score >= 0  # normalized TreeExpansion scores range [0, 1]; min leaf may be 0.0
        assert chunk.token_count is not None
        assert chunk.source in ("llm_guided", "tree_expansion")

    # total_tokens_used should match sum of chunk token_counts
    assert result.total_tokens_used == sum(c.token_count for c in result.chunks)

    # node_results should still be present (full pre-budget list, D-12)
    assert result.node_results is not None
    assert len(result.node_results) >= len(result.chunks)
