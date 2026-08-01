"""The LLM boundary: LiteLLM, instructor, and Pydantic against a real endpoint.

Unit tests mock LLMService.complete_structured, so everything below it has
never executed: request construction, JSON-mode parsing into the response
models, the retry loop, and the exception wrapping. These tests exercise that
layer against a deterministic provider.

Behaviours are in agent-testkit.config.ts. Prompts, services and models are
production code -- only the provider base URL changes.
"""

import uuid

import pytest

from openfable.exceptions import ChunkingError, TreeConstructionError
from openfable.schemas.retrieval import LLMNavigateResult, LLMSelectResult
from openfable.schemas.tree import TreeMergeResponse
from openfable.services.ingestion.chunking import ChunkingService
from openfable.services.ingestion.tree_builder import TreeBuilder
from tests.integration.conftest import make_chunk

# ---------------------------------------------------------------------------
# Each response model parses from a real round trip
# ---------------------------------------------------------------------------


def test_chunking_response_parses(llm) -> None:
    """The chunking prompt round-trips into ChunkingResponse."""
    result = ChunkingService(llm).segment(
        "# Alpha\n\nFirst section body.\n\n## Beta\n\nSecond section body.\n"
    )

    assert len(result) == 2
    assert result[0].chunk_text.startswith("# Alpha")


def test_tree_build_response_parses(llm) -> None:
    """The tree prompt round-trips into TreeBuildResponse and becomes NodeInserts."""
    chunks = [make_chunk(0, "First section body."), make_chunk(1, "Second section body.")]

    nodes = TreeBuilder(llm).build(chunks)

    assert [n.node_type for n in nodes if n.node_type == "root"] == ["root"]
    assert len([n for n in nodes if n.node_type == "leaf"]) == 2


def test_tree_merge_response_parses(llm) -> None:
    result: TreeMergeResponse = llm.complete_structured(
        response_model=TreeMergeResponse,
        messages=[{"role": "user", "content": "Part 0: Title: A, Summary: a"}],
    )

    assert result.merged_title == "Merged"
    assert result.merged_summary


def test_llmselect_response_parses(llm) -> None:
    result: LLMSelectResult = llm.complete_structured(
        response_model=LLMSelectResult,
        messages=[{"role": "user", "content": "Query: q\n\nDocuments:\nDocument x:\n"}],
    )

    assert len(result.selected_documents) == 2


def test_llmnavigate_response_parses(llm) -> None:
    result: LLMNavigateResult = llm.complete_structured(
        response_model=LLMNavigateResult,
        messages=[{"role": "user", "content": "Query: q\n\nDocument tree:\n  [x] toc: summary\n"}],
    )

    assert len(result.selected_nodes) == 2


# ---------------------------------------------------------------------------
# The retry loop -- the part unit tests cannot reach
# ---------------------------------------------------------------------------


def test_retry_recovers_from_a_validation_failure(llm) -> None:
    """instructor re-prompts with the validator message, and the retry parses.

    The provider returns an empty title first, which schemas/tree.py rejects
    with "title must not be empty -- provide a descriptive topic title.".
    A second behaviour matches that text and returns a corrected tree, so a
    pass here proves the validator message actually reaches the model.
    """
    chunks = [make_chunk(0, "RETRY-CASE body")]

    nodes = TreeBuilder(llm).build(chunks)

    root = next(n for n in nodes if n.node_type == "root")
    assert root.title == "Recovered"


def test_retries_exhausted_becomes_tree_construction_error(llm) -> None:
    """A response that stays invalid raises TreeConstructionError, not InstructorRetryException."""
    chunks = [make_chunk(0, "EXHAUST-CASE body")]

    with pytest.raises(TreeConstructionError):
        TreeBuilder(llm).build(chunks)


def test_chunking_retries_exhausted_becomes_chunking_error(llm) -> None:
    """A response that never satisfies ChunkResult surfaces as ChunkingError.

    The provider omits start_idx and end_idx on every attempt, including the
    retries, so instructor exhausts them and chunking.py wraps the result.
    """
    with pytest.raises(ChunkingError):
        ChunkingService(llm)._chunk_window("CHUNK-FAIL body text")


# ---------------------------------------------------------------------------
# Validation that runs on real parsed output
# ---------------------------------------------------------------------------


def test_missing_chunk_index_is_rejected(llm) -> None:
    """A schema-valid tree that omits a chunk fails coverage validation."""
    chunks = [make_chunk(0, "COVERAGE-CASE body"), make_chunk(1, "second")]

    with pytest.raises(TreeConstructionError, match="Missing chunk indexes"):
        TreeBuilder(llm).build(chunks)


def test_llmselect_drops_hallucinated_document_ids(llm, session) -> None:
    """The guard keeps only ids that exist in the corpus."""
    from openfable.repositories.document_repo import DocumentRepository
    from openfable.services.retrieval_service import get_retrieval_service

    real_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    repo = DocumentRepository()
    doc = repo.create(session, "body", "hash-select", 10)
    session.commit()

    service = get_retrieval_service()
    service.llm = llm
    service.document_abstractions = lambda s: {  # type: ignore[method-assign]
        real_id: [("toc", "summary")],
        doc.id: [("toc", "summary")],
    }

    scores = service._llmselect(session, "q")

    assert uuid.UUID("99999999-9999-9999-9999-999999999999") not in scores
    assert real_id in scores


def test_llmnavigate_drops_hallucinated_node_ids(llm) -> None:
    """The node-level guard keeps only ids present in the fetched nodes."""
    from unittest.mock import MagicMock

    from openfable.services.retrieval_service import get_retrieval_service

    real_id = uuid.UUID("22222222-2222-2222-2222-222222222222")
    node = MagicMock()
    node.id, node.depth, node.toc_path, node.summary, node.title = real_id, 1, "toc", "summary", "t"

    service = get_retrieval_service()
    service.llm = llm
    service.node_repo = MagicMock()
    service.node_repo.find_internal_nodes_by_depth.return_value = [node]

    scores = service._llmnavigate(MagicMock(), "q", [uuid.uuid4()])

    assert uuid.UUID("88888888-8888-8888-8888-888888888888") not in scores
    assert real_id in scores
