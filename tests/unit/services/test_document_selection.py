"""Unit tests for caller-supplied document selection.

FABLE narrows the corpus with LLMselect before structure-aware scoring runs.
When no LLM is available that step is skipped, and the structural score ends up
doing document selection unaided. These tests cover the alternative: a caller
that is itself an LLM reads document_abstractions() and passes its choice to
query(document_ids=...).

Covers:
- document_abstractions: grouping, and that it is the same view LLMselect uses
- _preselected: validation, order preservation, de-duplication
- query(document_ids=...): both document-level paths are skipped entirely
"""

import uuid
from unittest.mock import MagicMock

import pytest

from openfable.exceptions import RetrievalError
from openfable.services.retrieval_service import RetrievalService


@pytest.fixture
def service():
    return RetrievalService(
        llm_service=MagicMock(),
        embedding_service=MagicMock(),
        node_repo=MagicMock(),
        doc_repo=MagicMock(),
    )


def _node(doc_id: uuid.UUID, toc: str | None, summary: str | None, title: str | None = None):
    n = MagicMock()
    n.document_id = doc_id
    n.toc_path = toc
    n.summary = summary
    n.title = title
    return n


# ---------------------------------------------------------------------------
# document_abstractions
# ---------------------------------------------------------------------------


def test_abstractions_group_by_document(service) -> None:
    """Sections are grouped under their document, in fetch order."""
    a, b = uuid.uuid4(), uuid.uuid4()
    service.node_repo.find_internal_nodes_by_depth.return_value = [
        _node(a, "Report", "Overall summary"),
        _node(a, "Report.Optics", "Alignment and throughput"),
        _node(b, "Notes", "Cache tunables"),
    ]

    result = service.document_abstractions(MagicMock())

    assert set(result) == {a, b}
    assert result[a] == [
        ("Report", "Overall summary"),
        ("Report.Optics", "Alignment and throughput"),
    ]
    assert result[b] == [("Notes", "Cache tunables")]


def test_abstractions_fall_back_when_fields_empty(service) -> None:
    """A node missing toc_path falls back to title, then to a placeholder."""
    doc = uuid.uuid4()
    service.node_repo.find_internal_nodes_by_depth.return_value = [
        _node(doc, None, None, title="Titled"),
        _node(doc, None, None, title=None),
    ]

    assert service.document_abstractions(MagicMock())[doc] == [
        ("Titled", "(no summary)"),
        ("(root)", "(no summary)"),
    ]


def test_abstractions_empty_corpus(service) -> None:
    service.node_repo.find_internal_nodes_by_depth.return_value = []
    assert service.document_abstractions(MagicMock()) == {}


def test_llmselect_reads_the_same_view(service, monkeypatch) -> None:
    """LLMselect consumes document_abstractions, so a caller sees what it would see."""
    doc = uuid.uuid4()
    seen = {}

    def spy(session):
        seen["called"] = True
        return {doc: [("Report", "Summary")]}

    monkeypatch.setattr(service, "document_abstractions", spy)
    service.llm.complete_structured.side_effect = RuntimeError("no LLM")

    service._llmselect(MagicMock(), "anything")

    assert seen.get("called"), "LLMselect must use the shared abstraction view"


# ---------------------------------------------------------------------------
# _preselected
# ---------------------------------------------------------------------------


def _session_returning(known_ids):
    session = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = list(known_ids)
    session.execute.return_value.scalars.return_value = scalars
    return session


def test_preselected_preserves_caller_order(service) -> None:
    """Order is the caller's ranking and must survive; every score is 1.0."""
    a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    result = service._preselected(_session_returning([a, b, c]), [c, a, b])

    assert [doc_id for doc_id, _ in result] == [c, a, b]
    assert {score for _, score in result} == {1.0}


def test_preselected_drops_duplicates(service) -> None:
    a, b = uuid.uuid4(), uuid.uuid4()
    result = service._preselected(_session_returning([a, b]), [a, b, a])

    assert [doc_id for doc_id, _ in result] == [a, b]


def test_preselected_rejects_unknown_ids(service) -> None:
    """An id not in the corpus fails loudly, naming it, rather than silently shrinking."""
    known, unknown = uuid.uuid4(), uuid.uuid4()

    with pytest.raises(RetrievalError) as exc:
        service._preselected(_session_returning([known]), [known, unknown])

    assert str(unknown) in str(exc.value)
    assert str(known) not in str(exc.value)


def test_preselected_rejects_empty_selection(service) -> None:
    with pytest.raises(RetrievalError, match="Omit the selection"):
        service._preselected(_session_returning([]), [])


# ---------------------------------------------------------------------------
# query(document_ids=...)
# ---------------------------------------------------------------------------


def test_query_with_selection_skips_both_document_paths(service, monkeypatch) -> None:
    """The whole point: no LLMselect call, and no vector top-K over the corpus."""
    doc = uuid.uuid4()
    service.embed.embed_batch.return_value = [[0.1] * 1024]

    called = {"llmselect": False, "vector": False}
    monkeypatch.setattr(
        service, "_llmselect", lambda *a: called.__setitem__("llmselect", True) or {}
    )
    monkeypatch.setattr(
        service, "_vector_topk", lambda *a: called.__setitem__("vector", True) or {}
    )
    monkeypatch.setattr(service, "_preselected", lambda s, ids: [(doc, 1.0)])
    routed = MagicMock()
    routed.routing = "document_level"
    routed.documents = []
    monkeypatch.setattr(service, "_route", lambda *a: routed)

    service.query(MagicMock(), "q", 2000, document_ids=[doc])

    assert called == {"llmselect": False, "vector": False}


def test_query_without_selection_runs_both_paths(service, monkeypatch) -> None:
    """Default behaviour is unchanged when no selection is supplied."""
    service.embed.embed_batch.return_value = [[0.1] * 1024]

    called = {"llmselect": False, "vector": False}
    monkeypatch.setattr(
        service, "_llmselect", lambda *a: called.__setitem__("llmselect", True) or {}
    )
    monkeypatch.setattr(
        service, "_vector_topk", lambda *a: called.__setitem__("vector", True) or {}
    )

    service.query(MagicMock(), "q", 2000)

    assert called == {"llmselect": True, "vector": True}
