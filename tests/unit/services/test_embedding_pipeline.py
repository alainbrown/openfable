"""Unit tests for embedding pipeline: _build_embedding_text, embed_nodes, and pipeline stages.

All TEI calls are mocked — no live embedding server required. Tests cover:
- Multi-granularity text construction per node type (D-01, D-02)
- Batched embedding via embed_nodes (batch splitting, order preservation)
- TEI error propagation to EmbeddingError (HTTPStatusError, ConnectError)
- Pipeline embedding stage transitions (embedding -> complete)
- Pipeline embedding error handling (EmbeddingError -> raised)
"""

import uuid
from unittest.mock import MagicMock, patch

import httpx
import pytest

from openfable.exceptions import EmbeddingError
from openfable.services.embedding_service import EmbeddingService, _build_embedding_text
from openfable.services.ingestion.pipeline import IngestionPipeline

# ---------------------------------------------------------------------------
# Section 1: _build_embedding_text tests (synchronous, no mock needed)
# ---------------------------------------------------------------------------


def test_build_embedding_text_internal() -> None:
    """D-01: Internal node embeds as '{toc_path}: {summary}'."""
    result = _build_embedding_text("section", "root.intro", "An introduction to the topic", None)
    assert result == "root.intro: An introduction to the topic"


def test_build_embedding_text_leaf() -> None:
    """D-02: Leaf node embeds raw content."""
    result = _build_embedding_text("leaf", None, None, "The raw chunk content")
    assert result == "The raw chunk content"


def test_build_embedding_text_internal_none_fields() -> None:
    """D-01 with None toc_path and summary -> defensive '': ''."""
    result = _build_embedding_text("root", None, None, None)
    assert result == ": "


def test_build_embedding_text_leaf_none_content() -> None:
    """D-02 with None content -> empty string (defensive guard)."""
    result = _build_embedding_text("leaf", None, None, None)
    assert result == ""


# ---------------------------------------------------------------------------
# Section 2: EmbeddingService.embed_nodes tests (async, mock embed_batch)
# ---------------------------------------------------------------------------


def test_embed_nodes_batches_correctly() -> None:
    """5 nodes with batch_size=2 -> 3 embed_batch calls; 5 results in input order."""
    node_texts = [(uuid.uuid4(), f"text_{i}") for i in range(5)]

    def mock_embed(texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1024 for _ in texts]

    svc = EmbeddingService(embedding_url="http://test:80")
    with patch.object(svc, "embed_batch", side_effect=mock_embed) as mock_eb:
        results = svc.embed_nodes(node_texts, batch_size=2)

    # 3 batches: [0,1], [2,3], [4]
    assert mock_eb.call_count == 3
    assert len(results) == 5

    # Order preserved: result[i] matches node_texts[i] id
    for i, (result_id, vector) in enumerate(results):
        assert result_id == node_texts[i][0]
        assert len(vector) == 1024


def test_embed_nodes_single_batch() -> None:
    """3 nodes with batch_size=64 -> 1 embed_batch call; 3 results."""
    node_texts = [(uuid.uuid4(), f"text_{i}") for i in range(3)]

    def mock_embed(texts: list[str]) -> list[list[float]]:
        return [[0.2] * 1024 for _ in texts]

    svc = EmbeddingService(embedding_url="http://test:80")
    with patch.object(svc, "embed_batch", side_effect=mock_embed) as mock_eb:
        results = svc.embed_nodes(node_texts, batch_size=64)

    assert mock_eb.call_count == 1
    assert len(results) == 3


def test_embed_nodes_tei_http_error() -> None:
    """embed_batch raises HTTPStatusError -> embed_nodes raises EmbeddingError with status code."""
    mock_response = httpx.Response(
        status_code=500,
        text="Internal Server Error",
        request=httpx.Request("POST", "http://test:80/embed"),
    )
    exc = httpx.HTTPStatusError(
        "Server Error", request=mock_response.request, response=mock_response
    )

    svc = EmbeddingService(embedding_url="http://test:80")
    with patch.object(svc, "embed_batch", side_effect=exc):
        with pytest.raises(EmbeddingError) as exc_info:
            svc.embed_nodes([(uuid.uuid4(), "some text")], batch_size=64)

    assert "500" in str(exc_info.value)


def test_embed_nodes_tei_connect_error() -> None:
    """embed_batch raises ConnectError -> embed_nodes raises EmbeddingError with 'unreachable'."""
    svc = EmbeddingService(embedding_url="http://test:80")
    with patch.object(svc, "embed_batch", side_effect=httpx.ConnectError("Connection refused")):
        with pytest.raises(EmbeddingError) as exc_info:
            svc.embed_nodes([(uuid.uuid4(), "some text")], batch_size=64)

    assert "unreachable" in str(exc_info.value)


def test_embed_batch_parses_openai_format() -> None:
    """embed_batch sends OpenAI format and parses response correctly."""
    svc = EmbeddingService(embedding_url="http://test:80")
    openai_response = {
        "data": [
            {"embedding": [0.1, 0.2, 0.3], "index": 0},
            {"embedding": [0.4, 0.5, 0.6], "index": 1},
        ],
        "model": "BAAI/bge-m3",
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }
    mock_response = httpx.Response(
        status_code=200,
        json=openai_response,
        request=httpx.Request("POST", "http://test:80/v1/embeddings"),
    )

    with patch("httpx.Client.post", return_value=mock_response):
        result = svc.embed_batch(["hello", "world"])

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


# ---------------------------------------------------------------------------
# Section 3: Pipeline integration tests (heavy mocking)
# ---------------------------------------------------------------------------


def _setup_pipeline_mocks(
    mock_doc_repo_cls,
    mock_chunk_repo_cls,
    mock_chunking_cls,
    mock_tree_builder_cls,
    mock_node_repo_cls,
    mock_embed_svc_cls,
    mock_llm_cls,
    *,
    num_nodes: int = 3,
):
    """Set up mocks for a pipeline run that reaches the embedding stage."""
    # Document repo
    doc_repo = MagicMock()
    doc_repo.get_by_id = MagicMock()
    doc = MagicMock()
    doc.content = "Test document content"
    doc_repo.get_by_id.return_value = doc
    mock_doc_repo_cls.return_value = doc_repo

    # Chunk repo
    chunk_repo = MagicMock()
    chunk_repo.insert_chunks = MagicMock()
    mock_chunk_repo_cls.return_value = chunk_repo

    # Chunking service
    chunking = MagicMock()
    chunking.segment = MagicMock(return_value=[])
    mock_chunking_cls.return_value = chunking

    # Tree builder
    tree_builder = MagicMock()
    tree_builder.build = MagicMock(return_value=[])
    mock_tree_builder_cls.return_value = tree_builder

    # Node repo
    node_repo = MagicMock()
    node_repo.insert_tree = MagicMock(return_value=[])
    node_repo.link_chunks_to_leaves = MagicMock()
    mock_node_repo_cls.return_value = node_repo

    # Build mock nodes that session.execute(select(Node)...) will return
    mock_nodes = []
    for i in range(num_nodes):
        node = MagicMock()
        node.id = uuid.uuid4()
        node.node_type = "leaf" if i == num_nodes - 1 else "section"
        node.toc_path = f"root.section_{i}" if node.node_type != "leaf" else None
        node.summary = f"Summary {i}" if node.node_type != "leaf" else None
        node.content = f"Content {i}" if node.node_type == "leaf" else None
        mock_nodes.append(node)

    # session.execute returns different things at different points
    # chunk select, node select, then update(Node) per node
    node_scalars = MagicMock()
    node_scalars.all.return_value = mock_nodes
    node_result = MagicMock()
    node_result.scalars.return_value = node_scalars

    chunk_scalars = MagicMock()
    chunk_scalars.all.return_value = []
    chunk_result = MagicMock()
    chunk_result.scalars.return_value = chunk_scalars

    # Embedding service
    embed_svc = MagicMock()
    embeddings = [(n.id, [0.1] * 1024) for n in mock_nodes]
    embed_svc.embed_nodes = MagicMock(return_value=embeddings)
    mock_embed_svc_cls.return_value = embed_svc

    # LLM service
    mock_llm_cls.return_value = MagicMock()

    return doc_repo, embed_svc, mock_nodes, embeddings, chunk_result, node_result


@patch("openfable.services.ingestion.pipeline.LLMService")
@patch("openfable.services.ingestion.pipeline.EmbeddingService")
@patch("openfable.services.ingestion.pipeline.NodeRepository")
@patch("openfable.services.ingestion.pipeline.TreeBuilder")
@patch("openfable.services.ingestion.pipeline.ChunkingService")
@patch("openfable.services.ingestion.pipeline.ChunkRepository")
@patch("openfable.services.ingestion.pipeline.DocumentRepository")
def test_pipeline_completes(
    mock_doc_repo_cls,
    mock_chunk_repo_cls,
    mock_chunking_cls,
    mock_tree_builder_cls,
    mock_node_repo_cls,
    mock_embed_svc_cls,
    mock_llm_cls,
) -> None:
    """Pipeline transitions through stages and completes."""
    doc_id = uuid.uuid4()
    num_nodes = 3

    doc_repo, embed_svc, mock_nodes, _, chunk_result, node_result = _setup_pipeline_mocks(
        mock_doc_repo_cls,
        mock_chunk_repo_cls,
        mock_chunking_cls,
        mock_tree_builder_cls,
        mock_node_repo_cls,
        mock_embed_svc_cls,
        mock_llm_cls,
        num_nodes=num_nodes,
    )

    session = MagicMock()
    # session.execute: chunk select, node select, then UPDATE per node
    session.execute = MagicMock(side_effect=[chunk_result, node_result] + [MagicMock()] * num_nodes)

    pipeline = IngestionPipeline()
    pipeline.run(session, doc_id)

    # Verify embedding was called
    embed_svc.embed_nodes.assert_called_once()


@patch("openfable.services.ingestion.pipeline.LLMService")
@patch("openfable.services.ingestion.pipeline.EmbeddingService")
@patch("openfable.services.ingestion.pipeline.NodeRepository")
@patch("openfable.services.ingestion.pipeline.TreeBuilder")
@patch("openfable.services.ingestion.pipeline.ChunkingService")
@patch("openfable.services.ingestion.pipeline.ChunkRepository")
@patch("openfable.services.ingestion.pipeline.DocumentRepository")
def test_pipeline_embedding_stage_calls_embed_nodes(
    mock_doc_repo_cls,
    mock_chunk_repo_cls,
    mock_chunking_cls,
    mock_tree_builder_cls,
    mock_node_repo_cls,
    mock_embed_svc_cls,
    mock_llm_cls,
) -> None:
    """Pipeline calls embed_nodes with (node_id, text) pairs built from _build_embedding_text."""
    doc_id = uuid.uuid4()
    num_nodes = 3

    doc_repo, embed_svc, mock_nodes, _, chunk_result, node_result = _setup_pipeline_mocks(
        mock_doc_repo_cls,
        mock_chunk_repo_cls,
        mock_chunking_cls,
        mock_tree_builder_cls,
        mock_node_repo_cls,
        mock_embed_svc_cls,
        mock_llm_cls,
        num_nodes=num_nodes,
    )

    session = MagicMock()
    session.execute = MagicMock(side_effect=[chunk_result, node_result] + [MagicMock()] * num_nodes)

    pipeline = IngestionPipeline()
    pipeline.run(session, doc_id)

    embed_svc.embed_nodes.assert_called_once()
    call_args = embed_svc.embed_nodes.call_args

    # First positional arg is node_texts: list of (node_id, text) pairs
    node_texts_arg = call_args.args[0] if call_args.args else call_args.kwargs.get("node_texts")
    assert node_texts_arg is not None
    assert len(node_texts_arg) == len(mock_nodes)

    # Verify each (node_id, text) pair matches expected _build_embedding_text output
    for i, (node_id, text) in enumerate(node_texts_arg):
        node = mock_nodes[i]
        expected_text = _build_embedding_text(
            node.node_type, node.toc_path, node.summary, node.content
        )
        assert node_id == node.id
        assert text == expected_text


@patch("openfable.services.ingestion.pipeline.LLMService")
@patch("openfable.services.ingestion.pipeline.EmbeddingService")
@patch("openfable.services.ingestion.pipeline.NodeRepository")
@patch("openfable.services.ingestion.pipeline.TreeBuilder")
@patch("openfable.services.ingestion.pipeline.ChunkingService")
@patch("openfable.services.ingestion.pipeline.ChunkRepository")
@patch("openfable.services.ingestion.pipeline.DocumentRepository")
def test_pipeline_embedding_stage_writes_vectors(
    mock_doc_repo_cls,
    mock_chunk_repo_cls,
    mock_chunking_cls,
    mock_tree_builder_cls,
    mock_node_repo_cls,
    mock_embed_svc_cls,
    mock_llm_cls,
) -> None:
    """Pipeline issues UPDATE per node after embedding (2 SELECTs + num_nodes UPDATEs)."""
    doc_id = uuid.uuid4()
    num_nodes = 3

    doc_repo, embed_svc, mock_nodes, _, chunk_result, node_result = _setup_pipeline_mocks(
        mock_doc_repo_cls,
        mock_chunk_repo_cls,
        mock_chunking_cls,
        mock_tree_builder_cls,
        mock_node_repo_cls,
        mock_embed_svc_cls,
        mock_llm_cls,
        num_nodes=num_nodes,
    )

    session = MagicMock()
    session.execute = MagicMock(side_effect=[chunk_result, node_result] + [MagicMock()] * num_nodes)

    pipeline = IngestionPipeline()
    pipeline.run(session, doc_id)

    # session.execute calls: 1 chunk SELECT + 1 node SELECT + num_nodes UPDATE
    total_execute_calls = session.execute.call_count
    assert total_execute_calls == 2 + num_nodes


@patch("openfable.services.ingestion.pipeline.LLMService")
@patch("openfable.services.ingestion.pipeline.EmbeddingService")
@patch("openfable.services.ingestion.pipeline.NodeRepository")
@patch("openfable.services.ingestion.pipeline.TreeBuilder")
@patch("openfable.services.ingestion.pipeline.ChunkingService")
@patch("openfable.services.ingestion.pipeline.ChunkRepository")
@patch("openfable.services.ingestion.pipeline.DocumentRepository")
def test_pipeline_embedding_error_raises(
    mock_doc_repo_cls,
    mock_chunk_repo_cls,
    mock_chunking_cls,
    mock_tree_builder_cls,
    mock_node_repo_cls,
    mock_embed_svc_cls,
    mock_llm_cls,
) -> None:
    """EmbeddingError propagates from the pipeline."""
    doc_id = uuid.uuid4()
    num_nodes = 3

    doc_repo, embed_svc, mock_nodes, _, chunk_result, node_result = _setup_pipeline_mocks(
        mock_doc_repo_cls,
        mock_chunk_repo_cls,
        mock_chunking_cls,
        mock_tree_builder_cls,
        mock_node_repo_cls,
        mock_embed_svc_cls,
        mock_llm_cls,
        num_nodes=num_nodes,
    )

    # Override embed_nodes to raise EmbeddingError
    embed_svc.embed_nodes = MagicMock(side_effect=EmbeddingError("TEI down"))

    session = MagicMock()
    session.execute = MagicMock(side_effect=[chunk_result, node_result])

    pipeline = IngestionPipeline()
    with pytest.raises(EmbeddingError, match="TEI down"):
        pipeline.run(session, doc_id)
