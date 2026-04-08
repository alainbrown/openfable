"""Unit tests for ChunkingService and related helpers.

All LLM calls are mocked — no live LLM required.
"""

import pytest

from openfable.exceptions import ChunkingError
from openfable.schemas.chunking import ChunkingResponse, ChunkResult
from openfable.services.ingestion.chunking import (
    SYSTEM_PROMPT,
    ChunkingService,
    _build_windows,
    _deduplicate_chunks,
    _split_sentences,
)

# ---------------------------------------------------------------------------
# Prompt compliance test (synchronous)
# ---------------------------------------------------------------------------


def test_system_prompt_has_no_size_constraints() -> None:
    """SYSTEM_PROMPT must not contain size constraints — D-01 compliance.

    Boundaries are purely discourse-driven; any size/limit hint biases the LLM
    toward arbitrary splits.
    """
    lower = SYSTEM_PROMPT.lower()
    assert "token" not in lower, "SYSTEM_PROMPT must not mention 'token'"
    assert "size" not in lower, "SYSTEM_PROMPT must not mention 'size'"
    assert "limit" not in lower, "SYSTEM_PROMPT must not mention 'limit'"


# ---------------------------------------------------------------------------
# Service tests (async, use mock_llm fixture)
# ---------------------------------------------------------------------------

# Sample text used across service tests — 3 paragraphs with different topics
_SAMPLE_TEXT = (
    "Machine learning is a branch of artificial intelligence that enables systems to learn "
    "and improve from experience without being explicitly programmed. It focuses on developing "
    "computer programs that can access data and use it to learn for themselves. "
    "The process begins with observations or data to look for patterns in data. "
    "\n\n"
    "Natural language processing (NLP) is another subfield of artificial intelligence "
    "concerned with the interactions between computers and human language. It involves "
    "enabling computers to understand, interpret and generate human language in a way "
    "that is both meaningful and useful for various applications. "
    "\n\n"
    "Computer vision is a field of artificial intelligence that trains computers to "
    "interpret and understand the visual world. Using digital images from cameras and "
    "videos, and deep learning models, machines can accurately identify and classify objects."
)

_CHUNK_1_TEXT = (
    "Machine learning is a branch of artificial intelligence that enables systems to learn "
    "and improve from experience without being explicitly programmed. It focuses on developing "
    "computer programs that can access data and use it to learn for themselves. "
    "The process begins with observations or data to look for patterns in data."
)
_CHUNK_2_TEXT = (
    "Natural language processing (NLP) is another subfield of artificial intelligence "
    "concerned with the interactions between computers and human language. It involves "
    "enabling computers to understand, interpret and generate human language in a way "
    "that is both meaningful and useful for various applications."
)
_CHUNK_3_TEXT = (
    "Computer vision is a field of artificial intelligence that trains computers to "
    "interpret and understand the visual world. Using digital images from cameras and "
    "videos, and deep learning models, machines can accurately identify and classify objects."
)


def test_segment_returns_discourse_chunks(mock_llm) -> None:
    """ChunkingService.segment() with mocked LLM returns ChunkResult list."""
    mock_response = ChunkingResponse(
        chunks=[
            ChunkResult(chunk_text=_CHUNK_1_TEXT, start_idx=0, end_idx=len(_CHUNK_1_TEXT)),
            ChunkResult(
                chunk_text=_CHUNK_2_TEXT,
                start_idx=len(_CHUNK_1_TEXT) + 2,
                end_idx=len(_CHUNK_1_TEXT) + 2 + len(_CHUNK_2_TEXT),
            ),
            ChunkResult(
                chunk_text=_CHUNK_3_TEXT,
                start_idx=len(_CHUNK_1_TEXT) + len(_CHUNK_2_TEXT) + 4,
                end_idx=len(_CHUNK_1_TEXT) + len(_CHUNK_2_TEXT) + 4 + len(_CHUNK_3_TEXT),
            ),
        ]
    )
    mock_llm.complete_structured.return_value = mock_response

    svc = ChunkingService(mock_llm)
    result = svc.segment(_SAMPLE_TEXT)

    assert len(result) == 3
    for item in result:
        assert isinstance(item, ChunkResult)


def test_llm_called_with_temperature_zero(mock_llm) -> None:
    """complete_structured must be called with temperature=0."""
    mock_response = ChunkingResponse(
        chunks=[
            ChunkResult(chunk_text=_CHUNK_1_TEXT, start_idx=0, end_idx=len(_CHUNK_1_TEXT)),
        ]
    )
    mock_llm.complete_structured.return_value = mock_response

    svc = ChunkingService(mock_llm)
    svc.segment(_SAMPLE_TEXT)

    call_kwargs = mock_llm.complete_structured.call_args
    assert call_kwargs is not None
    # temperature may be passed as keyword argument
    assert call_kwargs.kwargs.get("temperature") == 0


def test_llm_failure_raises_chunking_error(mock_llm) -> None:
    """When the LLM raises an exception, segment() raises ChunkingError."""
    mock_llm.complete_structured.side_effect = Exception("LLM failed")

    svc = ChunkingService(mock_llm)
    with pytest.raises(ChunkingError):
        svc.segment(_SAMPLE_TEXT)


# ---------------------------------------------------------------------------
# Helper function tests (synchronous)
# ---------------------------------------------------------------------------


def test_split_sentences_basic() -> None:
    """_split_sentences splits at sentence-ending punctuation boundaries."""
    text = "First sentence. Second sentence. Third one here."
    result = _split_sentences(text)
    assert len(result) == 3
    assert result[0].startswith("First")


def test_deduplicate_chunks_removes_exact_duplicates() -> None:
    """_deduplicate_chunks keeps only the first occurrence of duplicate chunk_text."""
    # Use valid multi-word texts that satisfy the 20-token minimum
    text_a = "The quick brown fox jumps over the lazy dog. " * 3
    text_b = (
        "Natural language processing involves enabling computers to understand human language "
        "in a way that is meaningful and useful for many different applications."
    )

    chunk1 = ChunkResult(chunk_text=text_a, start_idx=0, end_idx=len(text_a))
    chunk2 = ChunkResult(
        chunk_text=text_a, start_idx=len(text_a) + 10, end_idx=len(text_a) * 2 + 10
    )  # duplicate of chunk1
    offset_b = len(text_a) * 2 + 20
    chunk3 = ChunkResult(chunk_text=text_b, start_idx=offset_b, end_idx=offset_b + len(text_b))

    result = _deduplicate_chunks([chunk1, chunk2, chunk3])
    assert len(result) == 2
    assert result[0].chunk_text == text_a
    assert result[1].chunk_text == text_b


def test_build_windows_single() -> None:
    """For short text within budget, _build_windows returns a single window."""
    text = "The quick brown fox jumps over the lazy dog. " * 5
    result = _build_windows(text, budget=60000)
    assert len(result) == 1
    assert isinstance(result[0], tuple)
    assert result[0][1] == 0  # char offset should be 0


def test_build_windows_multiple() -> None:
    """For text exceeding budget, _build_windows returns multiple overlapping windows."""
    # Repeat paragraphs enough to exceed a tiny artificial budget
    paragraph = (
        "This is a paragraph about a topic. It contains multiple sentences "
        "that discuss various aspects of the topic in detail. "
    )
    text = paragraph * 10  # ~160+ tokens
    result = _build_windows(text, budget=50)
    assert len(result) >= 2
    # Each window is a (str, int) tuple
    for window_text, offset in result:
        assert isinstance(window_text, str)
        assert isinstance(offset, int)
