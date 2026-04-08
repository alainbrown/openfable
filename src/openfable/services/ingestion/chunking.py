"""ChunkingService: LLM-based semantic chunking with sentence-aligned windowing.

Design notes:
- D-01: Prompt contains NO token/size constraints — boundaries are purely discourse-driven.
- D-03: Long documents are split into sentence-aligned windows with 12% overlap.
- D-04: Duplicate chunks at window boundaries are deduplicated by exact text match.
- D-05: Character offsets (start_idx, end_idx) are tracked and repaired after LLM output.
"""

import logging
import re

from instructor.core import InstructorRetryException

from openfable.exceptions import ChunkingError
from openfable.repositories.document_repo import count_tokens
from openfable.schemas.chunking import ChunkingResponse, ChunkResult
from openfable.services.llm_service import LLMService

logger = logging.getLogger(__name__)

# Max tokens of document text per LLM window (D-03: must fit in LLM context)
CHUNKING_CONTEXT_BUDGET = 60_000

# Overlap fraction between adjacent windows — 12% is within D-03's 10-15% range
OVERLAP_FRACTION = 0.12

# D-01: NO mention of token counts, word limits, or size constraints in the prompt.
# Boundaries are purely discourse-driven.
SYSTEM_PROMPT = """You are a document segmentation specialist.
Split the document into semantically coherent chunks. Each chunk should:
- Cover exactly one topic, concept, or coherent idea
- Begin and end at sentence boundaries
- NOT be split arbitrarily -- split only when the topic or concept changes

Return each chunk as an object with:
- chunk_text: the exact chunk content (verbatim from the document, no edits)
- start_idx: the character offset where this chunk starts in the input text
- end_idx: the character offset where this chunk ends (exclusive) in the input text

The chunks must be non-overlapping, cover the full input text, and appear in document order."""

USER_TEMPLATE = "Segment the following document text into semantically coherent chunks:\n\n{text}"


def _split_sentences(text: str) -> list[str]:
    """Split text at sentence boundaries using punctuation heuristics.

    Uses regex to split at sentence-ending punctuation followed by whitespace
    and a capital letter or quote. No NLTK/spaCy dependency required.
    """
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", text)
    return [p for p in parts if p]


def _build_windows(
    text: str,
    budget: int = CHUNKING_CONTEXT_BUDGET,
    overlap_fraction: float = OVERLAP_FRACTION,
) -> list[tuple[str, int]]:
    """Build sentence-aligned windows with token overlap for long documents (D-03).

    Returns a list of (window_text, char_offset_in_document) tuples.
    For documents within budget, returns [(text, 0)].
    """
    total_tokens = count_tokens(text)
    if total_tokens <= budget:
        return [(text, 0)]

    sentences = _split_sentences(text)
    overlap_tokens = int(budget * overlap_fraction)

    windows: list[tuple[str, int]] = []
    current_sentences: list[str] = []
    current_tokens = 0
    search_start = 0  # forward-advancing cursor for offset tracking

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence)

        if current_tokens + sentence_tokens > budget and current_sentences:
            # Finalize the current window
            window_text = " ".join(current_sentences)
            window_start = text.find(window_text[:100], search_start)
            if window_start == -1:
                window_start = search_start
            windows.append((window_text, window_start))

            # Advance search cursor past start of finalized window
            search_start = window_start + 1

            # Roll back to keep ~overlap_tokens worth of trailing sentences
            rollback: list[str] = []
            rollback_tokens = 0
            for s in reversed(current_sentences):
                t = count_tokens(s)
                if rollback_tokens + t > overlap_tokens:
                    break
                rollback.insert(0, s)
                rollback_tokens += t

            current_sentences = rollback + [sentence]
            current_tokens = rollback_tokens + sentence_tokens
        else:
            current_sentences.append(sentence)
            current_tokens += sentence_tokens

    # Flush the final window
    if current_sentences:
        window_text = " ".join(current_sentences)
        window_start = text.find(window_text[:100], search_start)
        if window_start == -1:
            window_start = search_start
        windows.append((window_text, window_start))

    return windows


def _deduplicate_chunks(chunks: list[ChunkResult]) -> list[ChunkResult]:
    """Remove duplicate chunks by exact text match, keeping the first occurrence (D-04).

    Duplicates arise at window boundaries where trailing sentences from one window
    become the leading sentences of the next. Only the first occurrence is kept.
    """
    seen: set[str] = set()
    result: list[ChunkResult] = []
    for chunk in chunks:
        key = chunk.chunk_text.strip()
        if key not in seen:
            seen.add(key)
            result.append(chunk)
    return result


class ChunkingService:
    """LLM-based semantic chunking service implementing FABLE D-01 through D-05."""

    def __init__(self, llm: LLMService) -> None:
        self.llm = llm

    def _chunk_window(self, text: str) -> list[ChunkResult]:
        """Call the LLM to chunk a single text window.

        Raises ChunkingError if instructor exhausts retries or validation fails.
        """
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=text)},
        ]
        try:
            response: ChunkingResponse = self.llm.complete_structured(
                response_model=ChunkingResponse,
                messages=messages,
                max_retries=3,
                temperature=0,
            )
            return response.chunks
        except InstructorRetryException as exc:
            raise ChunkingError(f"LLM chunking failed after retries: {exc}") from exc
        except Exception as exc:
            # Catch any other instructor/LiteLLM failures and surface as ChunkingError
            raise ChunkingError(f"LLM chunking failed: {exc}") from exc

    def _repair_offsets(
        self,
        text: str,
        chunks: list[ChunkResult],
        window_offset: int,
    ) -> list[ChunkResult]:
        """Remap chunk offsets from window-relative to document-relative coordinates.

        For each chunk, adds window_offset to start_idx and end_idx. Then verifies
        the text slice matches. If it does not, searches forward from the previous
        chunk's end to locate the actual position and repairs the offsets (Pitfall 2).
        """
        repaired: list[ChunkResult] = []
        prev_end = window_offset

        for chunk in chunks:
            # Remap to document-absolute offsets
            abs_start = chunk.start_idx + window_offset
            abs_end = chunk.end_idx + window_offset

            # Verify the slice matches
            if text[abs_start:abs_end].strip() == chunk.chunk_text.strip():
                repaired.append(
                    ChunkResult(
                        chunk_text=chunk.chunk_text,
                        start_idx=abs_start,
                        end_idx=abs_end,
                    )
                )
                prev_end = abs_end
            else:
                # Offset mismatch: search forward from previous end to locate the text
                found = text.find(chunk.chunk_text.strip(), prev_end)
                if found != -1:
                    new_start = found
                    new_end = found + len(chunk.chunk_text)
                    logger.warning(
                        "Repaired chunk offset mismatch: expected [%d:%d], found at [%d:%d]",
                        abs_start,
                        abs_end,
                        new_start,
                        new_end,
                    )
                    repaired.append(
                        ChunkResult(
                            chunk_text=chunk.chunk_text,
                            start_idx=new_start,
                            end_idx=new_end,
                        )
                    )
                    prev_end = new_end
                else:
                    # Cannot locate the chunk in the document; keep with remapped offsets
                    # and log a warning rather than failing ingest (Pitfall 2).
                    logger.warning(
                        "Could not locate chunk text in document for offset repair; "
                        "keeping remapped offsets [%d:%d]. Chunk preview: %.80r",
                        abs_start,
                        abs_end,
                        chunk.chunk_text,
                    )
                    repaired.append(
                        ChunkResult(
                            chunk_text=chunk.chunk_text,
                            start_idx=abs_start,
                            end_idx=abs_end,
                        )
                    )
                    prev_end = abs_end

        return repaired

    def segment(self, text: str) -> list[ChunkResult]:
        """Segment a document into semantically coherent chunks.

        Handles single-window and multi-window documents. Long documents are split
        into sentence-aligned windows with 12% overlap (D-03). Chunks from all
        windows are merged, deduplicated (D-04), and sorted by start_idx.

        Raises:
            ChunkingError: If any LLM window call fails after retries.
        """
        windows = _build_windows(text)
        all_chunks: list[ChunkResult] = []

        for window_text, window_offset in windows:
            window_chunks = self._chunk_window(window_text)
            repaired = self._repair_offsets(text, window_chunks, window_offset)
            all_chunks.extend(repaired)

        deduped = _deduplicate_chunks(all_chunks)
        return sorted(deduped, key=lambda c: c.start_idx)


def get_chunking_service(llm: LLMService | None = None) -> ChunkingService:
    """Factory function for ChunkingService with optional LLM override."""
    from openfable.services.llm_service import get_llm_service

    return ChunkingService(llm or get_llm_service())
