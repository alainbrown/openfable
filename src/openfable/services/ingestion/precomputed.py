"""Chunker / TreeStructurer implementations driven by externally-supplied structure.

These satisfy the same protocols as ChunkingService and TreeBuilder but make no
LLM calls. Structure arrives from outside the process -- in practice, from an
agent that read the document and produced chunk boundaries and a tree.

Chunk boundaries are supplied as *markers* (the opening words of each chunk)
rather than character offsets. Offsets are computed here, in code. Models are
unreliable at reporting character positions, which is why the LLM path needs
ChunkingService._repair_offsets; asking for markers avoids the problem rather
than correcting for it afterwards.
"""

import re

from openfable.exceptions import ChunkingError, TreeConstructionError
from openfable.models.chunk import Chunk as ChunkModel
from openfable.repositories.node_repo import NodeInsert
from openfable.schemas.chunking import ChunkResult
from openfable.schemas.tree import LLMInternalNode
from openfable.services.ingestion.tree_builder import (
    _flatten_excess_depth,
    _llm_tree_to_node_inserts,
    _validate_chunk_coverage,
)


def _find_marker(text: str, marker: str, start: int) -> int:
    """Locate a boundary marker at or after `start`. Returns -1 if not found.

    Falls back to a whitespace-tolerant match so a marker differing from the
    source only in line breaks or repeated spaces still resolves.
    """
    exact = text.find(marker, start)
    if exact != -1:
        return exact

    tokens = marker.split()
    if not tokens:
        return -1
    pattern = r"\s+".join(re.escape(tok) for tok in tokens)
    match = re.compile(pattern).search(text, start)
    return match.start() if match else -1


def chunks_from_boundaries(text: str, markers: list[str]) -> list[ChunkResult]:
    """Convert opening-text markers into exact, gap-free ChunkResults.

    Each marker is the verbatim opening of a chunk. Chunk i runs from its own
    marker to the start of marker i+1; the final chunk runs to end of text.
    Markers must appear in document order.

    Raises:
        ChunkingError: if the list is empty, or a marker cannot be located at or
            after the previous one. The message names the offending marker so
            the caller can correct it and retry.
    """
    if not markers:
        raise ChunkingError("No chunk boundaries supplied; expected at least one marker.")

    positions: list[int] = []
    search_from = 0
    for i, marker in enumerate(markers):
        cleaned = marker.strip()
        if not cleaned:
            raise ChunkingError(f"Boundary marker {i} is empty.")

        pos = _find_marker(text, cleaned, search_from)
        if pos == -1:
            raise ChunkingError(
                f"Boundary marker {i} not found at or after offset {search_from}: "
                f"{cleaned[:80]!r}. Markers must be verbatim openings of each chunk, "
                "in document order."
            )
        positions.append(pos)
        # Advance past the marker so a repeated phrase resolves to its next
        # occurrence rather than matching the same spot twice.
        search_from = pos + len(cleaned)

    if positions[0] != 0:
        # Text before the first marker would otherwise be silently dropped.
        positions.insert(0, 0)

    bounds = positions + [len(text)]
    chunks: list[ChunkResult] = []
    for i in range(len(bounds) - 1):
        start, end = bounds[i], bounds[i + 1]
        body = text[start:end]
        if not body.strip():
            continue
        chunks.append(ChunkResult(chunk_text=body, start_idx=start, end_idx=end))

    if not chunks:
        raise ChunkingError("Boundary markers produced no non-empty chunks.")
    return chunks


class PrecomputedChunker:
    """Chunker returning boundaries resolved from supplied markers."""

    def __init__(self, markers: list[str]) -> None:
        self.markers = markers

    def segment(self, text: str) -> list[ChunkResult]:
        return chunks_from_boundaries(text, self.markers)


class PrecomputedTreeStructurer:
    """TreeStructurer converting a supplied tree into NodeInserts.

    Runs the same validation and path construction as the LLM path: chunk
    coverage is checked (every chunk exactly once), toc paths are computed in
    code from ancestor titles, and depth violations are flattened.
    """

    def __init__(self, root: LLMInternalNode) -> None:
        self.root = root

    def build(self, chunks: list[ChunkModel]) -> list[NodeInsert]:
        if not chunks:
            raise TreeConstructionError("Cannot build tree from empty chunk list")
        _validate_chunk_coverage(self.root, len(chunks))
        nodes = _llm_tree_to_node_inserts(self.root, chunks)
        return _flatten_excess_depth(nodes)
