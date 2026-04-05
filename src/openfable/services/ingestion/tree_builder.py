"""TreeBuilder: LLM-based hierarchical tree construction from document chunks.

Design notes:
- D-01/D-02: Single-pass TreeBuild when all chunks fit in 70% of the model's context
  window; progressive TreeBuild + TreeMerge for long documents.
- D-03: Partitions for progressive build are sequential, non-overlapping, bounded by
  content_budget tokens. Each partition runs TreeBuild independently.
- D-04: TreeMerge creates a unified root from partial tree summaries without modifying
  partition structure.
- D-06: ltree paths are computed in code from ancestor titles via _sanitize_ltree_label
  and _build_toc_path -- the LLM never generates paths.
- D-07: Post-construction depth enforcement; nodes at depth > 4 are re-parented to
  their grandparent and a warning is logged.
- D-08: Leaf nodes receive chunk_id so NodeRepository.link_chunks_to_leaves() can
  update Chunk.node_id after tree persistence.
- D-09: All complete_structured calls use temperature=0 for reproducibility.
"""

import logging
import re
import uuid
from collections import deque
from typing import TYPE_CHECKING

import litellm
from instructor.core import InstructorRetryException

from openfable.exceptions import TreeConstructionError
from openfable.repositories.node_repo import NodeInsert
from openfable.schemas.tree import (
    LLMInternalNode,
    LLMLeafNode,
    PartialTreeSummary,
    TreeBuildResponse,
    TreeMergeResponse,
)

if TYPE_CHECKING:
    from openfable.models.chunk import Chunk
    from openfable.services.llm_service import LLMService

logger = logging.getLogger(__name__)

_DEFAULT_CONTEXT_WINDOW = 100_000  # safe fallback for unknown models
_CONTENT_BUDGET_FRACTION = 0.70  # D-02: 70% of context window for chunk content
_MAX_DEPTH = 4  # D-07: maximum tree depth

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

TREE_BUILD_SYSTEM_PROMPT = """\
Organize numbered chunks into a JSON tree. Every chunk must appear exactly once as a leaf.

Output format: a single JSON object with one key "root" containing the tree.
- Internal nodes have: "type": "internal", "node_type" (root/section/subsection), "title", "summary", "children" (array)
- Leaf nodes have: "type": "leaf", "chunk_index" (0-based integer)
- Maximum depth: 4 levels (root > section > subsection > leaf)
- All chunks must be included. Do not skip any chunk_index.

Example — given chunks 0 (short intro), 1 (topic A), 2 (topic B), output:

{"root": {"type": "internal", "node_type": "root", "title": "Document Title", "summary": "Brief summary of entire document", "children": [{"type": "leaf", "chunk_index": 0}, {"type": "internal", "node_type": "section", "title": "Topic A", "summary": "Summary of topic A", "children": [{"type": "leaf", "chunk_index": 1}]}, {"type": "internal", "node_type": "section", "title": "Topic B", "summary": "Summary of topic B", "children": [{"type": "leaf", "chunk_index": 2}]}]}}

Note: even short intro chunks must be leaf nodes. Do not absorb them into summaries.
"""

TREE_BUILD_PARTITION_SYSTEM_PROMPT = """\
Organize numbered chunks into a JSON tree. Every chunk must appear exactly once as a leaf.

Output format: a single JSON object with one key "root" containing the tree.
- Internal nodes have: "type": "internal", "node_type" (root/section), "title", "summary", "children" (array)
- Leaf nodes have: "type": "leaf", "chunk_index" (0-based integer)
- Maximum depth: 3 levels (root > section > leaf). Do not use subsection.
- All chunks must be included. Do not skip any chunk_index.

Example — given chunks 0 (short intro), 1 (topic A), 2 (topic B), output:

{"root": {"type": "internal", "node_type": "root", "title": "Partition Title", "summary": "Brief summary", "children": [{"type": "leaf", "chunk_index": 0}, {"type": "internal", "node_type": "section", "title": "Topic A", "summary": "Summary of topic A", "children": [{"type": "leaf", "chunk_index": 1}]}, {"type": "internal", "node_type": "section", "title": "Topic B", "summary": "Summary of topic B", "children": [{"type": "leaf", "chunk_index": 2}]}]}}

Note: even short intro chunks must be leaf nodes. Do not absorb them into summaries.
"""

TREE_MERGE_SYSTEM_PROMPT = """\
Create a unified root for the partial tree summaries below.

Output format: a JSON object with "merged_title" (string) and "merged_summary" (string).
- The title should describe the whole document's scope.
- The summary should cover key themes from all parts.

Example:

{"merged_title": "Complete Document", "merged_summary": "Covers topics A, B, and C across all parts."}
"""

# ---------------------------------------------------------------------------
# Module-level helper functions (directly testable without instantiating TreeBuilder)
# ---------------------------------------------------------------------------


def _get_content_budget(model: str) -> int:
    """Return the token budget for chunk content (D-01/D-02).

    Calls litellm.get_max_tokens(model) to determine context window size.
    Falls back to _DEFAULT_CONTEXT_WINDOW if the model is not known to LiteLLM.
    Returns 70% of the context window as the content budget.
    """
    try:
        window = litellm.get_max_tokens(model)
        if window is None:
            logger.warning(
                "litellm.get_max_tokens(%r) returned None; using default context window %d",
                model,
                _DEFAULT_CONTEXT_WINDOW,
            )
            window = _DEFAULT_CONTEXT_WINDOW
    except Exception:
        logger.warning(
            "litellm.get_max_tokens(%r) raised an exception; using default context window %d",
            model,
            _DEFAULT_CONTEXT_WINDOW,
        )
        window = _DEFAULT_CONTEXT_WINDOW
    return int(window * _CONTENT_BUDGET_FRACTION)


def _sanitize_ltree_label(title: str, max_length: int = 63) -> str:
    """Convert a title string to a valid ltree label component (D-06).

    Replaces any character not in [A-Za-z0-9_] with '_' (this also replaces dots).
    Collapses multiple consecutive underscores to a single underscore.
    Strips leading and trailing underscores.
    Truncates to max_length characters.
    Returns 'node' if the result is empty.
    """
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", title)
    sanitized = re.sub(r"_+", "_", sanitized)
    sanitized = sanitized.strip("_")
    sanitized = sanitized[:max_length]
    return sanitized if sanitized else "node"


def _build_toc_path(ancestor_titles: list[str], current_title: str) -> str:
    """Build a dot-separated ltree path from ancestor titles and current title (D-06).

    Each component is sanitized via _sanitize_ltree_label before joining.
    """
    parts = [_sanitize_ltree_label(t) for t in ancestor_titles]
    parts.append(_sanitize_ltree_label(current_title))
    return ".".join(parts)


def _flatten_excess_depth(
    nodes: list[NodeInsert], max_depth: int = _MAX_DEPTH
) -> list[NodeInsert]:
    """Re-parent children of nodes that exceed max_depth, then recompute depths (D-07).

    Nodes at depth > max_depth are removed from the list and their children are
    re-parented to the violating node's parent. Depth values and paths are then
    recomputed by walking the tree from the root downward.

    Logs a warning for each flattened node.
    """
    # Identify violations: nodes deeper than max_depth, sorted deepest-first
    violations = sorted(
        [n for n in nodes if n.depth > max_depth],
        key=lambda n: n.depth,
        reverse=True,
    )

    nodes_set = list(nodes)  # working copy

    for violating in violations:
        if violating not in nodes_set:
            continue  # already removed as part of a deeper chain
        logger.warning(
            "Flattened node at depth %d (title=%r) to parent at depth %d",
            violating.depth,
            violating.title,
            violating.depth - 1,
        )
        # Re-parent children of violating node to violating node's parent
        for child in nodes_set:
            if child.parent_id == violating.id:
                child.parent_id = violating.parent_id
        # Remove the violating node
        nodes_set.remove(violating)

    # Recompute depth and path by walking from roots downward (BFS)
    queue: deque[tuple[NodeInsert, int, list[str]]] = deque()
    for n in nodes_set:
        if n.parent_id is None:
            queue.append((n, 1, []))

    while queue:
        node, depth, ancestor_titles = queue.popleft()
        node.depth = depth
        if node.node_type == "leaf":
            # Leaf path uses chunk index label; preserve the last component
            leaf_label = _sanitize_ltree_label(
                node.path.split(".")[-1] if node.path else "chunk"
            )
            node.path = _build_toc_path(ancestor_titles, leaf_label)
            # toc_path stays None for leaves
        else:
            node.path = _build_toc_path(ancestor_titles, node.title or "node")
            node.toc_path = node.path
        # Enqueue children
        if node.node_type != "leaf":
            child_ancestors = ancestor_titles + [node.title or "node"]
        else:
            child_ancestors = ancestor_titles
        for child in nodes_set:
            if child.parent_id == node.id:
                queue.append((child, depth + 1, child_ancestors))

    return nodes_set


def _validate_chunk_coverage(root: LLMInternalNode, num_chunks: int) -> None:
    """Validate that every chunk index appears exactly once in the tree (Pitfall 4/5).

    Raises TreeConstructionError if:
    - Any chunk_index is out of bounds (>= num_chunks)
    - Any chunk_index is missing (some chunks not represented as leaves)
    - Any chunk_index appears more than once (duplicate leaves)
    """
    all_indexes: list[int] = []

    def _collect(node: LLMInternalNode | LLMLeafNode) -> None:
        if isinstance(node, LLMLeafNode):
            all_indexes.append(node.chunk_index)
        else:
            for child in node.children:
                _collect(child)

    _collect(root)

    # Check out-of-bounds
    for idx in all_indexes:
        if idx >= num_chunks or idx < 0:
            raise TreeConstructionError(
                f"LLM returned chunk_index={idx} but only {num_chunks} chunks were provided."
            )

    # Check missing coverage
    expected = set(range(num_chunks))
    found = set(all_indexes)
    missing = expected - found
    if missing:
        raise TreeConstructionError(
            f"Missing chunk indexes: {sorted(missing)}. "
            "Every chunk must appear exactly once as a leaf node."
        )

    # Check duplicates
    if len(all_indexes) != len(set(all_indexes)):
        raise TreeConstructionError(
            "Duplicate chunk_index values found in tree. Each chunk must appear exactly once."
        )


def _llm_tree_to_node_inserts(
    root: LLMInternalNode, chunks: list["Chunk"]
) -> list[NodeInsert]:
    """Recursively convert an LLMInternalNode tree into a flat list of NodeInsert objects.

    Traverses in DFS order (parent before children). Computes toc_path and path from
    ancestor titles via _build_toc_path. Leaf nodes receive chunk content and chunk_id
    for subsequent DB linkage (D-06, D-08).
    """
    result: list[NodeInsert] = []

    def _traverse(
        node: LLMInternalNode | LLMLeafNode,
        parent_id: uuid.UUID | None,
        depth: int,
        position: int,
        ancestor_titles: list[str],
    ) -> None:
        if isinstance(node, LLMLeafNode):
            chunk = chunks[node.chunk_index]
            leaf_path = _build_toc_path(ancestor_titles, f"chunk_{node.chunk_index}")
            ni = NodeInsert(
                node_type="leaf",
                depth=depth,
                position=position,
                title=None,
                summary=None,
                toc_path=None,
                content=chunk.content,
                token_count=chunk.token_count,
                parent_id=parent_id,
                path=leaf_path,
                chunk_id=chunk.id,
            )
            result.append(ni)
        else:
            toc = _build_toc_path(ancestor_titles, node.title)
            ni = NodeInsert(
                node_type=node.node_type,
                depth=depth,
                position=position,
                title=node.title,
                summary=node.summary,
                toc_path=toc,
                content=None,
                token_count=None,
                parent_id=parent_id,
                path=toc,
            )
            result.append(ni)
            child_ancestors = ancestor_titles + [node.title]
            for i, child in enumerate(node.children):
                _traverse(child, ni.id, depth + 1, i, child_ancestors)

    _traverse(root, None, 1, 0, [])
    return result


def _recompute_paths(root_node: NodeInsert, all_nodes: list[NodeInsert]) -> list[NodeInsert]:
    """Recompute toc_path and path for all nodes from the root downward.

    Walks the tree in BFS order starting from root_node, rebuilding paths
    based on updated titles and hierarchy positions.
    """
    # Build children mapping
    children_map: dict[uuid.UUID, list[NodeInsert]] = {}
    for n in all_nodes:
        if n.parent_id is not None:
            children_map.setdefault(n.parent_id, []).append(n)
    for pid in children_map:
        children_map[pid].sort(key=lambda n: n.position)

    queue: deque[tuple[NodeInsert, list[str]]] = deque()
    queue.append((root_node, []))

    while queue:
        node, ancestor_titles = queue.popleft()
        if node.node_type == "leaf":
            # Leaf: reconstruct path from ancestors + chunk label
            leaf_label = node.path.split(".")[-1] if node.path else "chunk"
            node.path = _build_toc_path(ancestor_titles, leaf_label)
            # toc_path remains None for leaves
        else:
            node.path = _build_toc_path(ancestor_titles, node.title or "node")
            node.toc_path = node.path

        if node.node_type != "leaf":
            next_ancestors = ancestor_titles + [node.title or "node"]
        else:
            next_ancestors = ancestor_titles
        for child in children_map.get(node.id, []):
            queue.append((child, next_ancestors))

    return all_nodes


# ---------------------------------------------------------------------------
# TreeBuilder class
# ---------------------------------------------------------------------------


class TreeBuilder:
    """Builds a hierarchical tree from document chunks using LLM-based structuring.

    Dispatches to single-pass or progressive build based on content budget (D-01, D-03).
    """

    def __init__(self, llm: "LLMService") -> None:
        self.llm = llm

    def build(self, chunks: list["Chunk"]) -> list[NodeInsert]:
        """Build a hierarchical tree from chunks. Returns flat list of NodeInsert objects.

        Uses single-pass TreeBuild if all chunks fit within the content budget (D-01).
        Uses progressive TreeBuild + TreeMerge for long documents (D-03, D-04).
        Post-construction: validates depth constraint and flattens violations (D-07).
        """
        if not chunks:
            raise TreeConstructionError("Cannot build tree from empty chunk list")

        content_budget = _get_content_budget(self.llm.model)
        total_tokens = sum(c.token_count for c in chunks)

        if total_tokens <= content_budget:
            nodes = self._single_pass_build(chunks)
        else:
            nodes = self._progressive_build(chunks, content_budget)

        # D-07: flatten any nodes exceeding max depth
        nodes = _flatten_excess_depth(nodes)
        return nodes

    def _single_pass_build(self, chunks: list["Chunk"]) -> list[NodeInsert]:
        """Build the tree in a single LLM call (D-01: fits within content budget)."""
        chunk_text = "\n\n".join(
            f"Chunk {i}:\n{chunk.content}" for i, chunk in enumerate(chunks)
        )
        user_msg = f"There are {len(chunks)} chunks (indexes 0 to {len(chunks) - 1}). Every index must appear exactly once as a leaf.\n\n{chunk_text}"
        messages = [
            {"role": "system", "content": TREE_BUILD_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        try:
            response: TreeBuildResponse = self.llm.complete_structured(  # type: ignore[assignment]
                response_model=TreeBuildResponse,
                messages=messages,
                max_retries=3,
                temperature=0,
            )
        except InstructorRetryException as exc:
            raise TreeConstructionError(
                f"LLM tree construction failed after retries: {exc}"
            ) from exc
        except Exception as exc:
            raise TreeConstructionError(f"LLM tree construction failed: {exc}") from exc

        _validate_chunk_coverage(response.root, len(chunks))
        return _llm_tree_to_node_inserts(response.root, chunks)

    def _progressive_build(
        self, chunks: list["Chunk"], content_budget: int
    ) -> list[NodeInsert]:
        """Build tree progressively: partition chunks, build each, then merge (D-03, D-04)."""
        # Partition chunks into sequential groups bounded by content_budget
        partitions: list[list["Chunk"]] = []
        current_partition: list["Chunk"] = []
        current_tokens = 0

        for chunk in chunks:
            if current_tokens + chunk.token_count > content_budget and current_partition:
                partitions.append(current_partition)
                current_partition = [chunk]
                current_tokens = chunk.token_count
            else:
                current_partition.append(chunk)
                current_tokens += chunk.token_count

        if current_partition:
            partitions.append(current_partition)

        # Build each partition
        partial_roots: list[LLMInternalNode] = []
        all_partition_nodes: list[list[NodeInsert]] = []
        for partition_chunks in partitions:
            partial_root, partition_nodes = self._build_partition(partition_chunks)
            partial_roots.append(partial_root)
            all_partition_nodes.append(partition_nodes)

        # Collect PartialTreeSummary objects for the merge call
        summaries = [
            PartialTreeSummary(
                part_index=i,
                root_title=partial_roots[i].title,
                root_summary=partial_roots[i].summary,
            )
            for i in range(len(partitions))
        ]

        return self._merge_trees(summaries, all_partition_nodes)

    def _build_partition(
        self, chunks: list["Chunk"]
    ) -> tuple[LLMInternalNode, list[NodeInsert]]:
        """Build a depth-constrained tree (max 3 levels) for a single partition (D-03)."""
        chunk_text = "\n\n".join(
            f"Chunk {i}:\n{chunk.content}" for i, chunk in enumerate(chunks)
        )
        user_msg = f"There are {len(chunks)} chunks (indexes 0 to {len(chunks) - 1}). Every index must appear exactly once as a leaf.\n\n{chunk_text}"
        messages = [
            {"role": "system", "content": TREE_BUILD_PARTITION_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        try:
            response: TreeBuildResponse = self.llm.complete_structured(  # type: ignore[assignment]
                response_model=TreeBuildResponse,
                messages=messages,
                max_retries=3,
                temperature=0,
            )
        except InstructorRetryException as exc:
            raise TreeConstructionError(
                f"LLM partition tree construction failed after retries: {exc}"
            ) from exc
        except Exception as exc:
            raise TreeConstructionError(
                f"LLM partition tree construction failed: {exc}"
            ) from exc

        _validate_chunk_coverage(response.root, len(chunks))
        node_inserts = _llm_tree_to_node_inserts(response.root, chunks)
        return response.root, node_inserts

    def _merge_trees(
        self,
        summaries: list[PartialTreeSummary],
        partition_nodes: list[list[NodeInsert]],
    ) -> list[NodeInsert]:
        """Merge partition trees under a unified root (D-04).

        Creates a new merged root from LLM-generated title/summary.
        Partition roots become section children of the merged root.
        All descendant depths are incremented by 1.
        toc_path and path are recomputed from the merged root downward.
        """
        user_msg = "\n\n".join(
            f"Part {s.part_index}: Title: {s.root_title}, Summary: {s.root_summary}"
            for s in summaries
        )
        messages = [
            {"role": "system", "content": TREE_MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        try:
            response: TreeMergeResponse = self.llm.complete_structured(  # type: ignore[assignment]
                response_model=TreeMergeResponse,
                messages=messages,
                max_retries=3,
                temperature=0,
            )
        except InstructorRetryException as exc:
            raise TreeConstructionError(
                f"LLM tree merge failed after retries: {exc}"
            ) from exc
        except Exception as exc:
            raise TreeConstructionError(f"LLM tree merge failed: {exc}") from exc

        # Build merged root
        merged_root = NodeInsert(
            node_type="root",
            depth=1,
            position=0,
            title=response.merged_title,
            summary=response.merged_summary,
            toc_path=_sanitize_ltree_label(response.merged_title),
            content=None,
            token_count=None,
            parent_id=None,
            path=_sanitize_ltree_label(response.merged_title),
        )

        all_nodes: list[NodeInsert] = [merged_root]

        for part_idx, nodes_in_partition in enumerate(partition_nodes):
            if not nodes_in_partition:
                continue

            # The first node in each partition's list is the partition root
            partition_root = nodes_in_partition[0]
            partition_root.parent_id = merged_root.id
            partition_root.node_type = "section"
            partition_root.depth = 2
            partition_root.position = part_idx

            # Depth delta: partition root was depth=1, now section=depth=2
            depth_delta = 1  # increment all non-root partition nodes by 1

            # Update all non-root nodes in partition: increment depth by delta
            for node in nodes_in_partition[1:]:
                node.depth += depth_delta

            all_nodes.extend(nodes_in_partition)

        # Recompute all toc_path and path values from merged root downward
        all_nodes = _recompute_paths(merged_root, all_nodes)

        return all_nodes


def get_tree_builder(llm: "LLMService") -> "TreeBuilder":
    """Factory function for TreeBuilder."""
    return TreeBuilder(llm)
