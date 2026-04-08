import logging
import uuid
from collections import defaultdict
from typing import Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from openfable.config import settings
from openfable.exceptions import EmbeddingError, RetrievalError
from openfable.models.document import Document
from openfable.models.node import Node
from openfable.repositories.document_repo import DocumentRepository
from openfable.repositories.node_repo import NodeRepository
from openfable.schemas.retrieval import (
    ChunkResult,
    DocumentResult,
    LLMNavigateResult,
    LLMSelectResult,
    NodeResult,
    QueryResponse,
)
from openfable.services.embedding_service import EmbeddingService
from openfable.services.llm_service import LLMService

logger = logging.getLogger(__name__)


def _build_children_map(nodes: list[Node]) -> dict[uuid.UUID, list[Node]]:
    """Build parent_id -> [children] mapping for tree walking."""
    children: dict[uuid.UUID, list[Node]] = defaultdict(list)
    for node in nodes:
        if node.parent_id is not None:
            children[node.parent_id].append(node)
    return dict(children)


def _expand_llmnav_to_leaves(
    llm_nav_scores: dict[uuid.UUID, float],
    all_nodes: list[Node],
) -> dict[uuid.UUID, float]:
    """Propagate LLMnavigate internal-node scores to descendant leaf nodes.

    LLMnavigate selects internal (non-leaf) subtree roots with relevance scores.
    TreeExpansion produces leaf-level scores. To merge them, we expand each
    internal node's score to all its descendant leaves.

    For each selected internal node, all descendant leaves receive its score.
    If a leaf is a descendant of multiple selected nodes, it gets the max score.
    """
    if not llm_nav_scores:
        return {}

    children_map = _build_children_map(all_nodes)
    leaf_scores: dict[uuid.UUID, float] = {}

    for node_id, score in llm_nav_scores.items():
        # BFS to find all descendant leaves of this internal node
        queue = [node_id]
        while queue:
            current = queue.pop(0)
            children = children_map.get(current, [])
            for child in children:
                if child.node_type == "leaf":
                    # Max-score: if leaf descended from multiple selected nodes
                    leaf_scores[child.id] = max(leaf_scores.get(child.id, 0.0), score)
                else:
                    queue.append(child.id)

    return leaf_scores


def _compute_tree_expansion_scores(
    nodes: list[Node],
    leaf_similarities: dict[uuid.UUID, float],
    internal_similarities: dict[uuid.UUID, float],
) -> dict[uuid.UUID, float]:
    """TreeExpansion scoring: S(v) = 1/3[S_sim + S_inh + S_child] (D-05 through D-10).

    Args:
        nodes: All nodes (all types) for the documents being scored.
        leaf_similarities: node_id -> cosine similarity for leaf nodes.
        internal_similarities: node_id -> cosine similarity for internal nodes.

    Returns:
        dict mapping leaf node_id -> normalized score [0, 1].
    """
    if not nodes:
        return {}

    # Step 1: S_sim for all nodes (D-06)
    # S_sim(v) = cosine_similarity / max(depth, 1) -- linear depth decay
    all_similarities = {**internal_similarities, **leaf_similarities}
    s_sim: dict[uuid.UUID, float] = {
        node.id: all_similarities.get(node.id, 0.0) / max(node.depth, 1) for node in nodes
    }

    # Step 2: S_inh top-down, depth ascending (D-07)
    # Root has S_inh = 0; child inherits max(parent's S_inh, parent's S_sim)
    node_map = {n.id: n for n in nodes}
    s_inh: dict[uuid.UUID, float] = {}
    for node in sorted(nodes, key=lambda n: n.depth):
        if node.parent_id is None or node.parent_id not in node_map:
            s_inh[node.id] = 0.0
        else:
            s_inh[node.id] = max(
                s_inh.get(node.parent_id, 0.0),
                s_sim.get(node.parent_id, 0.0),
            )

    # Step 3: S_child + S(v) bottom-up, depth descending (D-08)
    # Leaves: S_child = 0; internal: S_child = mean of children's full S(v)
    children_map = _build_children_map(nodes)
    s_v: dict[uuid.UUID, float] = {}
    for node in sorted(nodes, key=lambda n: n.depth, reverse=True):
        children = children_map.get(node.id, [])
        if children:
            s_child = sum(s_v[c.id] for c in children) / len(children)
        else:
            s_child = 0.0
        s_v[node.id] = (s_sim.get(node.id, 0.0) + s_inh.get(node.id, 0.0) + s_child) / 3.0

    # Step 4: Min-max normalization across leaves only (D-10)
    leaf_ids = {n.id for n in nodes if n.node_type == "leaf"}
    leaf_s_v = {nid: s_v[nid] for nid in leaf_ids if nid in s_v}
    if not leaf_s_v:
        return {}
    min_s = min(leaf_s_v.values())
    max_s = max(leaf_s_v.values())
    if (max_s - min_s) < 1e-9:
        return {nid: 1.0 for nid in leaf_s_v}
    return {nid: (s - min_s) / (max_s - min_s) for nid, s in leaf_s_v.items()}


LLMSELECT_SYSTEM = (
    "You are a document relevance assessor for a "
    "retrieval system.\n\n"
    "You will receive a user query and a set of document "
    "abstractions — each document is represented by its "
    "section headings (table-of-contents paths) and "
    "summaries from the top levels of its semantic tree."
    "\n\n"
    "Your task is to reason about which documents are "
    "likely to contain information relevant to the query "
    "based on these high-level abstractions. Consider:\n"
    "- Whether the document's topic areas overlap with "
    "the query's intent\n"
    "- Whether the section structure suggests the document "
    "covers the query's subject matter\n"
    "- Partial relevance counts — a document that covers "
    "part of the query is still relevant\n\n"
    "Assign a relevance score (0.0 to 1.0) to each "
    "relevant document. Only return documents with "
    "meaningful relevance (score > 0.1). Do not invent "
    "document IDs — only return IDs from the provided list."
)

LLMNAVIGATE_SYSTEM = (
    "You are a hierarchical document navigator for a "
    "retrieval system.\n\n"
    "You will receive a user query and a tree of document "
    "sections, shown with their table-of-contents paths, "
    "summaries, and depth levels. The tree represents a "
    "document's semantic structure from broad topics "
    "(shallow depth) to specific passages (deep depth)."
    "\n\n"
    "Your task is to navigate this hierarchy and identify "
    "the subtree roots most relevant to the query. "
    "Selecting a node means \"the content under this "
    "subtree is relevant.\" Consider:\n"
    "- Start broad: which top-level sections relate to "
    "the query?\n"
    "- Narrow down: within those, which subsections are "
    "most specific to what's being asked?\n"
    "- Prefer the most specific relevant node — select a "
    "subsection over its parent when the subsection is a "
    "better match\n"
    "- A node's summary describes all content beneath it "
    "in the tree\n\n"
    "Assign a relevance score (0.0 to 1.0) to each "
    "selected node. Do not invent node IDs — only return "
    "IDs from the provided list."
)


class RetrievalService:
    """Orchestrates bi-path retrieval (FABLE document + node paths).

    Combines LLMselect (document-level) with vector top-K, fuses results,
    routes by budget. When routing is node_level, drills into trees via
    LLMnavigate + TreeExpansion to produce scored leaf node candidates.
    """

    def __init__(
        self,
        llm_service: LLMService,
        embedding_service: EmbeddingService,
        node_repo: NodeRepository,
        doc_repo: DocumentRepository,
    ) -> None:
        self.llm = llm_service
        self.embed = embedding_service
        self.node_repo = node_repo
        self.doc_repo = doc_repo

    def query(
        self,
        session: Session,
        query: str,
        token_budget: int,
    ) -> QueryResponse:
        """Full bi-path retrieval pipeline: embed → LLMselect + vector top-K → fuse → route."""
        # Embed query
        try:
            vectors = self.embed.embed_batch([query])
            query_vector = vectors[0]
        except EmbeddingError as e:
            raise RetrievalError(f"Query embedding failed: {e}") from e

        # Run both retrieval paths
        llm_scores = self._llmselect(session, query)
        vector_scores = self._vector_topk(session, query_vector)

        # Early return for empty corpus
        if not llm_scores and not vector_scores:
            return QueryResponse(
                query=query,
                routing="node_level",
                total_tokens=0,
                documents=[],
            )

        # Fuse results and route by budget
        fused = self._fuse(llm_scores, vector_scores)
        doc_response = self._route(session, query, fused, token_budget)

        if doc_response.routing == "node_level" and doc_response.documents:
            # Phase 8 D-14: node_level_retrieval -> node_fusion -> budget_select
            doc_order = [d.document_id for d in doc_response.documents]
            node_results = self._node_level_retrieval(session, query, query_vector, doc_order)
            fusion_ordered = self._node_fusion(node_results, doc_order)
            selected, total_tokens_used, over_budget = self._budget_select(
                fusion_ordered, token_budget
            )

            # Build ChunkResult list from budget-selected NodeResults (D-10, D-12)
            chunks = [
                ChunkResult(
                    node_id=r.node_id,
                    document_id=r.document_id,
                    content=r.content,
                    token_count=r.token_count,
                    score=r.score,
                    position=r.position,
                    source=r.source,  # type: ignore[arg-type]
                )
                for r in selected
            ]

            return QueryResponse(
                query=query,
                routing="node_level",
                total_tokens=doc_response.total_tokens,
                documents=doc_response.documents,
                node_results=node_results,  # full pre-budget list (D-12)
                chunks=chunks,
                total_tokens_used=total_tokens_used,
                over_budget=over_budget,
            )
        return doc_response

    def _llmselect(
        self,
        session: Session,
        query: str,
    ) -> dict[uuid.UUID, float]:
        """LLM-guided document selection at shallow depth (D-04, D-05, D-06).

        Fetches internal nodes at depth <= retrieval_llmselect_depth,
        groups summaries by document, and asks the LLM to score relevance.
        Returns dict mapping document_id -> relevance_score.
        LLM failure returns {} (vector path provides fallback).
        """
        nodes = self.node_repo.find_internal_nodes_by_depth(
            session, settings.retrieval_llmselect_depth
        )

        # Group by document_id with toc_path + summary per FABLE spec
        sections_by_doc: dict[uuid.UUID, list[tuple[str, str]]] = {}
        for node in nodes:
            doc_id = node.document_id
            if doc_id not in sections_by_doc:
                sections_by_doc[doc_id] = []
            toc = node.toc_path or node.title or "(root)"
            summary = node.summary or "(no summary)"
            sections_by_doc[doc_id].append((toc, summary))

        if not sections_by_doc:
            return {}

        # Build user prompt: present toc path + summary for each section
        user_prompt = f"Query: {query}\n\nDocuments:\n"
        for doc_id, sections in sections_by_doc.items():
            user_prompt += f"\nDocument {doc_id}:\n"
            for toc, summary in sections:
                user_prompt += f"  [{toc}] {summary}\n"

        try:
            result: LLMSelectResult = self.llm.complete_structured(
                response_model=LLMSelectResult,
                messages=[
                    {"role": "system", "content": LLMSELECT_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )  # type: ignore[assignment]
            # Guard against LLM hallucinating document IDs
            return {
                sel.document_id: sel.relevance_score
                for sel in result.selected_documents
                if sel.document_id in sections_by_doc
            }
        except Exception as e:
            logger.warning("LLMselect failed (falling back to vector path): %s", e)
            return {}

    def _llmnavigate(
        self,
        session: Session,
        query: str,
        document_ids: list[uuid.UUID],
    ) -> dict[uuid.UUID, float]:
        """LLM-guided navigation over non-leaf nodes from fused documents (D-01 to D-04).

        Presents all non-leaf nodes grouped by document with depth indicators.
        Returns dict mapping node_id -> relevance_score.
        LLM failure returns {} (TreeExpansion vector path acts as fallback, D-04).
        """
        nodes = self.node_repo.find_internal_nodes_by_depth(
            session, max_depth=99, document_ids=document_ids
        )
        if not nodes:
            return {}

        # Build user prompt: show toc path, summary, and depth indentation (D-01)
        user_prompt = f"Query: {query}\n\nDocument tree:\n"
        for node in nodes:
            indent = "  " * node.depth
            toc = node.toc_path or node.title or "(root)"
            summary = node.summary or "(no summary)"
            user_prompt += f"{indent}[{node.id}] {toc}: {summary}\n"

        try:
            result: LLMNavigateResult = self.llm.complete_structured(
                response_model=LLMNavigateResult,
                messages=[
                    {"role": "system", "content": LLMNAVIGATE_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
            )  # type: ignore[assignment]
            # Guard against LLM hallucinating node IDs (D-01)
            valid_ids = {n.id for n in nodes}
            return {
                sel.node_id: sel.relevance_score
                for sel in result.selected_nodes
                if sel.node_id in valid_ids
            }
        except Exception as e:
            logger.warning("LLMnavigate failed (using TreeExpansion only): %s", e)
            return {}

    def _tree_expansion(
        self,
        session: Session,
        query_vector: list[float],
        document_ids: list[uuid.UUID],
    ) -> tuple[dict[uuid.UUID, float], list[Node]]:
        """TreeExpansion structure-aware scoring over fused documents (D-05 to D-12).

        Bulk-fetches all nodes, computes cosine similarities via HNSW indexes,
        then runs in-memory tree walking to produce normalized leaf scores.
        Returns tuple of (dict mapping leaf node_id -> normalized score [0, 1], all_nodes list).
        """
        # Bulk fetch all nodes for tree walking (D-09, D-14a)
        all_nodes = self.node_repo.find_all_nodes_for_documents(session, document_ids)
        if not all_nodes:
            return {}, []

        # Get leaf similarities from HNSW partial index (D-11, D-12, D-14b)
        leaf_sim_rows = self.node_repo.find_leaf_similarities_for_documents(
            session, query_vector, document_ids
        )
        leaf_similarities = {node_id: sim for node_id, sim in leaf_sim_rows}

        # Get internal node similarities from HNSW partial index (D-11)
        internal_rows = self.node_repo.find_similar_internal_nodes(
            session, query_vector, top_k=9999
        )
        # Filter to only nodes from our fused documents
        doc_id_set = set(document_ids)
        internal_similarities = {
            node_id: sim for node_id, doc_id, sim in internal_rows if doc_id in doc_id_set
        }

        scores = _compute_tree_expansion_scores(all_nodes, leaf_similarities, internal_similarities)
        return scores, all_nodes

    def _node_level_retrieval(
        self,
        session: Session,
        query: str,
        query_vector: list[float],
        document_ids: list[uuid.UUID],
    ) -> list[NodeResult]:
        """Node-level bi-path retrieval: LLMnavigate + TreeExpansion (D-13, D-16).

        Runs both node-level paths, expands LLMnavigate internal-node scores to
        descendant leaves, merges via max-score fusion, and returns scored leaf nodes
        for Phase 8 fusion and budget control.
        """
        # Run both node-level paths
        llm_nav_scores = self._llmnavigate(session, query, document_ids)
        tree_exp_scores, all_nodes = self._tree_expansion(session, query_vector, document_ids)

        # Expand LLMnavigate internal-node scores to descendant leaf nodes.
        # _llmnavigate returns {internal_node_id: score} but merge operates on leaf IDs.
        # Without expansion, all LLMnavigate entries resolve to None in node_map
        # (which only contains leaves) and are silently dropped.
        llm_leaf_scores = _expand_llmnav_to_leaves(llm_nav_scores, all_nodes)

        # Merge: union of both leaf score dicts, max score for overlapping leaves
        # Phase 8 D-01/D-02: track source provenance for priority-based fusion
        all_leaf_ids = set(llm_leaf_scores) | set(tree_exp_scores)
        merged_scores: dict[uuid.UUID, float] = {}
        source_map: dict[uuid.UUID, Literal["llm_guided", "tree_expansion"]] = {}
        for leaf_id in all_leaf_ids:
            merged_scores[leaf_id] = max(
                llm_leaf_scores.get(leaf_id, 0.0),
                tree_exp_scores.get(leaf_id, 0.0),
            )
            # D-02: LLM gets priority in all overlap cases
            source_map[leaf_id] = "llm_guided" if leaf_id in llm_leaf_scores else "tree_expansion"

        # Reuse bulk-fetched nodes from _tree_expansion (no second fetch needed)
        node_map = {n.id: n for n in all_nodes if n.node_type == "leaf"}

        # Build sorted NodeResult list with source field
        results = []
        for leaf_id, score in sorted(merged_scores.items(), key=lambda x: x[1], reverse=True):
            node = node_map.get(leaf_id)
            if node is None:
                continue
            results.append(
                NodeResult(
                    node_id=node.id,
                    document_id=node.document_id,
                    content=node.content,
                    token_count=node.token_count,
                    score=score,
                    depth=node.depth,
                    position=node.position,
                    source=source_map[leaf_id],
                )
            )
        return results

    def _node_fusion(
        self,
        node_results: list[NodeResult],
        doc_order: list[uuid.UUID],
    ) -> list[NodeResult]:
        """Partition by source, order by (doc_rank, position) within each partition (D-05, D-06).

        Priority: LLM-guided leaves first (source == 'llm_guided'), then
        TreeExpansion-only leaves (source == 'tree_expansion').
        Within each partition: sorted by (document rank from doc-level fusion, position).
        """
        llm_group = [r for r in node_results if r.source == "llm_guided"]
        tree_group = [r for r in node_results if r.source == "tree_expansion"]

        def sort_key(result: NodeResult) -> tuple[int, int]:
            try:
                doc_rank = doc_order.index(result.document_id)
            except ValueError:
                doc_rank = len(doc_order)
            return (doc_rank, result.position)

        llm_sorted = sorted(llm_group, key=sort_key)
        tree_sorted = sorted(tree_group, key=sort_key)

        return llm_sorted + tree_sorted

    def _budget_select(
        self,
        fusion_ordered: list[NodeResult],
        token_budget: int,
    ) -> tuple[list[NodeResult], int, bool]:
        """Greedy skip-and-continue budget selection with over-budget fallback (D-07, D-08).

        Returns: (selected_results, total_tokens_used, over_budget).
        - Iterates fusion_ordered; includes chunk if total + chunk.token_count <= budget.
        - Skips chunks that don't fit but continues (a later smaller chunk may fit).
        - If NO chunks fit and fusion_ordered is non-empty, returns the single
          highest-scored chunk with over_budget=True (Pitfall 6 fallback).
        """
        selected: list[NodeResult] = []
        total_used = 0

        for node in fusion_ordered:
            chunk_tokens = node.token_count or 0
            if total_used + chunk_tokens <= token_budget:
                selected.append(node)
                total_used += chunk_tokens
                logger.debug(
                    "Budget: included node %s (%d tokens, running total %d/%d)",
                    node.node_id,
                    chunk_tokens,
                    total_used,
                    token_budget,
                )
            else:
                logger.debug(
                    "Budget: skipped node %s (%d tokens would exceed %d)",
                    node.node_id,
                    chunk_tokens,
                    token_budget,
                )

        if not selected and fusion_ordered:
            # Pitfall 6 fallback: return single best-scored chunk (D-08)
            best = max(fusion_ordered, key=lambda r: r.score)
            logger.warning(
                "Budget: all chunks exceed token_budget=%d; returning best-scored "
                "chunk (node %s, %d tokens) with over_budget=True",
                token_budget,
                best.node_id,
                best.token_count or 0,
            )
            return [best], best.token_count or 0, True

        return selected, total_used, False

    def _vector_topk(
        self,
        session: Session,
        query_vector: list[float],
    ) -> dict[uuid.UUID, float]:
        """Vector top-K retrieval over all nodes (FABLE Algorithm 1, line 7).

        Queries all embedded nodes (internal + leaf) by cosine similarity,
        aggregates to document-level by taking max similarity per document.
        Returns dict mapping document_id -> max similarity score.
        """
        rows = self.node_repo.find_similar_nodes(
            session, query_vector, settings.retrieval_top_k
        )
        doc_scores: dict[uuid.UUID, float] = {}
        for _node_id, document_id, similarity in rows:
            if document_id not in doc_scores or similarity > doc_scores[document_id]:
                doc_scores[document_id] = similarity
        return doc_scores

    def _fuse(
        self,
        llm_scores: dict[uuid.UUID, float],
        vector_scores: dict[uuid.UUID, float],
    ) -> list[tuple[uuid.UUID, float]]:
        """Max-score bi-path fusion (D-10).

        Union of both path document sets. For documents in both paths,
        uses max(llm_score, vector_score). Returns sorted list descending by score.
        """
        all_doc_ids = set(llm_scores) | set(vector_scores)
        fused: dict[uuid.UUID, float] = {}
        for doc_id in all_doc_ids:
            fused[doc_id] = max(
                llm_scores.get(doc_id, 0.0),
                vector_scores.get(doc_id, 0.0),
            )
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)

    def _route(
        self,
        session: Session,
        query: str,
        fused: list[tuple[uuid.UUID, float]],
        token_budget: int,
    ) -> QueryResponse:
        """Budget-adaptive routing (D-11, D-12, D-13).

        Fetches document records and root node titles for all fused documents.
        Computes total_tokens and decides routing:
        - document_level: total tokens <= budget (content included)
        - node_level: total tokens > budget (content omitted, Phase 7 handles drill-down)
        """
        doc_ids = [doc_id for doc_id, _ in fused]

        # Batch-fetch document records
        result = session.execute(select(Document).where(Document.id.in_(doc_ids)))
        doc_map: dict[uuid.UUID, Document] = {d.id: d for d in result.scalars().all()}

        # Batch-fetch root node titles
        roots = session.execute(
            select(Node.document_id, Node.title).where(
                Node.document_id.in_(doc_ids),
                Node.node_type == "root",
            )
        )
        title_map: dict[uuid.UUID, str | None] = {row.document_id: row.title for row in roots}

        # Compute total tokens
        total_tokens = sum(
            doc_map[doc_id].token_count or 0 for doc_id, _ in fused if doc_id in doc_map
        )

        # Determine routing
        routing = "document_level" if total_tokens <= token_budget else "node_level"

        # Build ordered document results
        documents = [
            DocumentResult(
                document_id=doc_id,
                title=title_map.get(doc_id),
                score=score,
                token_count=doc_map[doc_id].token_count,
                content=doc_map[doc_id].content if routing == "document_level" else None,
            )
            for doc_id, score in fused
            if doc_id in doc_map
        ]

        return QueryResponse(
            query=query,
            routing=routing,  # type: ignore[arg-type]
            total_tokens=total_tokens,
            documents=documents,
        )


def get_retrieval_service() -> RetrievalService:
    from openfable.services.embedding_service import get_embedding_service
    from openfable.services.llm_service import get_llm_service

    return RetrievalService(
        llm_service=get_llm_service(),
        embedding_service=get_embedding_service(),
        node_repo=NodeRepository(),
        doc_repo=DocumentRepository(),
    )
