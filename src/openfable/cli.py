"""OpenFable CLI -- the exec target for the agent-driven skill stack.

Ingestion is a handshake, because the LLM work happens outside this process:

    openfable plan <path>                        register the document
    openfable apply-chunks <doc-id> <markers>    persist supplied boundaries
    openfable apply-tree <doc-id> <tree>         persist tree, then embed

Every command writes a single JSON object to stdout. Errors write
{"error": "..."} and exit non-zero, so a caller can read the message, correct
its input, and retry the same step.

`openfable index <path>` runs the whole thing through LiteLLM instead, which
needs an LLM configured. It is the reproducible reference path.
"""

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from openfable.db import SessionLocal
from openfable.exceptions import ChunkingError, TreeConstructionError
from openfable.repositories.document_repo import (
    DocumentRepository,
    compute_content_hash,
    count_tokens,
)
from openfable.schemas.tree import TreeBuildResponse
from openfable.services.ingestion.pipeline import IngestionPipeline
from openfable.services.ingestion.precomputed import (
    PrecomputedChunker,
    PrecomputedTreeStructurer,
)


def _emit(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def _fail(message: str) -> int:
    _emit({"error": message})
    return 1


def _read_json(path: str) -> Any:
    """Load JSON from a file, or from stdin when path is '-'.

    Stdin is the usual route: /work is mounted read-only, so a caller has
    nowhere to put a temp file and pipes the JSON in instead.
    """
    if path == "-":
        return json.loads(sys.stdin.read())
    return json.loads(Path(path).read_text())


def _register_document(source: Path) -> tuple[uuid.UUID, str, int, bool]:
    """Create or reset the Document row for `source`.

    Re-submitting identical content reuses the existing row and clears its
    derived data, matching the REST path's idempotent re-ingest.
    Returns (document_id, content_hash, token_count, was_reingest).
    """
    text = source.read_text()
    content_hash = compute_content_hash(text)
    token_count = count_tokens(text)
    repo = DocumentRepository()

    with SessionLocal() as session:
        existing = repo.get_by_content_hash(session, content_hash)
        if existing:
            repo.reset_document_for_reingest(session, existing.id, text, content_hash, token_count)
            document_id = existing.id
            reingest = True
        else:
            document_id = repo.create(session, text, content_hash, token_count).id
            reingest = False
        session.commit()

    return document_id, content_hash, token_count, reingest


def _validate_source(path: str) -> tuple[Path, str | None]:
    source = Path(path)
    if not source.is_file():
        return source, f"No such file: {source}"
    if not source.read_text().strip():
        return source, f"{source} is empty."
    return source, None


# ---------------------------------------------------------------------------
# Ingestion handshake
# ---------------------------------------------------------------------------


def cmd_plan(args: argparse.Namespace) -> int:
    """Register a document and report what the caller must produce next."""
    source, error = _validate_source(args.path)
    if error:
        return _fail(error)

    document_id, content_hash, token_count, reingest = _register_document(source)

    _emit(
        {
            "document_id": document_id,
            "content_hash": content_hash,
            "token_count": token_count,
            "reingest": reingest,
            "source": str(source),
            "next": (
                "Read the document, choose semantic chunk boundaries, and produce a "
                'JSON array of verbatim opening snippets -- e.g. ["# Title", '
                '"## Section 2 ..."] -- one per chunk, in document order. Pipe it to: '
                f"openfable apply-chunks {document_id} -"
            ),
        }
    )
    return 0


def cmd_apply_chunks(args: argparse.Namespace) -> int:
    """Resolve supplied boundary markers into chunks and persist them."""
    try:
        raw = _read_json(args.markers)
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(f"Could not read markers file {args.markers}: {exc}")

    markers = raw.get("markers") if isinstance(raw, dict) else raw
    if not isinstance(markers, list) or not all(isinstance(m, str) for m in markers):
        return _fail(
            "Markers file must be a JSON array of strings, or an object with a "
            '"markers" key holding one.'
        )

    document_id = uuid.UUID(args.document_id)
    pipeline = IngestionPipeline()

    with SessionLocal() as session:
        try:
            pipeline.chunk_stage(session, document_id, PrecomputedChunker(markers))
        except ChunkingError as exc:
            return _fail(str(exc))
        chunks = pipeline.load_chunks(session, document_id)

        _emit(
            {
                "document_id": document_id,
                "chunk_count": len(chunks),
                "chunks": [
                    {
                        "chunk_index": c.position,
                        "token_count": c.token_count,
                        "preview": (c.content or "")[:160],
                    }
                    for c in chunks
                ],
                "next": (
                    "Organise these chunk_index values into a tree and write it as "
                    '{"root": {"type": "internal", "node_type": "root", "title": ..., '
                    '"summary": ..., "children": [...]}}, where leaves are '
                    '{"type": "leaf", "chunk_index": N}. Every chunk_index must appear '
                    f"exactly once. Pipe it to: openfable apply-tree {document_id} -"
                ),
            }
        )
    return 0


def cmd_apply_tree(args: argparse.Namespace) -> int:
    """Persist a supplied tree, then embed every node."""
    try:
        raw = _read_json(args.tree)
    except (OSError, json.JSONDecodeError) as exc:
        return _fail(f"Could not read tree file {args.tree}: {exc}")

    try:
        parsed = TreeBuildResponse.model_validate(raw)
    except Exception as exc:
        return _fail(f"Tree does not match the expected schema: {exc}")

    document_id = uuid.UUID(args.document_id)
    pipeline = IngestionPipeline()

    with SessionLocal() as session:
        try:
            chunks = pipeline.tree_stage(
                session, document_id, PrecomputedTreeStructurer(parsed.root)
            )
        except TreeConstructionError as exc:
            return _fail(str(exc))
        embedded = pipeline.embed_stage(session, document_id)

    _emit(
        {
            "document_id": document_id,
            "chunk_count": len(chunks),
            "nodes_embedded": embedded,
            "status": "indexed",
        }
    )
    return 0


# ---------------------------------------------------------------------------
# Reference path + reads
# ---------------------------------------------------------------------------


def cmd_index(args: argparse.Namespace) -> int:
    """Full LiteLLM-driven ingest. Needs an LLM configured."""
    source, error = _validate_source(args.path)
    if error:
        return _fail(error)

    document_id, _, token_count, reingest = _register_document(source)

    with SessionLocal() as session:
        IngestionPipeline().run(session, document_id)

    _emit(
        {
            "document_id": document_id,
            "token_count": token_count,
            "reingest": reingest,
            "status": "indexed",
            "via": "litellm",
        }
    )
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    from openfable.services.retrieval_service import get_retrieval_service

    service = get_retrieval_service()
    if args.vector_only:
        # RetrievalService already treats an LLM failure as "fall back to the
        # vector path", so a service that always raises yields vector-only
        # retrieval without a second code path to maintain.
        class _NoLLM:
            def complete_structured(self, *a: Any, **kw: Any) -> Any:
                raise RuntimeError("vector-only mode: LLM paths disabled")

        service.llm = _NoLLM()  # type: ignore[assignment]

    with SessionLocal() as session:
        response = service.query(session, args.query, args.budget)
        _emit(response.model_dump())
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    repo = DocumentRepository()
    with SessionLocal() as session:
        docs = repo.list_all(session)
        _emit(
            {
                "total": len(docs),
                "documents": [
                    {
                        "document_id": d.id,
                        "content_hash": d.content_hash,
                        "token_count": d.token_count,
                        "created_at": d.created_at,
                    }
                    for d in docs
                ],
            }
        )
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    repo = DocumentRepository()
    with SessionLocal() as session:
        doc = repo.get_by_id(session, uuid.UUID(args.document_id))
        if doc is None:
            return _fail(f"Document {args.document_id} not found")
        _emit(
            {
                "document_id": doc.id,
                "content_hash": doc.content_hash,
                "token_count": doc.token_count,
                "created_at": doc.created_at,
                "content": None if args.meta_only else doc.content,
            }
        )
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    from sqlalchemy import text as sql_text

    from openfable.services.embedding_service import EmbeddingService

    components: dict[str, str] = {}
    ok = True

    try:
        with SessionLocal() as session:
            session.execute(sql_text("SELECT 1"))
        components["database"] = "healthy"
    except Exception as exc:
        components["database"] = f"unhealthy: {exc}"
        ok = False

    try:
        EmbeddingService().embed_batch(["health probe"])
        components["embeddings"] = "healthy"
    except Exception as exc:
        components["embeddings"] = f"unhealthy: {exc}"
        ok = False

    _emit({"status": "healthy" if ok else "unhealthy", "components": components})
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openfable")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="register a document for agent-driven indexing")
    p.add_argument("path")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("apply-chunks", help="persist supplied chunk boundaries")
    p.add_argument("document_id")
    p.add_argument(
        "markers",
        help="JSON file, or '-' to read from stdin: array of chunk-opening snippets",
    )
    p.set_defaults(func=cmd_apply_chunks)

    p = sub.add_parser("apply-tree", help="persist supplied tree, then embed")
    p.add_argument("document_id")
    p.add_argument("tree", help="JSON file, or '-' for stdin: {\"root\": {...}}")
    p.set_defaults(func=cmd_apply_tree)

    p = sub.add_parser("index", help="full LiteLLM ingest (needs an LLM configured)")
    p.add_argument("path")
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("query", help="bi-path retrieval within a token budget")
    p.add_argument("query")
    p.add_argument("--budget", type=int, default=2000, help="token budget (100-32000)")
    p.add_argument(
        "--vector-only",
        action="store_true",
        help="skip LLMselect/LLMnavigate; needs no LLM",
    )
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("list", help="list indexed documents")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("get", help="fetch one document")
    p.add_argument("document_id")
    p.add_argument("--meta-only", action="store_true")
    p.set_defaults(func=cmd_get)

    p = sub.add_parser("health", help="check database and embedding server")
    p.set_defaults(func=cmd_health)

    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        result: int = args.func(args)
        return result
    except ValueError as exc:
        return _fail(str(exc))


if __name__ == "__main__":
    sys.exit(main())
