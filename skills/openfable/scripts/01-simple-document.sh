#!/usr/bin/env bash
# JOURNEY: index one short, flat document — the minimal complete case.
#
# Read this first. Everything else is a variation on these three steps.
# Document: CONTRIBUTING.md, four top-level headings, no nesting.
set -euo pipefail
OF="docker compose -f .openfable/docker-compose.yml"

# ---- Step 1: register -------------------------------------------------------
# `plan` only records the document. It does no chunking — that is your job.
DOC_ID=$($OF exec -T openfable openfable plan /work/CONTRIBUTING.md \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["document_id"])')
  : "${DOC_ID:?plan failed — see the error above}"
echo "registered: $DOC_ID"

# ---- Step 2: chunk boundaries ----------------------------------------------
# One marker per topic shift. Here the headings ARE the topic shifts, so each
# heading is a boundary. Markers are verbatim openings — copy them from the
# file, do not paraphrase.
#
# Note there is no marker for the closing list under "Submitting changes": it
# belongs to that topic, so it stays inside that chunk.
$OF exec -T openfable openfable apply-chunks "$DOC_ID" - <<'JSON'
["# Contributing",
 "## Setup (Docker — recommended)",
 "## Setup (local)",
 "## Submitting changes"]
JSON

# ---- Step 3: tree -----------------------------------------------------------
# Four chunks. The two setup chunks are the same topic seen two ways, so they
# get a shared parent. The intro and the PR process stand alone as leaves of
# the root — not every chunk needs a section wrapper.
#
# The summary on "Development Setup" states what the subtree SAYS. That text is
# embedded and searched; "this section covers setup" would retrieve nothing.
$OF exec -T openfable openfable apply-tree "$DOC_ID" - <<'JSON'
{"root": {"type": "internal", "node_type": "root",
  "title": "Contributing to OpenFable",
  "summary": "Docker and local development setup, test and lint commands, and the pull-request process.",
  "children": [
    {"type": "leaf", "chunk_index": 0},
    {"type": "internal", "node_type": "section",
     "title": "Development Setup",
     "summary": "Dev container with hot reload, or a local uv install; running pytest, ruff check and mypy.",
     "children": [
       {"type": "leaf", "chunk_index": 1},
       {"type": "leaf", "chunk_index": 2}
     ]},
    {"type": "leaf", "chunk_index": 3}
  ]}}
JSON

# ---- Verify -----------------------------------------------------------------
# Tight budget so it must drill into the tree rather than return the whole doc.
$OF exec -T openfable openfable query "how do I run type checking" \
  --budget 300 --vector-only
