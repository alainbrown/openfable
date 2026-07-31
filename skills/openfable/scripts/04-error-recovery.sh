#!/usr/bin/env bash
# JOURNEY: both failure modes, and how to recover from each.
#
# Every error is actionable and names what is wrong. Correct your input and
# re-run THE SAME step — never restart the workflow. Nothing is left
# half-written: a rejected step persists nothing.
#
# `set -e` is deliberately NOT used here, because we expect non-zero exits.
set -uo pipefail
OF="docker compose -f .openfable/docker-compose.yml"

DOC_ID=$($OF exec -T openfable openfable plan /work/CONTRIBUTING.md \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["document_id"])')
  : "${DOC_ID:?plan failed — see the error above}"
echo "registered: $DOC_ID"

# ---- Failure 1: a marker that is not verbatim ------------------------------
# "## Setup (Docker - recommended)" uses a hyphen. The file has an em-dash.
# Whitespace differences are tolerated; different characters are not.
echo "== expect: marker not found =="
$OF exec -T openfable openfable apply-chunks "$DOC_ID" - <<'JSON'
["# Contributing",
 "## Setup (Docker - recommended)",
 "## Submitting changes"]
JSON
echo "exit=$?  <- non-zero, nothing persisted"

# RECOVERY: the error quotes the offending marker and gives the offset it
# searched from. Copy the text out of the file exactly. Re-send the whole
# array — this step is all-or-nothing, not incremental.
echo "== corrected =="
$OF exec -T openfable openfable apply-chunks "$DOC_ID" - <<'JSON'
["# Contributing",
 "## Setup (Docker — recommended)",
 "## Setup (local)",
 "## Submitting changes"]
JSON

# ---- Failure 2: incomplete chunk coverage ----------------------------------
# The previous step produced chunk_index 0..3. A tree covering only 0 and 1
# is rejected: every chunk must appear exactly once. Duplicates are rejected
# the same way.
echo "== expect: missing chunk indexes =="
$OF exec -T openfable openfable apply-tree "$DOC_ID" - <<'JSON'
{"root": {"type": "internal", "node_type": "root",
  "title": "Contributing", "summary": "How to contribute.",
  "children": [
    {"type": "leaf", "chunk_index": 0},
    {"type": "leaf", "chunk_index": 1}
  ]}}
JSON
echo "exit=$?  <- non-zero, no tree written"

# RECOVERY: the error lists exactly which indexes are missing. Add them.
echo "== corrected =="
$OF exec -T openfable openfable apply-tree "$DOC_ID" - <<'JSON'
{"root": {"type": "internal", "node_type": "root",
  "title": "Contributing to OpenFable",
  "summary": "Docker and local development setup, test and lint commands, and the pull-request process.",
  "children": [
    {"type": "leaf", "chunk_index": 0},
    {"type": "internal", "node_type": "section",
     "title": "Development Setup",
     "summary": "Dev container with hot reload, or a local uv install; running pytest, ruff and mypy.",
     "children": [
       {"type": "leaf", "chunk_index": 1},
       {"type": "leaf", "chunk_index": 2}
     ]},
    {"type": "leaf", "chunk_index": 3}
  ]}}
JSON

# ---- Other errors you may see ----------------------------------------------
#   "No such file"                            -> use the /work/... path, not a
#                                                host path. Check: $OF exec
#                                                openfable ls /work
#   "Tree does not match the expected schema" -> a node is malformed. node_type
#                                                must be root|section|subsection
#                                                and every leaf needs both
#                                                "type" and "chunk_index".
#   "Duplicate chunk_index values found"      -> a chunk appears twice.
