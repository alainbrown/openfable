#!/usr/bin/env bash
# JOURNEY: retrieval returns something vaguely related instead of the fact you
# asked for. The usual cause is chunks that are too coarse.
#
# A chunk is the smallest unit retrieval can return. If the fact you need sits
# inside a 290-token chunk, a 150-token budget can never deliver it — no amount
# of re-querying helps. Granularity is decided at index time and is the single
# most consequential choice you make.
#
# This script indexes one document twice, coarse then fine, and reports the
# size of the chunk holding the answer. Document: fixtures/ferrite-cache-notes.md,
# where "flush concurrency must never exceed 6" sits mid-section.
set -euo pipefail
OF="docker compose -f .openfable/docker-compose.yml"
DOC=/work/skills/openfable/assets/fixtures/ferrite-cache-notes.md
Q="what is the maximum safe flush concurrency on a single-socket host"

register() {
  $OF exec -T openfable openfable plan "$DOC" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["document_id"])'
}

# Report the size of whichever chunk contains the answer. This is deterministic
# — it depends only on where you put the boundaries.
answer_chunk_size() {
  python3 -c '
import json, sys
d = json.load(sys.stdin)
for c in d["chunks"]:
    if "flush concurrency" in c["preview"] or "never exceed" in c["preview"]:
        print("    answer lives in chunk %d at %d tokens" % (c["chunk_index"], c["token_count"]))
        break
else:
    big = max(d["chunks"], key=lambda c: c["token_count"])
    print("    answer is inside chunk %d (%d tokens) — too large to preview"
          % (big["chunk_index"], big["token_count"]))
print("    %d chunks total" % d["chunk_count"])
'
}

# ---- Pass 1: coarse — split at headings only --------------------------------
# The obvious choice, and wrong here. "Tuning and Operational Limits" becomes
# one large chunk covering three separate tunables; the concurrency limit is a
# single sentence inside it.
DOC_ID=$(register)
: "${DOC_ID:?plan failed — see the error above}"
echo "pass 1 — coarse (headings only):"
$OF exec -T openfable openfable apply-chunks "$DOC_ID" - <<'JSON' | answer_chunk_size
["# Ferrite Cache Subsystem: Operator Notes",
 "## Tuning and Operational Limits",
 "## Recovery"]
JSON
$OF exec -T openfable openfable apply-tree "$DOC_ID" - >/dev/null <<'JSON'
{"root": {"type": "internal", "node_type": "root",
  "title": "Ferrite Cache Subsystem Operator Notes",
  "summary": "Write-behind cache between planner and columnar store: runtime tunables, their interactions and limits, and intent-log replay on unclean shutdown.",
  "children": [
    {"type": "leaf", "chunk_index": 0},
    {"type": "leaf", "chunk_index": 1},
    {"type": "leaf", "chunk_index": 2}
  ]}}
JSON

# ---- Pass 2: fine — one chunk per tunable -----------------------------------
# Same document, same tree shape. Only the boundaries move: each tunable
# becomes its own chunk, so the concurrency limit is retrievable alone.
#
# Boundaries need not be headings. These markers are mid-paragraph sentence
# openings — any verbatim opening works.
DOC_ID=$(register)   # identical content -> reingest: true, old index cleared
echo "pass 2 — fine (one chunk per tunable):"
$OF exec -T openfable openfable apply-chunks "$DOC_ID" - <<'JSON' | answer_chunk_size
["# Ferrite Cache Subsystem: Operator Notes",
 "## Tuning and Operational Limits",
 "The first tunable is the residency window",
 "The second is the admission high-water mark.",
 "The third is flush concurrency,",
 "Beyond these three, the segment size",
 "## Recovery"]
JSON
$OF exec -T openfable openfable apply-tree "$DOC_ID" - >/dev/null <<'JSON'
{"root": {"type": "internal", "node_type": "root",
  "title": "Ferrite Cache Subsystem Operator Notes",
  "summary": "Write-behind cache between planner and columnar store: runtime tunables, their interactions and limits, and intent-log replay on unclean shutdown.",
  "children": [
    {"type": "leaf", "chunk_index": 0},
    {"type": "internal", "node_type": "section",
     "title": "Tuning and Operational Limits",
     "summary": "Residency window trading burst absorption against recovery objective, admission high-water mark as hard backpressure, and a flush concurrency ceiling of 6 on single-socket hosts above which compaction-latch contention reduces throughput.",
     "children": [
       {"type": "leaf", "chunk_index": 1},
       {"type": "leaf", "chunk_index": 2},
       {"type": "leaf", "chunk_index": 3},
       {"type": "leaf", "chunk_index": 4},
       {"type": "leaf", "chunk_index": 5}
     ]},
    {"type": "leaf", "chunk_index": 6}
  ]}}
JSON

# ---- Retrieval against the fine index ---------------------------------------
echo "query at budget=400:"
$OF exec -T openfable openfable query "$Q" --budget 400 --vector-only \
| python3 -c '
import json, sys
d = json.load(sys.stdin)
hits = sorted(d.get("chunks") or [], key=lambda c: -c["score"])
got = [c for c in hits if "never exceed 6" in (c["content"] or "")]
print("    routing=%s used=%s  answer returned: %s"
      % (d["routing"], d.get("total_tokens_used"), "YES" if got else "no"))
for c in hits[:4]:
    mark = "<--" if "never exceed 6" in (c["content"] or "") else "   "
    print("    %.3f %3dtok %s %s"
          % (c["score"], c["token_count"], mark, (c["content"] or "").strip().splitlines()[0][:40]))
'

# ---- Rules of thumb ---------------------------------------------------------
#   * one chunk per distinct fact a user might ask about
#   * a chunk larger than your typical budget can NEVER be returned at that
#     budget; compare token_count against the budgets you expect to use
#   * boundaries need not be headings — split mid-section when the topic shifts
#   * over-splitting is cheaper than under-splitting, but strips context; keep
#     each chunk self-contained enough to be read alone
#
# CAVEAT on the scores printed above: node scores are normalised per document,
# so they are NOT comparable across documents. In a corpus holding several
# documents, an unrelated document's best chunk can show a higher score than
# the chunk that actually answers your question, and a larger budget can pull
# that noise in ahead of the answer. Trust the document-level ranking first,
# and treat node scores as meaningful only within one document.
