#!/usr/bin/env bash
# JOURNEY: a document with real hierarchy — mapping heading depth to node types,
# and what to do when the document nests deeper than the tree allows.
#
# Document: fixtures/orbital-q1-report.md, which nests to four heading levels
# (## 2 -> ### 2.2 -> #### 2.2.1). The buried fact is in 2.2.1: the throughput
# numbers only hold at night. A flat index would lose that qualifier.
set -euo pipefail
OF="docker compose -f .openfable/docker-compose.yml"
DOC=/work/skills/openfable/assets/fixtures/orbital-q1-report.md

DOC_ID=$($OF exec -T openfable openfable plan "$DOC" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["document_id"])')
  : "${DOC_ID:?plan failed — see the error above}"
echo "registered: $DOC_ID"

# ---- Boundaries at the deepest meaningful level ----------------------------
# Split at #### here, not at ##. "Measurement Conditions" has to be its own
# chunk: it is the qualifier that makes the throughput number meaningful, and
# a question about measurement conditions must be able to retrieve it alone.
#
# Splitting only at ## would bury it inside a 300-token chunk where a tight
# budget could never surface it.
$OF exec -T openfable openfable apply-chunks "$DOC_ID" - <<'JSON'
["# Helios Array: Q1 2027 Engineering Report",
 "## 2. Optical Performance",
 "### 2.1 Alignment",
 "#### 2.2.1 Measurement Conditions",
 "#### 2.2.2 Recorded Throughput",
 "#### 2.2.3 Target",
 "## 3. Thermal Management",
 "## 4. Programme Cost"]
JSON

# ---- Mirror the document's hierarchy ----------------------------------------
# node_type tracks depth: root -> section -> subsection. There are only these
# three internal types and a hard depth cap of 4 (root=1, section=2,
# subsection=3, leaf=4).
#
# The document nests one level deeper than that (## -> ### -> #### -> text), so
# something must collapse. Here "2.2 Throughput" is dropped as a level and its
# three #### chunks hang directly off the "Optical Performance" subsection —
# the distinction between 2.2.1/2/3 matters more than the 2.2 grouping does.
#
# If you exceed depth 4 anyway, OpenFable re-parents the offending node to its
# grandparent and logs a warning. Better to choose the collapse yourself.
$OF exec -T openfable openfable apply-tree "$DOC_ID" - <<'JSON'
{"root": {"type": "internal", "node_type": "root",
  "title": "Helios Array Q1 2027 Engineering Report",
  "summary": "First-light status of the northern reflector cluster: optical alignment and figure error, downlink throughput and its measurement conditions, thermal control, and programme spend against allocation.",
  "children": [
    {"type": "leaf", "chunk_index": 0},
    {"type": "internal", "node_type": "section",
     "title": "Optical Performance",
     "summary": "Reflector alignment at 14 nm RMS figure error, and sustained downlink throughput with the conditions under which it was measured.",
     "children": [
       {"type": "leaf", "chunk_index": 1},
       {"type": "leaf", "chunk_index": 2},
       {"type": "internal", "node_type": "subsection",
        "title": "Throughput",
        "summary": "780 GB/s aggregate downlink measured during night-cycle operation with the thermal shroud deployed; daytime figures excluded. Q4 target of 1.2 TB/s depends on the Ka-band amplifier refit.",
        "children": [
          {"type": "leaf", "chunk_index": 3},
          {"type": "leaf", "chunk_index": 4},
          {"type": "leaf", "chunk_index": 5}
        ]}
     ]},
    {"type": "leaf", "chunk_index": 6},
    {"type": "leaf", "chunk_index": 7}
  ]}}
JSON

# ---- The payoff -------------------------------------------------------------
# A tight budget drills to node level and ranks the measurement-conditions
# chunk FIRST (~0.97), ahead of the throughput number it qualifies. That is
# what splitting at #### bought: the qualifier is retrievable on its own terms.
#
# Several chunks come back, not one — the budget is filled with the best
# candidates. Read them in score order.
echo "== drill-down to the buried qualifier =="
$OF exec -T openfable openfable query \
  "under what conditions were the throughput figures measured" \
  --budget 200 --vector-only \
| python3 -c '
import json, sys
d = json.load(sys.stdin)
print("routing:", d["routing"], "| used:", d.get("total_tokens_used"))
for c in sorted(d.get("chunks") or [], key=lambda c: -c["score"]):
    print("  %.4f  %s" % (c["score"], (c["content"] or "").strip().splitlines()[0][:55]))
'
