#!/usr/bin/env bash
# JOURNEY: index two documents that disagree, then retrieve across both.
#
# This is what OpenFable is actually for. Two sources make the same claim under
# different conditions; the answer exists in neither document alone. Retrieval
# has to pull the matching sections out of both at once.
#
# Documents: the Q1 engineering report (780 GB/s) and the Q2 audit that
# replicates it and then reports 415 GB/s under conditions Q1 never tested.
set -euo pipefail
OF="docker compose -f .openfable/docker-compose.yml"
REF=/work/skills/openfable/assets/fixtures

index_report() {
  local id
  id=$($OF exec -T openfable openfable plan "$REF/orbital-q1-report.md" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["document_id"])')
    : "${id:?plan failed — see the error above}"

  $OF exec -T openfable openfable apply-chunks "$id" - >/dev/null <<'JSON'
["# Helios Array: Q1 2027 Engineering Report",
 "## 2. Optical Performance",
 "#### 2.2.1 Measurement Conditions",
 "#### 2.2.2 Recorded Throughput",
 "#### 2.2.3 Target",
 "## 3. Thermal Management",
 "## 4. Programme Cost"]
JSON

  # Summaries carry the CLAIMS, including the conditions attached to them.
  # "Throughput results" would be useless here — the whole question is under
  # what conditions the number holds, so the condition goes in the summary.
  $OF exec -T openfable openfable apply-tree "$id" - >/dev/null <<'JSON'
{"root": {"type": "internal", "node_type": "root",
  "title": "Helios Array Q1 2027 Engineering Report",
  "summary": "First-light engineering status: optical alignment, downlink throughput measured at night with the shroud deployed, thermal control, and spend at 61% of allocation.",
  "children": [
    {"type": "leaf", "chunk_index": 0},
    {"type": "internal", "node_type": "section",
     "title": "Optical Performance and Throughput",
     "summary": "780 GB/s aggregate downlink recorded during night-cycle operation only, with daytime passes excluded; Q4 target of 1.2 TB/s assumes the Ka-band refit lands in August.",
     "children": [
       {"type": "leaf", "chunk_index": 1},
       {"type": "leaf", "chunk_index": 2},
       {"type": "leaf", "chunk_index": 3},
       {"type": "leaf", "chunk_index": 4}
     ]},
    {"type": "leaf", "chunk_index": 5},
    {"type": "leaf", "chunk_index": 6}
  ]}}
JSON
  echo "$id"
}

index_audit() {
  local id
  id=$($OF exec -T openfable openfable plan "$REF/orbital-audit.md" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["document_id"])')
    : "${id:?plan failed — see the error above}"

  $OF exec -T openfable openfable apply-chunks "$id" - >/dev/null <<'JSON'
["# Helios Array: Independent Technical Audit, Q2 2027",
 "## 2. Throughput Under Daytime Conditions",
 "### 2.2 Assessment of the Q4 Target",
 "## 3. Cost Projection"]
JSON

  $OF exec -T openfable openfable apply-tree "$id" - >/dev/null <<'JSON'
{"root": {"type": "internal", "node_type": "root",
  "title": "Helios Array Independent Technical Audit Q2 2027",
  "summary": "Assurance review of the Q1 report: confirms 780 GB/s at night but measures 415 GB/s in daylight, judges the Q4 target unachievable, and projects a 13-18% cost overrun.",
  "children": [
    {"type": "leaf", "chunk_index": 0},
    {"type": "internal", "node_type": "section",
     "title": "Throughput and Target Assessment",
     "summary": "Daytime sustained throughput of 415 GB/s against the Q1 night-cycle figure of 780 GB/s, attributed to thermal figure error rising to 61 nm RMS; recommends restating the Q4 target as 900 GB/s.",
     "children": [
       {"type": "leaf", "chunk_index": 1},
       {"type": "leaf", "chunk_index": 2}
     ]},
    {"type": "leaf", "chunk_index": 3}
  ]}}
JSON
  echo "$id"
}

REPORT_ID=$(index_report); echo "indexing report: $REPORT_ID"
AUDIT_ID=$(index_audit);   echo "indexing audit:  $AUDIT_ID"

# ---- Cross-document retrieval -----------------------------------------------
# Budget matters more here than in single-document work. A tight budget will
# surface one source and silently hide the disagreement — which reads like a
# confident, wrong answer. Give synthesis questions room.
#
# The check below looks for these two specific document_ids. The corpus may
# hold unrelated documents (these scripts share one database), so counting
# distinct documents would not prove both sources were retrieved.
echo "== budget=8000: both sources must appear =="
$OF exec -T openfable openfable query \
  "what is the sustained downlink throughput and under what conditions" \
  --budget 8000 --vector-only \
| REPORT_ID="$REPORT_ID" AUDIT_ID="$AUDIT_ID" python3 -c '
import json, os, sys
d = json.load(sys.stdin)
found = {c["document_id"] for c in (d.get("chunks") or [])}
found |= {x["document_id"] for x in d.get("documents", [])}
want = {"report": os.environ["REPORT_ID"], "audit": os.environ["AUDIT_ID"]}
print("routing:", d["routing"])
for name, doc_id in want.items():
    print("  %-7s %s" % (name, "RETRIEVED" if doc_id in found else "MISSING"))
if not set(want.values()) <= found:
    print("  Only one side of the disagreement came back. Raise the budget, or")
    print("  make each root summary state what is distinctive about that source.")
    sys.exit(1)
print("  both sources retrieved — the contradiction is visible to the reader")
'
