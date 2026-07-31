#!/usr/bin/env bash
# JOURNEY: answer a question from an already-indexed corpus. No indexing.
#
# This is the common case and the cheapest. Always check `list` before you
# reach for the indexing workflow — the answer may already be retrievable.
set -euo pipefail
OF="docker compose -f .openfable/docker-compose.yml"

# ---- What is already indexed? ----------------------------------------------
echo "== corpus =="
$OF exec -T openfable openfable list

# ---- The budget ladder ------------------------------------------------------
# Same question, three budgets. The budget does not just truncate the answer;
# it changes HOW retrieval runs.
#
#   small budget -> routing "node_level": drills into the tree, returns
#                   individual chunks with scores.
#   large budget -> routing "document_level": whole documents fit, so they come
#                   back entire and `chunks` is absent.
#
# Start at 2000. Narrow when you get noise, widen when the answer looks cut off.
for BUDGET in 200 2000 12000; do
  echo "== budget=$BUDGET =="
  $OF exec -T openfable openfable query "how are chunk boundaries chosen" \
    --budget "$BUDGET" --vector-only \
  | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("  routing:", d["routing"], "| used:", d.get("total_tokens_used"),
      "| over_budget:", d.get("over_budget"))
for c in (d.get("chunks") or [])[:3]:
    head = (c["content"] or "").strip().splitlines()[0]
    print("    %.3f  %s" % (c["score"], head[:60]))
for doc in d.get("documents", [])[:3]:
    print("    doc %.3f  %s tokens" % (doc["score"], doc["token_count"]))
'
done

# ---- Reading the response ---------------------------------------------------
# over_budget: true means the budget could not fit even the top result. Raise
# it and re-query rather than trying to interpret a truncated answer.
