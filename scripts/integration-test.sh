#!/usr/bin/env bash
# Run the test suite against real collaborators: PostgreSQL and a
# deterministic LLM provider.
#
#   ./scripts/integration-test.sh              whole suite
#   ./scripts/integration-test.sh -k retry     any pytest arguments
#
# One operation: start clean, run once, tear down. Exits with pytest's status.
set -euo pipefail

cd "$(dirname "$0")/.."

COMPOSE=(docker compose -f docker-compose.integration.yml)

# Runs on success, failure and interrupt alike. Failures here are reported
# rather than swallowed: a teardown that cannot complete is worth seeing, and
# hiding it once already turned a one-line problem into a long diagnosis.
cleanup() {
  "${COMPOSE[@]}" down -v --remove-orphans >/dev/null 2>&1 \
    || echo "warning: teardown incomplete" >&2
}
trap cleanup EXIT INT TERM

"${COMPOSE[@]}" build --quiet

# `run` starts db and agent-testkit and waits for their health checks, so the
# suite never executes against a stack that is not up. If a service cannot
# become healthy -- a malformed agent-testkit.config.ts, say -- compose fails
# here and prints why.
"${COMPOSE[@]}" run --rm tests uv run pytest --tb=short -q "$@"
