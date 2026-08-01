# Contributing

Thanks for your interest in OpenFable! Here's how to get started.

## Setup (Docker — recommended)

The dev container has Python 3.12, uv, and all dependencies pre-installed.

Start the API server with hot reload (auto-restarts on file changes):

```bash
docker compose -f docker-compose.dev.yml up
```

The API is at http://localhost:8000. Edit files locally — changes reload automatically.

For an interactive shell (tests, linting, type checking):

```bash
docker compose -f docker-compose.dev.yml run --rm api
```

```bash
# Tests
.venv/bin/pytest

# Lint and type check
.venv/bin/ruff check src/ tests/
.venv/bin/mypy src/
```

## Setup (local)

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --dev
```

Tests (no external services needed):

```bash
uv run pytest
```

Linting and type checking:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
```

## Tests

```bash
./scripts/integration-test.sh              # everything
./scripts/integration-test.sh -k retry     # any pytest arguments
```

One operation: it starts PostgreSQL and a deterministic LLM provider, runs the
suite once against them, and tears everything down again -- including on
failure or Ctrl-C. Every run begins with an empty database, so nothing carries
over between runs. CI runs the same command.

The provider is [agent-testkit](https://github.com/agent-testkit/agent-testkit),
pinned to a release tag. Its behaviours live in
`tests/integration/agent-testkit.config.ts` and are matched on fragments of
each prompt, so **changing a prompt's opening words breaks them**. The server
also errors when several behaviours match rather than taking the first, which
is why case-specific behaviours carry `priority: 10`.

## Working on the skill

The Claude Code plugin lives in `skills/openfable/`, with manifests in
`.claude-plugin/`. Nothing in CI covers it, so check changes by hand.

```bash
claude plugin validate .
```

`SKILL.md` has a YAML frontmatter and an **XML body**. Keep it well-formed, keep
`name` matching the directory, and keep the file under 500 lines — agents load
it whole. Paths it references must stay one level deep, per the
[Agent Skills spec](https://agentskills.io/specification).

```bash
python3 -c "
import re, pathlib, xml.etree.ElementTree as ET
raw = pathlib.Path('skills/openfable/SKILL.md').read_text()
ET.fromstring(re.sub(r'^---\n.*?\n---\n', '', raw, count=1, flags=re.S))
print('ok', raw.count(chr(10)) + 1, 'lines')"
```

The six scripts in `skills/openfable/scripts/` are runnable journeys that double
as regression tests. They need a live stack, so onboard first (or generate
`.openfable/` by hand from `skills/openfable/assets/compose-all-local.yml`),
then:

```bash
bash -n skills/openfable/scripts/*.sh          # syntax only
bash skills/openfable/scripts/01-simple-document.sh
```

Two things worth knowing before editing them. Error strings quoted in `SKILL.md`
and in the scripts must match `src/openfable/` verbatim — they are the contract
an agent retries against. And a failing `VAR=$(...)` does not trip `set -e`, so
every command substitution capturing a `document_id` is followed by a
`: "${VAR:?...}"` guard; keep them.

## Submitting changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Ensure tests, linting, and type checking pass
4. Open a pull request
