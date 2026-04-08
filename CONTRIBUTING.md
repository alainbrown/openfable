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
# Run unit tests
.venv/bin/pytest -m "not integration"

# Run all tests (integration tests use the dev compose PostgreSQL)
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

Unit tests (no external services needed):

```bash
uv run pytest -m "not integration"
```

Integration tests (requires Docker for PostgreSQL via testcontainers):

```bash
uv run pytest
```

Linting and type checking:

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/
```

## Submitting changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Ensure tests, linting, and type checking pass
4. Open a pull request
