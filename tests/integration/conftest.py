"""Integration test fixtures.

Services come from docker-compose.integration.yml and are reached by name.
Nothing here starts a container, so no Docker socket is required.

Run them with:

    ./scripts/integration-test.sh

Both URLs come from docker-compose.integration.yml as fixed values, and the
runner cannot start until both services report healthy. There is no
half-configured state to guard against.
"""

import os
import pathlib
import uuid
from collections.abc import Generator

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

DB_URL = os.environ.get("OPENFABLE_TEST_DATABASE_URL")
LLM_URL = os.environ.get("OPENFABLE_TEST_LLM_BASE_URL")

_HERE = pathlib.Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark tests in THIS directory `integration`.

    The path check matters: pytest hands this hook every item in the session,
    not just the ones under this conftest. Without it the unit suite would be
    marked integration too.

    Deliberately no skipif. A missing stack makes these ERROR, loudly. A test
    that quietly skips is a test that is not running.
    """
    for item in items:
        if _HERE in pathlib.Path(str(item.path)).parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def engine():  # type: ignore[no-untyped-def]
    """Engine against the compose database, with the schema applied once."""
    eng = create_engine(DB_URL or "", echo=False)
    with eng.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS ltree"))
    _migrate(eng)
    yield eng
    eng.dispose()


def _migrate(eng) -> None:  # type: ignore[no-untyped-def]
    """Apply migrations in-process, so no subprocess and no uv invocation."""
    from alembic.config import Config

    from alembic import command

    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", DB_URL or "")
    command.upgrade(cfg, "head")


@pytest.fixture
def session(engine) -> Generator[Session, None, None]:  # type: ignore[no-untyped-def]
    """A session on a clean corpus.

    The database is thrown away with the container after every run, so this
    truncate is only about isolating tests from each other within a run.
    """
    factory = sessionmaker(engine, expire_on_commit=False)
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE documents, chunks, nodes RESTART IDENTITY CASCADE"))
    with factory() as s:
        yield s


@pytest.fixture(autouse=True)
def point_litellm_at_the_sidecar(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider boundary only: base URL, model and a fake key.

    Prompts, the agent loop and every service stay exactly as in production --
    that is the whole point of testing here rather than mocking
    complete_structured().
    """
    import litellm

    monkeypatch.setenv("OPENAI_API_KEY", "agent-testkit-fake-key")
    monkeypatch.setattr(litellm, "api_base", LLM_URL, raising=False)


@pytest.fixture
def llm():  # type: ignore[no-untyped-def]
    """A real LLMService talking to the deterministic provider."""
    from openfable.services.llm_service import LLMService

    return LLMService(model="gpt-4.1-mini")


def make_chunk(position: int, content: str) -> object:
    """Minimal stand-in for a persisted Chunk, for tree-building tests."""
    from unittest.mock import MagicMock

    c = MagicMock()
    c.id = uuid.uuid4()
    c.content = content
    c.token_count = max(1, len(content) // 4)
    c.position = position
    return c
