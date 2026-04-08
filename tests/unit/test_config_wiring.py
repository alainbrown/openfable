"""Unit tests for Settings.model_post_init env var wiring to LiteLLM runtime.

Tests verify:
- OPENFABLE_LITELLM_API_KEY is forwarded to ANTHROPIC_API_KEY and OPENAI_API_KEY via setdefault
- OPENFABLE_LITELLM_BASE_URL is forwarded to litellm.api_base
- Empty values are no-ops (do not pollute os.environ)
- Existing provider-specific keys are not overwritten (setdefault semantics)
"""

import os

import litellm
import pytest

from openfable.config import Settings


def test_litellm_api_key_wired_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENFABLE_LITELLM_API_KEY is forwarded to ANTHROPIC_API_KEY via setdefault."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENFABLE_LITELLM_API_KEY", raising=False)

    Settings(litellm_api_key="sk-test-123")
    assert os.environ.get("ANTHROPIC_API_KEY") == "sk-test-123"


def test_litellm_api_key_wired_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENFABLE_LITELLM_API_KEY is forwarded to OPENAI_API_KEY via setdefault."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENFABLE_LITELLM_API_KEY", raising=False)

    Settings(litellm_api_key="sk-test-123")
    assert os.environ.get("OPENAI_API_KEY") == "sk-test-123"


def test_empty_api_key_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """When OPENFABLE_LITELLM_API_KEY is empty, no ANTHROPIC/OPENAI keys are set."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENFABLE_LITELLM_API_KEY", raising=False)

    Settings(litellm_api_key="")
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ


def test_litellm_base_url_wired(monkeypatch: pytest.MonkeyPatch) -> None:
    """OPENFABLE_LITELLM_BASE_URL is forwarded to litellm.api_base."""
    monkeypatch.delenv("OPENFABLE_LITELLM_BASE_URL", raising=False)
    original_api_base = litellm.api_base

    try:
        Settings(litellm_base_url="http://localhost:11434")
        assert litellm.api_base == "http://localhost:11434"
    finally:
        litellm.api_base = original_api_base


def test_empty_base_url_no_litellm_change(monkeypatch: pytest.MonkeyPatch) -> None:
    """When OPENFABLE_LITELLM_BASE_URL is empty, litellm.api_base is not modified."""
    monkeypatch.delenv("OPENFABLE_LITELLM_BASE_URL", raising=False)
    original_api_base = litellm.api_base

    try:
        Settings(litellm_base_url="")
        assert litellm.api_base == original_api_base
    finally:
        litellm.api_base = original_api_base


def test_existing_provider_key_not_overwritten(monkeypatch: pytest.MonkeyPatch) -> None:
    """os.environ.setdefault does NOT overwrite an existing ANTHROPIC_API_KEY."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "existing-anthropic-key")
    monkeypatch.delenv("OPENFABLE_LITELLM_API_KEY", raising=False)

    Settings(litellm_api_key="sk-new-key")
    # setdefault should not overwrite the existing key
    assert os.environ.get("ANTHROPIC_API_KEY") == "existing-anthropic-key"
