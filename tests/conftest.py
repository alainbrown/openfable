from unittest.mock import MagicMock

import pytest

from openfable.services.llm_service import LLMService


@pytest.fixture
def mock_llm():
    """Mock LLMService with sync complete_structured method."""
    llm = MagicMock(spec=LLMService)
    llm.complete_structured = MagicMock()
    llm.model = "test-model"
    return llm
