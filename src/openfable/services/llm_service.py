import logging
from typing import Any, TypeVar, cast

import instructor
import litellm
from pydantic import BaseModel

from openfable.config import settings

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger("openfable.llm")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


def _log_llm_call(kwargs: dict[str, Any], response: Any, *args: Any, **kw: Any) -> None:
    """LiteLLM success callback — logs the actual messages sent and response received."""
    logger.info("LLM actual request: %s", kwargs.get("messages"))
    if hasattr(response, "choices") and response.choices:
        logger.info("LLM actual response: %s", response.choices[0].message.content)


if settings.debug:
    litellm.success_callback = [_log_llm_call]

# Patch LiteLLM's sync completion with instructor for structured output
_patched_client = instructor.from_litellm(litellm.completion, mode=instructor.Mode.JSON)


class LLMService:
    """Wrapper around LiteLLM + instructor for all LLM calls.

    All structured output calls go through complete_structured().
    """

    def __init__(self, model: str | None = None):
        self.model = model or settings.litellm_model
        self.client = _patched_client

    def complete_structured(
        self,
        response_model: type[T],
        messages: list[dict[str, str]],
        max_retries: int = 3,
        temperature: float = 0,
    ) -> T:
        """Call LLM and parse response into a Pydantic model with retry on validation failure."""
        return cast(
            T,
            self.client.chat.completions.create(
                model=self.model,
                response_model=response_model,
                max_retries=max_retries,
                messages=messages,
                temperature=temperature,
            ),
        )

    def health_probe(self) -> None:
        """Minimal LLM call for health check. Raises on failure.

        Uses max_tokens=1 and single-word prompt to minimize cost (Pitfall 5).
        """
        litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
        )


def get_llm_service() -> LLMService:
    return LLMService()
