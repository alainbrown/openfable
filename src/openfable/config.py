import os
from typing import Any

import litellm as _litellm
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENFABLE_",
        case_sensitive=False,
    )

    # Database
    database_url: str = "postgresql://openfable:openfable@db:5432/openfable"

    # LLM
    litellm_model: str = "gpt-5.4"
    litellm_api_key: str = ""
    litellm_base_url: str = ""

    # Embeddings
    embedding_url: str = "https://api.openai.com"
    embedding_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = 64

    # App
    debug: bool = False

    # Retrieval
    retrieval_top_k: int = 10
    retrieval_llmselect_depth: int = 2

    def model_post_init(self, __context: Any) -> None:
        """Wire OPENFABLE_ env vars to LiteLLM runtime.

        LiteLLM reads provider-specific env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY)
        directly. This bridges the OPENFABLE_ namespace to provider vars.
        Uses setdefault so explicit provider vars take precedence.
        """
        if self.litellm_api_key:
            os.environ.setdefault("ANTHROPIC_API_KEY", self.litellm_api_key)
            os.environ.setdefault("OPENAI_API_KEY", self.litellm_api_key)
            if not self.embedding_api_key:
                self.embedding_api_key = self.litellm_api_key
        if self.litellm_base_url:
            _litellm.api_base = self.litellm_base_url
        if self.debug:
            os.environ.setdefault("LITELLM_LOG", "DEBUG")


settings = Settings()
