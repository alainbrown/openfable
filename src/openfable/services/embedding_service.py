import logging
import uuid

import httpx

from openfable.config import settings
from openfable.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


def _build_embedding_text(
    node_type: str,
    toc_path: str | None,
    summary: str | None,
    content: str | None,
) -> str:
    """Construct embedding text per FABLE multi-granularity spec.

    D-01: Internal nodes (root, section, subsection) -> "{toc_path}: {summary}"
    D-02: Leaf nodes -> raw content
    """
    if node_type == "leaf":
        text = content or ""
    else:
        toc = toc_path or ""
        s = summary or ""
        text = f"{toc}: {s}"
    if not text or text == ": ":
        logger.warning(
            "Empty embedding text for %s node (toc_path=%r, summary=%r, content=%r)",
            node_type,
            toc_path,
            summary,
            content,
        )
    return text


class EmbeddingService:
    """Sync client for OpenAI-compatible /v1/embeddings endpoint.

    Compatible with HuggingFace TEI, OpenAI, vLLM, Ollama, and any provider
    that implements the OpenAI embeddings API format. The embedding_url config field
    serves as the base URL for the embedding provider.

    When OPENFABLE_EMBEDDING_API_KEY is set, requests include an Authorization
    Bearer header. When empty, no Authorization header is sent (backward
    compatible with TEI which does not require authentication).
    """

    def __init__(self, embedding_url: str | None = None):
        self.embedding_url = embedding_url or settings.embedding_url

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via /v1/embeddings (OpenAI-compatible format).

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors (each 1024-dim for BGE-M3).
        """
        headers: dict[str, str] = {}
        if settings.embedding_api_key:
            headers = {"Authorization": f"Bearer {settings.embedding_api_key}"}

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                f"{self.embedding_url}/v1/embeddings",
                json={
                    "input": texts,
                    "model": settings.embedding_model,
                    "dimensions": settings.embedding_dimensions,
                },
                headers=headers,
            )
            resp.raise_for_status()
            return [item["embedding"] for item in resp.json()["data"]]

    def embed_nodes(
        self,
        node_texts: list[tuple[uuid.UUID, str]],
        batch_size: int = 64,
    ) -> list[tuple[uuid.UUID, list[float]]]:
        """Embed nodes in batches of batch_size via TEI /embed.

        Args:
            node_texts: List of (node_id, text) pairs.
            batch_size: Max texts per TEI call (D-03: default 64, D-04: configurable).

        Returns:
            List of (node_id, embedding_vector) pairs in input order.

        Raises:
            EmbeddingError: If TEI returns an error (D-07: fail immediately, no retry).
        """
        results: list[tuple[uuid.UUID, list[float]]] = []
        for i in range(0, len(node_texts), batch_size):
            batch = node_texts[i : i + batch_size]
            texts = [text for _, text in batch]
            try:
                vectors = self.embed_batch(texts)
            except httpx.HTTPStatusError as exc:
                raise EmbeddingError(
                    f"TEI embedding failed (batch {i // batch_size + 1}): "
                    f"HTTP {exc.response.status_code} - {exc.response.text[:200]}"
                ) from exc
            except httpx.ConnectError as exc:
                raise EmbeddingError(f"TEI unreachable at {self.embedding_url}: {exc}") from exc
            results.extend(zip([nid for nid, _ in batch], vectors))
        return results


def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
