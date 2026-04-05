class OpenFableError(Exception):
    """Base exception for OpenFable."""

    pass


class ServiceUnavailableError(OpenFableError):
    """A required service (DB, TEI, LLM) is not reachable."""

    pass


class DocumentNotFoundError(OpenFableError):
    """Document with given ID does not exist."""

    pass


class ChunkingError(OpenFableError):
    """LLM-based chunking failed -- all retries exhausted or validation failed."""

    pass


class TreeConstructionError(OpenFableError):
    """LLM-based tree construction failed -- all retries exhausted or validation failed."""

    pass


class EmbeddingError(OpenFableError):
    """TEI embedding failed -- service unreachable or returned an error."""

    pass


class RetrievalError(OpenFableError):
    """Document-level retrieval failed -- LLMselect or vector path error."""

    pass
