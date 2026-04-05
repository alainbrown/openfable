from pydantic import BaseModel


class ComponentStatus(BaseModel):
    status: str  # "healthy" or "unhealthy"
    latency_ms: float | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str  # "healthy" or "unhealthy"
    components: dict[str, ComponentStatus]
