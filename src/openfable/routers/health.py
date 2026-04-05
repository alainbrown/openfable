import time

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from openfable.db import get_session
from openfable.schemas.health import ComponentStatus, HealthResponse
from openfable.services.llm_service import LLMService, get_llm_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check(
    db: Session = Depends(get_session),
    llm: LLMService = Depends(get_llm_service),
) -> JSONResponse:
    """Check health of database and LLM.

    Returns per-component status with latency. Any component failure
    results in HTTP 503 (per D-06: binary healthy/unhealthy, no degraded state).
    """
    components: dict[str, ComponentStatus] = {}
    all_healthy = True

    # Database probe: SELECT 1 (per D-05)
    t = time.monotonic()
    try:
        db.execute(text("SELECT 1"))
        components["database"] = ComponentStatus(
            status="healthy",
            latency_ms=round((time.monotonic() - t) * 1000, 1),
        )
    except Exception as e:
        components["database"] = ComponentStatus(
            status="unhealthy",
            error=str(e),
        )
        all_healthy = False

    # LLM probe: tiny completion call (per D-05, Pitfall 5: max_tokens=1)
    t = time.monotonic()
    try:
        llm.health_probe()
        components["llm"] = ComponentStatus(
            status="healthy",
            latency_ms=round((time.monotonic() - t) * 1000, 1),
        )
    except Exception as e:
        components["llm"] = ComponentStatus(
            status="unhealthy",
            error=str(e),
        )
        all_healthy = False

    # D-06: binary — any failure = 503
    status_code = 200 if all_healthy else 503
    response = HealthResponse(
        status="healthy" if all_healthy else "unhealthy",
        components=components,
    )
    return JSONResponse(status_code=status_code, content=response.model_dump())
