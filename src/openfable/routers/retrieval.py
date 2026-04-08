import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from openfable.db import get_session
from openfable.exceptions import EmbeddingError, RetrievalError
from openfable.schemas.retrieval import QueryRequest, QueryResponse
from openfable.services.retrieval_service import RetrievalService, get_retrieval_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
def query_documents(
    body: QueryRequest,
    session: Session = Depends(get_session),
    service: RetrievalService = Depends(get_retrieval_service),
) -> QueryResponse:
    """Bi-path document-level retrieval with budget-adaptive routing.

    Combines LLM-guided document selection with vector top-K retrieval,
    fuses results, and routes based on token budget.
    """
    try:
        return service.query(session, body.query, body.token_budget)
    except RetrievalError as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingError as exc:
        logger.error("Embedding service unavailable: %s", exc)
        raise HTTPException(status_code=503, detail="Embedding service unavailable") from exc
