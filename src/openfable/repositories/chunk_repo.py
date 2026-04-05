import uuid

from sqlalchemy.orm import Session

from openfable.models.chunk import Chunk
from openfable.repositories.document_repo import count_tokens
from openfable.schemas.chunking import ChunkResult


class ChunkRepository:
    """Repository for persisting chunk records to the database."""

    def insert_chunks(
        self,
        session: Session,
        document_id: uuid.UUID,
        chunks: list[ChunkResult],
    ) -> list[Chunk]:
        """Persist a list of ChunkResult objects as Chunk ORM rows.

        Note: count_tokens() is called here as the single authoritative place
        where token_count is computed for persistence. The Pydantic validator's
        bounds check is for instructor retry feedback, not storage.
        """
        chunk_models = [
            Chunk(
                document_id=document_id,
                content=c.chunk_text,
                token_count=count_tokens(c.chunk_text),
                position=i,
                start_idx=c.start_idx,
                end_idx=c.end_idx,
            )
            for i, c in enumerate(chunks)
        ]
        session.add_all(chunk_models)
        session.flush()
        return chunk_models


def get_chunk_repo() -> ChunkRepository:
    """Factory function for ChunkRepository."""
    return ChunkRepository()
