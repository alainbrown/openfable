from pydantic import BaseModel


class ChunkResult(BaseModel):
    chunk_text: str
    start_idx: int
    end_idx: int


class ChunkingResponse(BaseModel):
    chunks: list[ChunkResult]
