"""Pydantic response models for tree construction LLM output.

These schemas are used with instructor + LiteLLM structured output to parse
the hierarchical tree that the LLM builds from document chunks.

Design notes:
- LLMLeafNode and LLMInternalNode use `type` as a discriminator field for
  Pydantic v2 discriminated union resolution in the `children` list.
- LLMInternalNode.model_rebuild() is called after both classes are defined
  to resolve the forward reference in children (Pydantic v2 Pitfall 3).
- field_validator messages are prescriptive so instructor can feed them back
  to the LLM in retry prompts and get corrected output.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator


class LLMLeafNode(BaseModel):
    """A leaf node referencing a single chunk by index."""

    type: Literal["leaf"]
    chunk_index: int  # 0-based index into the chunks list sent to LLM


class LLMInternalNode(BaseModel):
    """An internal node (root, section, or subsection) with an abstractive summary."""

    type: Literal["internal"]
    node_type: str  # "root" | "section" | "subsection"
    title: str
    summary: str
    children: list[Annotated["LLMInternalNode | LLMLeafNode", Field(discriminator="type")]]

    @field_validator("node_type", mode="after")
    @classmethod
    def validate_node_type(cls, v: str) -> str:
        if v not in ("root", "section", "subsection"):
            raise ValueError(
                f"node_type must be 'root', 'section', or 'subsection'. Got {v!r}. "
                "Use only root, section, or subsection for internal nodes."
            )
        return v

    @field_validator("title", mode="after")
    @classmethod
    def validate_title(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "title must not be empty -- provide a descriptive topic title."
            )
        return v

    @field_validator("summary", mode="after")
    @classmethod
    def validate_summary(cls, v: str) -> str:
        if not v.strip():
            raise ValueError(
                "summary must not be empty -- provide a concise abstractive summary."
            )
        return v


# Pydantic v2 requires model_rebuild() after both classes are defined to resolve
# the forward reference in LLMInternalNode.children (which references LLMInternalNode).
LLMInternalNode.model_rebuild()


class TreeBuildResponse(BaseModel):
    """Top-level response from the TreeBuild LLM call."""

    root: LLMInternalNode


class PartialTreeSummary(BaseModel):
    """Summary of a single partition's tree, used for TreeMerge input."""

    part_index: int
    root_title: str
    root_summary: str


class TreeMergeResponse(BaseModel):
    """Response from the TreeMerge LLM call: a unified root for all partitions."""

    merged_title: str
    merged_summary: str
