
from typing import Optional

from pydantic import BaseModel, Field


class EvidenceChunk(BaseModel):
    """
    Represents a document chunk retrieved by the RAG system.

    The original retrieved text and its source metadata are
    preserved for traceability during response verification.
    """

    evidence_id: str = Field(
        ...,
        description="Unique identifier for this evidence chunk."
    )

    text: str = Field(
        ...,
        min_length=1,
        description="Original text retrieved from the knowledge base."
    )

    source: str = Field(
        ...,
        description="Original document or file name."
    )

    page: Optional[str | int] = Field(
        default=None,
        description="Page number or page reference."
    )

    section: Optional[str] = Field(
        default=None,
        description="Document section, if available."
    )

    retrieval_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Retrieval relevance score. This is not a factual support score."
    )
