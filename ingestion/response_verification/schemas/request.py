
from typing import List

from pydantic import BaseModel, Field

from .evidence import EvidenceChunk


class VerificationRequest(BaseModel):
    """
    Input received by the response verification module.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="Original user query."
    )

    generated_response: str = Field(
        ...,
        min_length=1,
        description="AI-generated response requiring verification."
    )

    retrieved_chunks: List[EvidenceChunk] = Field(
        default_factory=list,
        description="Evidence chunks retrieved by the RAG pipeline."
    )
