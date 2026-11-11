
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from .claim import Claim


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"
    CONTRADICTORY = "contradictory"
    UNVERIFIED = "unverified"


class ClaimVerification(BaseModel):
    """
    Stores the verification outcome for an individual claim.
    """

    claim: Claim

    status: ClaimStatus

    supporting_evidence_ids: List[str] = Field(
        default_factory=list
    )

    explanation: str = Field(
        ...,
        description="Reason for the verification outcome."
    )


class VerificationResult(BaseModel):
    """
    Aggregated result of verifying all claims in a response.
    """

    total_claims: int = 0

    supported_count: int = 0
    partially_supported_count: int = 0
    unsupported_count: int = 0
    contradictory_count: int = 0
    unverified_count: int = 0

    claim_results: List[ClaimVerification] = Field(
        default_factory=list
    )
