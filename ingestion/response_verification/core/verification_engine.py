
from typing import List

from ..schemas.verification import (
    ClaimStatus,
    ClaimVerification,
    VerificationResult,
)


def aggregate_verification(
    claim_results: List[ClaimVerification],
) -> VerificationResult:
    """
    Aggregate individual claim verification results.

    This function calculates the number of claims in each
    verification category and returns a structured summary.

    It does not independently verify factual accuracy.
    """

    counts = {
        status: 0
        for status in ClaimStatus
    }

    for result in claim_results:
        counts[result.status] += 1

    return VerificationResult(
        total_claims=len(claim_results),
        supported_count=counts[ClaimStatus.SUPPORTED],
        partially_supported_count=counts[
            ClaimStatus.PARTIALLY_SUPPORTED
        ],
        unsupported_count=counts[ClaimStatus.UNSUPPORTED],
        contradictory_count=counts[ClaimStatus.CONTRADICTORY],
        unverified_count=counts[ClaimStatus.UNVERIFIED],
        claim_results=claim_results,
    )