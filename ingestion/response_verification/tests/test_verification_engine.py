
import unittest

from ingestion.response_verification.schemas.claim import Claim
from ingestion.response_verification.schemas.verification import (
    ClaimStatus,
    ClaimVerification,
)
from ingestion.response_verification.core.verification_engine import (
    aggregate_verification,
)


class TestVerificationEngine(unittest.TestCase):

    def test_aggregate_verification_counts(self):

        results = [
            ClaimVerification(
                claim=Claim(
                    claim_id="C1",
                    text="The library opens at 9 AM."
                ),
                status=ClaimStatus.SUPPORTED,
                supporting_evidence_ids=["E1"],
                explanation="Supported by the retrieved document."
            ),
            ClaimVerification(
                claim=Claim(
                    claim_id="C2",
                    text="The library closes at 8 PM."
                ),
                status=ClaimStatus.PARTIALLY_SUPPORTED,
                supporting_evidence_ids=["E2"],
                explanation="Only part of the claim is supported."
            ),
            ClaimVerification(
                claim=Claim(
                    claim_id="C3",
                    text="The library is open on Sundays."
                ),
                status=ClaimStatus.UNSUPPORTED,
                explanation="No supporting evidence was identified."
            ),
            ClaimVerification(
                claim=Claim(
                    claim_id="C4",
                    text="The library has 500 seats."
                ),
                status=ClaimStatus.CONTRADICTORY,
                supporting_evidence_ids=["E3"],
                explanation="Retrieved evidence states a different capacity."
            ),
            ClaimVerification(
                claim=Claim(
                    claim_id="C5",
                    text="The library has a reading room."
                ),
                status=ClaimStatus.UNVERIFIED,
                explanation="The claim has not been assessed."
            ),
        ]

        summary = aggregate_verification(results)

        self.assertEqual(summary.total_claims, 5)
        self.assertEqual(summary.supported_count, 1)
        self.assertEqual(summary.partially_supported_count, 1)
        self.assertEqual(summary.unsupported_count, 1)
        self.assertEqual(summary.contradictory_count, 1)
        self.assertEqual(summary.unverified_count, 1)


if __name__ == "__main__":
    unittest.main()
