
import unittest

from pydantic import ValidationError

from ingestion.response_verification.schemas.evidence import EvidenceChunk
from ingestion.response_verification.schemas.request import VerificationRequest


class TestVerificationSchemas(unittest.TestCase):

    def test_valid_evidence_chunk(self):
        evidence = EvidenceChunk(
            evidence_id="E1",
            text="Students must complete the admission application.",
            source="admission_policy.pdf",
            page="4",
            section="Admission Process",
            retrieval_score=0.85,
        )

        self.assertEqual(evidence.evidence_id, "E1")
        self.assertEqual(evidence.source, "admission_policy.pdf")
        self.assertEqual(evidence.retrieval_score, 0.85)

    def test_invalid_empty_evidence_text(self):
        with self.assertRaises(ValidationError):
            EvidenceChunk(
                evidence_id="E1",
                text="",
                source="admission_policy.pdf",
            )

    def test_invalid_retrieval_score(self):
        with self.assertRaises(ValidationError):
            EvidenceChunk(
                evidence_id="E1",
                text="Admission information.",
                source="admission_policy.pdf",
                retrieval_score=1.5,
            )

    def test_valid_verification_request(self):
        evidence = EvidenceChunk(
            evidence_id="E1",
            text="The library opens at 9 AM.",
            source="library_policy.pdf",
            page="2",
        )

        request = VerificationRequest(
            query="When does the library open?",
            generated_response="The library opens at 9 AM.",
            retrieved_chunks=[evidence],
        )

        self.assertEqual(len(request.retrieved_chunks), 1)
        self.assertEqual(request.query, "When does the library open?")

    def test_verification_request_without_evidence(self):
        request = VerificationRequest(
            query="What are the library timings?",
            generated_response="The timings are not available.",
        )

        self.assertEqual(request.retrieved_chunks, [])

    def test_invalid_empty_generated_response(self):
        with self.assertRaises(ValidationError):
            VerificationRequest(
                query="What are the library timings?",
                generated_response="",
            )
    def test_evidence_accepts_integer_page(self):
        evidence = EvidenceChunk(
            evidence_id="E2",
            text="Admission requirements.",
            source="admission_policy.pdf",
            page=4,
        )

        self.assertEqual(evidence.page, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
