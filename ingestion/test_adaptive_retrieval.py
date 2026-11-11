
import unittest
from unittest.mock import Mock

from adaptive_retrieval import (
    AdaptiveRetriever,
    RetrievalConfig,
    assess_retrieval_quality,
    deduplicate_chunks,
    rerank_chunks,
)


def make_chunk(text, score, source="test.pdf", page=1):
    return {
        "text": text,
        "source": source,
        "page": page,
        "section": "Test",
        "score": score,
    }


class TestRetrievalQuality(unittest.TestCase):

    def setUp(self):
        self.config = RetrievalConfig()

    def test_good_retrieval(self):
        chunks = [
            make_chunk("admission process students", 0.85),
            make_chunk("student admission requirements", 0.70),
        ]

        result = assess_retrieval_quality(chunks, self.config)

        self.assertEqual(result["quality"], "good")
        self.assertFalse(result["should_rewrite"])

    def test_poor_retrieval(self):
        chunks = [
            make_chunk("unrelated content", 0.40),
            make_chunk("other information", 0.35),
        ]

        result = assess_retrieval_quality(chunks, self.config)

        self.assertEqual(result["quality"], "poor")
        self.assertTrue(result["should_rewrite"])

    def test_empty_retrieval(self):
        result = assess_retrieval_quality([], self.config)

        self.assertEqual(result["quality"], "poor")
        self.assertTrue(result["should_rewrite"])


class TestChunkProcessing(unittest.TestCase):

    def test_deduplication(self):
        chunks = [
            make_chunk("Admission process", 0.70),
            make_chunk("Admission process", 0.90),
            make_chunk("Scholarship details", 0.60),
        ]

        result = deduplicate_chunks(chunks)

        self.assertEqual(len(result), 2)
        self.assertEqual(
            next(c["score"] for c in result
                 if c["text"] == "Admission process"),
            0.90,
        )

    def test_reranking(self):
        chunks = [
            make_chunk("Hostel facilities and transport", 0.90),
            make_chunk("Admission process for students", 0.60),
        ]

        result = rerank_chunks(
            "admission process students",
            chunks,
            RetrievalConfig(),
        )

        self.assertEqual(
            result[0]["text"],
            "Admission process for students",
        )
        self.assertIn("dense_score", result[0])
        self.assertIn("lexical_score", result[0])


class TestAdaptiveRetriever(unittest.TestCase):

    def setUp(self):
        self.config = RetrievalConfig(
            initial_top_k=5,
            expanded_top_k=5,
            final_top_k=5,
        )

        self.embed = Mock(return_value=[0.1, 0.2, 0.3])
        self.rewrite = Mock(return_value="Rewritten admission query")
        self.index = Mock()

    def set_matches(self, chunks):
        matches = []

        for chunk in chunks:
            match = Mock()
            match.metadata = {
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "section": chunk["section"],
            }
            match.score = chunk["score"]
            matches.append(match)

        response = Mock()
        response.matches = matches

        return response

    def test_strong_retrieval_skips_rewriting(self):
        self.index.query.return_value = self.set_matches([
            make_chunk("admission process students", 0.90),
            make_chunk("admission eligibility", 0.80),
        ])

        retriever = AdaptiveRetriever(
            self.index,
            self.embed,
            self.rewrite,
            self.config,
        )

        result = retriever.retrieve(
            "admission process",
            "admissions",
        )

        self.assertEqual(
            result["retrieval"]["strategy"],
            "initial_retrieval",
        )
        self.assertFalse(result["retrieval"]["query_rewritten"])
        self.rewrite.assert_not_called()
        self.assertEqual(self.index.query.call_count, 1)

    def test_weak_retrieval_triggers_rewriting(self):
        self.index.query.side_effect = [
            self.set_matches([
                make_chunk("unrelated content", 0.40),
                make_chunk("other information", 0.30),
            ]),
            self.set_matches([
                make_chunk("admission process students", 0.80),
                make_chunk("admission eligibility", 0.70),
            ]),
        ]

        retriever = AdaptiveRetriever(
            self.index,
            self.embed,
            self.rewrite,
            self.config,
        )

        result = retriever.retrieve(
            "admission process",
            "admissions",
        )

        self.assertEqual(
            result["retrieval"]["strategy"],
            "adaptive_retrieval",
        )
        self.assertTrue(result["retrieval"]["query_rewritten"])
        self.assertEqual(self.index.query.call_count, 2)
        self.rewrite.assert_called_once()

    def test_empty_retrieval(self):
        self.index.query.side_effect = [
            self.set_matches([]),
            self.set_matches([]),
        ]

        retriever = AdaptiveRetriever(
            self.index,
            self.embed,
            self.rewrite,
            self.config,
        )

        result = retriever.retrieve(
            "unknown query",
            "admissions",
        )

        self.assertEqual(
            result["retrieval"]["retrieval_quality"],
            "poor",
        )
        self.assertEqual(
            result["retrieval"]["final_result_count"],
            0,
        )
        self.assertTrue(result["retrieval"]["query_rewritten"])

    def test_rewriting_failure_preserves_initial_results(self):
        self.index.query.return_value = self.set_matches([
            make_chunk("unrelated information", 0.40),
            make_chunk("other information", 0.30),
        ])

        self.rewrite.side_effect = Exception("Groq unavailable")

        retriever = AdaptiveRetriever(
            self.index,
            self.embed,
            self.rewrite,
            self.config,
        )

        result = retriever.retrieve(
            "unknown query",
            "admissions",
        )

        self.assertEqual(
            result["retrieval"]["strategy"],
            "initial_retrieval_fallback",
        )
        self.assertEqual(
            result["retrieval"]["final_result_count"],
            2,
        )
        self.assertFalse(result["retrieval"]["query_rewritten"])

    def test_duplicate_results_are_removed_after_retrieval(self):
        self.index.query.side_effect = [
            self.set_matches([
                make_chunk("admission process", 0.40),
                make_chunk("scholarship details", 0.30),
            ]),
            self.set_matches([
                make_chunk("admission process", 0.90),
                make_chunk("hostel facilities", 0.80),
            ]),
        ]

        retriever = AdaptiveRetriever(
            self.index,
            self.embed,
            self.rewrite,
            self.config,
        )

        result = retriever.retrieve(
            "admission process",
            "admissions",
        )

        texts = [chunk["text"] for chunk in result["chunks"]]

        self.assertEqual(len(texts), len(set(texts)))

    # NEW TEST 1: Initial Pinecone failure
    def test_initial_pinecone_failure(self):
        self.index.query.side_effect = Exception(
            "Pinecone connection unavailable"
        )

        retriever = AdaptiveRetriever(
            self.index,
            self.embed,
            self.rewrite,
            self.config,
        )

        result = retriever.retrieve(
            "admission process",
            "admissions",
        )

        self.assertEqual(result["chunks"], [])
        self.assertEqual(
            result["retrieval"]["strategy"],
            "retrieval_unavailable",
        )
        self.assertEqual(
            result["retrieval"]["retrieval_quality"],
            "unavailable",
        )
        self.assertFalse(result["retrieval"]["query_rewritten"])

        self.rewrite.assert_not_called()
        self.assertEqual(self.index.query.call_count, 1)

    # NEW TEST 2: Expanded retrieval does not improve results
    def test_expanded_retrieval_no_improvement(self):
        self.index.query.side_effect = [
            self.set_matches([
                make_chunk(
                    "admission process requirements",
                    0.40,
                ),
                make_chunk(
                    "student admission eligibility",
                    0.30,
                ),
            ]),
            self.set_matches([
                make_chunk(
                    "unrelated expanded content",
                    0.35,
                ),
                make_chunk(
                    "other unrelated information",
                    0.20,
                ),
            ]),
        ]

        retriever = AdaptiveRetriever(
            self.index,
            self.embed,
            self.rewrite,
            self.config,
        )

        result = retriever.retrieve(
            "admission process",
            "admissions",
        )

        self.assertEqual(
            result["retrieval"]["strategy"],
            "adaptive_retrieval",
        )

        self.assertTrue(
            result["retrieval"]["query_rewritten"]
        )

        self.assertFalse(
            result["retrieval"]["expanded_retrieval_improved"]
        )

        # Initial relevant results should remain available.
        self.assertIn(
            "admission process requirements",
            [chunk["text"] for chunk in result["chunks"]],
        )

        self.assertEqual(self.index.query.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
