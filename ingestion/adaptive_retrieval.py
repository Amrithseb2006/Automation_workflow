
"""
Adaptive Retrieval Intelligence

Performs:
1. Initial dense retrieval
2. Retrieval quality assessment
3. Conditional query rewriting
4. Expanded retrieval
5. Deduplication and hybrid reranking
6. Failure handling and retrieval diagnostics

This module does not generate the final answer.
"""

import re
from dataclasses import dataclass
from typing import Callable


@dataclass
class RetrievalConfig:
    initial_top_k: int = 10
    expanded_top_k: int = 10
    final_top_k: int = 5

    # Initial heuristic thresholds.
    # Calibrate using a labelled evaluation dataset.
    good_score_threshold: float = 0.65
    minimum_results: int = 2

    dense_weight: float = 0.70
    lexical_weight: float = 0.30


def tokenize(text: str) -> set[str]:
    """Convert text into lowercase alphanumeric tokens."""
    return set(re.findall(r"\b[a-zA-Z0-9]+\b", text.lower()))


def lexical_similarity(query: str, text: str) -> float:
    """Calculate query-token coverage in a retrieved chunk."""
    query_tokens = tokenize(query)
    text_tokens = tokenize(text)

    if not query_tokens:
        return 0.0

    return len(query_tokens & text_tokens) / len(query_tokens)


def normalize_dense_score(score: float) -> float:
    """Keep the dense score within the range [0, 1]."""
    return max(0.0, min(1.0, float(score)))


def assess_retrieval_quality(
    chunks: list[dict],
    config: RetrievalConfig,
) -> dict:
    """
    Assess whether initial retrieval is sufficiently relevant.

    This is a heuristic assessment, not a calibrated metric.
    """
    if not chunks:
        return {
            "quality": "poor",
            "top_score": 0.0,
            "result_count": 0,
            "should_rewrite": True,
        }

    scores = [
        normalize_dense_score(chunk["score"])
        for chunk in chunks
    ]

    top_score = max(scores)

    good = (
        len(chunks) >= config.minimum_results
        and top_score >= config.good_score_threshold
    )

    return {
        "quality": "good" if good else "poor",
        "top_score": round(top_score, 4),
        "result_count": len(chunks),
        "should_rewrite": not good,
    }


def deduplicate_chunks(chunks: list[dict]) -> list[dict]:
    """Remove duplicate chunks using source, page and text."""
    unique = {}

    for chunk in chunks:
        key = (
            chunk.get("source", ""),
            str(chunk.get("page", "")),
            chunk.get("text", "").strip().lower(),
        )

        if key not in unique:
            unique[key] = chunk
        elif chunk["score"] > unique[key]["score"]:
            unique[key] = chunk

    return list(unique.values())


def rerank_chunks(
    query: str,
    chunks: list[dict],
    config: RetrievalConfig,
) -> list[dict]:
    """
    Rerank using dense relevance and lexical query coverage.

    Original Pinecone scores are retained as dense_score.
    """
    reranked = []

    for chunk in chunks:
        # Avoid modifying the original candidate dictionaries.
        ranked_chunk = dict(chunk)

        dense = normalize_dense_score(chunk["score"])
        lexical = lexical_similarity(query, chunk["text"])

        ranked_chunk["dense_score"] = round(dense, 4)
        ranked_chunk["lexical_score"] = round(lexical, 4)

        ranked_chunk["score"] = round(
            config.dense_weight * dense
            + config.lexical_weight * lexical,
            4,
        )

        reranked.append(ranked_chunk)

    return sorted(
        reranked,
        key=lambda chunk: chunk["score"],
        reverse=True,
    )


class AdaptiveRetriever:
    """
    Coordinates Pinecone retrieval, quality assessment,
    optional query rewriting, merging and reranking.

    embed_query:
        Function accepting a query and returning a Gemini
        embedding vector.

    rewrite_query:
        Function accepting a query and returning a rewritten
        query.
    """

    def __init__(
        self,
        index,
        embed_query: Callable,
        rewrite_query: Callable,
        config: RetrievalConfig | None = None,
    ):
        self.index = index
        self.embed_query = embed_query
        self.rewrite_query = rewrite_query
        self.config = config or RetrievalConfig()

    def _search(
        self,
        query: str,
        namespace: str,
        top_k: int,
    ) -> list[dict]:
        """Generate an embedding and retrieve chunks from Pinecone."""

        vector = self.embed_query(query)

        results = self.index.query(
            vector=vector,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
        )

        chunks = []

        for match in results.matches:
            metadata = match.metadata or {}

            chunks.append({
                "text": metadata.get("text", ""),
                "source": metadata.get("source", ""),
                "page": metadata.get("page", ""),
                "section": metadata.get("section", ""),
                "score": float(match.score or 0.0),
            })

        return chunks

    def retrieve(
        self,
        query: str,
        namespace: str,
        top_k: int | None = None,
    ) -> dict:
        """Execute adaptive retrieval with controlled failure handling."""

        initial_k = (
            top_k
            if top_k is not None and top_k > 0
            else self.config.initial_top_k
        )

        # Stage 1: Initial retrieval
        try:
            initial_chunks = self._search(
                query,
                namespace,
                initial_k,
            )

        except Exception as exc:
            # Do not attempt rewriting if initial retrieval fails.
            return {
                "chunks": [],
                "retrieval": {
                    "initial_result_count": 0,
                    "final_result_count": 0,
                    "retrieval_quality": "unavailable",
                    "top_initial_score": 0.0,
                    "query_rewritten": False,
                    "rewritten_query": None,
                    "strategy": "retrieval_unavailable",
                    "expanded_retrieval_improved": None,
                    "error": type(exc).__name__,
                },
            }

        # Stage 2: Assess initial retrieval quality
        assessment = assess_retrieval_quality(
            initial_chunks,
            self.config,
        )

        all_chunks = list(initial_chunks)
        rewritten_query = None
        strategy = "initial_retrieval"
        expanded_improved = None

        # Stage 3: Conditional query rewriting
        if assessment["should_rewrite"]:
            try:
                candidate_query = self.rewrite_query(query)

                if (
                    isinstance(candidate_query, str)
                    and candidate_query.strip()
                ):
                    rewritten_query = candidate_query.strip()

                    expanded_chunks = self._search(
                        rewritten_query,
                        namespace,
                        self.config.expanded_top_k,
                    )

                    # Compare best raw dense scores.
                    # This is a heuristic diagnostic only:
                    # scores from different queries are not
                    # calibrated evidence of relevance improvement.
                    initial_best = max(
                        (
                            chunk["score"]
                            for chunk in initial_chunks
                        ),
                        default=0.0,
                    )

                    expanded_best = max(
                        (
                            chunk["score"]
                            for chunk in expanded_chunks
                        ),
                        default=0.0,
                    )

                    expanded_improved = (
                        expanded_best > initial_best
                    )

                    # Preserve both candidate sets.
                    # Final reranking uses the original query.
                    all_chunks.extend(expanded_chunks)

                    strategy = "adaptive_retrieval"

            except Exception:
                # Preserve initial results if rewriting or
                # expanded retrieval fails.
                strategy = "initial_retrieval_fallback"

        # Stage 4: Deduplication
        unique_chunks = deduplicate_chunks(all_chunks)

        # Stage 5: Hybrid reranking
        final_chunks = rerank_chunks(
            query,
            unique_chunks,
            self.config,
        )[:self.config.final_top_k]

        # Stage 6: Return chunks and diagnostics
        return {
            "chunks": final_chunks,
            "retrieval": {
                "initial_result_count": len(initial_chunks),
                "final_result_count": len(final_chunks),
                "retrieval_quality": assessment["quality"],
                "top_initial_score": assessment["top_score"],
                "query_rewritten": rewritten_query is not None,
                "rewritten_query": rewritten_query,
                "strategy": strategy,
                "expanded_retrieval_improved": expanded_improved,
            },
        }
