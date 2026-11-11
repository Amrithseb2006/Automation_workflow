
# Phase 1 – Adaptive Retrieval Intelligence

## 1. Project Overview

This phase implements the Adaptive Retrieval Intelligence component of the project:

**Adaptive Evidence-Grounded Email Automation with Retrieval Quality Assessment and Response Verification**

The objective is to improve the quality of document retrieval by evaluating initial search results and adapting the retrieval process when the results are considered insufficient.

Instead of relying entirely on a single dense retrieval operation, the system evaluates retrieval quality, conditionally rewrites the query, performs expanded retrieval, removes duplicates and reranks the retrieved chunks.

## 2. System Architecture

The implemented retrieval pipeline follows this workflow:

User Query
    |
    v
Initial Dense Retrieval
    |
    v
Retrieval Quality Assessment
    |
    +---- Good Retrieval ----> Continue with Initial Results
    |
    +---- Poor Retrieval ----> Query Rewriting
                                  |
                                  v
                           Expanded Retrieval
                                  |
                                  v
                         Candidate Result Merging
                                  |
                                  v
                            Deduplication
                                  |
                                  v
                         Hybrid Reranking
                                  |
                                  v
                         Final Retrieved Chunks
                                  |
                                  v
                       Evidence Context Generation
                                  |
                                  v
                         Grounded Answer Generation

## 3. Technology Stack

| Component | Technology |
|---|---|
| Backend API | FastAPI |
| Programming Language | Python |
| Vector Database | Pinecone |
| Embedding Model | Gemini gemini-embedding-001 |
| Query Rewriting | Groq |
| Language Model | qwen/qwen3.8-27b |
| Document Processing | LangChain Text Splitters |
| API Testing | FastAPI Swagger UI |
| Unit Testing | Python unittest |
| Configuration | Python dotenv |

## 4. Adaptive Retrieval Components

### 4.1 Initial Dense Retrieval

The original user query is converted into an embedding using Google's Gemini embedding model.

The embedding is submitted to Pinecone to retrieve the most relevant document chunks from the specified namespace.

The initial retrieval results include:
- Retrieved text
- Source document
- Page number
- Section information
- Dense similarity score

### 4.2 Retrieval Quality Assessment

The retrieved results are evaluated using an initial heuristic assessment.

The assessment considers:
- Number of retrieved results.
- Highest dense similarity score.
- Configured minimum result count.
- Configured similarity threshold.

Retrieval is classified as good when the minimum result count and similarity threshold requirements are satisfied.

Otherwise, the retrieval is classified as poor and query rewriting is triggered.

### 4.3 Conditional Query Rewriting

When initial retrieval quality is insufficient, the original query is passed to the Groq language model.

The rewriting prompt instructs the model to:
- Preserve the original intent.
- Retain important entities.
- Improve query clarity and specificity.
- Return only the rewritten query.
- Avoid answering the original question.

Query rewriting is conditional and is not performed when initial retrieval quality is considered sufficient.

### 4.4 Expanded Retrieval

The rewritten query is converted into an embedding and submitted to Pinecone for an additional retrieval operation.

The expanded retrieval results are combined with the initial candidate results.

If rewriting or expanded retrieval fails, the system preserves the initial retrieval results.

### 4.5 Deduplication

Duplicate chunks are identified using:
- Source document.
- Page number.
- Normalized text.

Only unique candidates are retained, preventing repeated chunks from unnecessarily occupying the final retrieval results.

### 4.6 Hybrid Reranking

The candidate chunks are reranked using a combination of dense similarity and lexical similarity.

The implemented scoring formula is:

Final Score = (0.70 × Dense Score) + (0.30 × Lexical Score)

Dense similarity measures semantic relevance.

Lexical similarity measures the proportion of query tokens present in the retrieved chunk.

The resulting scores are used to sort the candidate chunks before selecting the final top-K results.

## 5. Initial Configuration

The current retrieval configuration is:

| Parameter | Value |
|---|---:|
| Initial Top-K | 5 (API default) |
| Expanded Top-K | 10 |
| Final Top-K | 5 (API default) |
| Good Retrieval Threshold | 0.65 |
| Minimum Results | 2 |
| Dense Weight | 0.70 |
| Lexical Weight | 0.30 |

These values represent initial heuristic settings. They have not yet been empirically calibrated against a labelled evaluation dataset.

## 6. Failure Handling

The retrieval module incorporates controlled failure handling.

### Initial Retrieval Failure

If the initial Pinecone retrieval operation fails:
- The system does not attempt query rewriting.
- Empty chunks are returned.
- Retrieval diagnostics identify the strategy as retrieval_unavailable.
- The API returns HTTP 503.

### Query Rewriting Failure

If query rewriting fails:
- Initial retrieval results are preserved.
- Expanded retrieval is skipped.
- The strategy is identified as initial_retrieval_fallback.

### Empty Retrieval

If no relevant chunks are retrieved:
- The system does not generate an answer using empty context.
- An appropriate no-context response is returned.

### Expanded Retrieval Failure

If expanded retrieval fails, the initial candidate results are retained and processing continues with the available results.

## 7. API Integration

The adaptive retrieval module is integrated into the FastAPI backend.

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /health | Backend health check |
| POST | /ingest | Upload and process PDF documents |
| GET | /ingest/status/{job_id} | Check ingestion progress |
| POST | /query | Execute adaptive retrieval and generate answers |
| DELETE | /clear | Clear a specified Pinecone namespace |

### Query Request

Example request:

    {
        "query": "What is the admission process?",
        "namespace": "academics",
        "top_k": 5
    }

### Query Response

The response includes:

- Generated answer.
- Retrieved document chunks.
- Retrieval diagnostics.

The retrieval diagnostics include:
- Initial result count.
- Final result count.
- Retrieval quality.
- Highest initial similarity score.
- Whether query rewriting occurred.
- Rewritten query.
- Retrieval strategy.
- Expanded retrieval improvement indicator.

## 8. Testing and Verification

### Unit Testing

The adaptive retrieval module was tested using Python's unittest framework.

The test suite contains 12 test cases covering:

1. Good retrieval quality assessment.
2. Poor retrieval quality assessment.
3. Empty retrieval assessment.
4. Chunk deduplication.
5. Hybrid reranking.
6. Skipping rewriting for strong retrieval.
7. Triggering rewriting for weak retrieval.
8. Empty retrieval handling.
9. Preserving initial results when rewriting fails.
10. Duplicate removal after retrieval.
11. Initial Pinecone retrieval failure.
12. Expanded retrieval with no improvement.

**Recorded result: 12 tests passed, 0 failures.**

### API Integration Testing

The FastAPI backend was tested through Swagger UI.

The verification covered:
- Backend health check.
- PDF ingestion.
- Ingestion status tracking.
- Adaptive query execution.
- Retrieval diagnostics in the API response.
- Empty namespace handling.
- Namespace-specific clearing.

**Recorded result: API integration testing completed successfully.**

## 9. Limitations and Pending Validation

The following limitations are acknowledged:

1. The retrieval quality threshold is heuristic and has not been empirically calibrated.
2. Dense and lexical reranking weights are initial configuration choices.
3. Lexical similarity uses basic token overlap and does not capture all semantic relationships.
4. Comparing the highest raw similarity scores across different queries is only a heuristic diagnostic and does not establish actual relevance improvement.
5. Query rewriting may occasionally alter the original query's intent.
6. The current implementation does not include a comprehensive labelled retrieval evaluation dataset.

### Pending External-Data Validation

The following activities will be completed using the shared document collection:

- Creation of a labelled query and relevant-chunk evaluation dataset.
- Baseline dense retrieval versus adaptive retrieval comparison.
- Recall@5 evaluation.
- Mean Reciprocal Rank (MRR) evaluation.
- Normalized Discounted Cumulative Gain (nDCG@5) evaluation.
- Retrieval latency comparison.
- Query rewriting effectiveness analysis.
- Validation of retrieval thresholds and reranking weights.

These experiments are pending and no benchmark results are claimed in this documentation.

## 10. Phase 1 Deliverables

| Deliverable | Status |
|---|---|
| Adaptive retrieval module | Completed |
| Retrieval quality assessment | Implemented |
| Conditional query rewriting | Implemented |
| Expanded retrieval | Implemented |
| Deduplication | Implemented |
| Hybrid reranking | Implemented |
| Failure handling | Implemented |
| FastAPI integration | Completed |
| Unit testing | 12/12 passed |
| API integration testing | Completed |
| Technical documentation | Completed |
| External-data benchmarking | Pending |
| Threshold calibration | Pending |

## 11. Conclusion

Phase 1 establishes the Adaptive Retrieval Intelligence component of the proposed system.

The implementation moves beyond conventional single-pass dense retrieval by introducing retrieval quality assessment, conditional query rewriting, expanded retrieval, deduplication, hybrid reranking and controlled failure handling.

The component is integrated into the existing FastAPI and Pinecone-based document retrieval backend.

The implementation and functional testing milestones are complete. Empirical benchmarking and threshold calibration remain scheduled for validation against the shared document collection.
