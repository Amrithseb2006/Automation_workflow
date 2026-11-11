\# Response Verification \& Decision Layer



\## 1. Overview



The Response Verification \& Decision Layer is responsible for evaluating AI-generated responses against evidence retrieved from the institutional knowledge base.



It is designed to prevent unsupported or contradictory information from being presented as reliable answers.



\*\*Module Owner:\*\* Aditya Sarathy



\## 2. Objectives



\- Represent retrieved evidence in a structured format.

\- Represent individual factual claims extracted from generated responses.

\- Classify claim verification outcomes.

\- Aggregate individual verification results into a structured summary.

\- Maintain traceability between claims and retrieved evidence.



\## 3. Architecture



The module is designed to operate after response generation and adaptive retrieval.



\*\*Input:\*\*

\- Original user query.

\- AI-generated response.

\- Retrieved document chunks and metadata.



\*\*Processing:\*\*

\- Evidence representation.

\- Claim representation.

\- Verification-result aggregation.



\*\*Output:\*\*

\- Claim-level verification results.

\- Evidence references.

\- Aggregate verification counts.



\## 4. Claim Verification Categories



| Category | Description |

|---|---|

| Supported | Claim is supported by evidence. |

| Partially Supported | Evidence supports only part of the claim. |

| Unsupported | Retrieved evidence does not support the claim. |

| Contradictory | Evidence conflicts with the claim. |

| Unverified | Claim has not yet been assessed. |



\## 5. Implementation Structure



```text

response\_verification/

├── schemas/

│   ├── evidence.py

│   ├── claim.py

│   ├── request.py

│   └── verification.py

├── core/

│   └── verification\_engine.py

└── tests/

&#x20;   ├── test\_schemas.py

&#x20;   └── test\_verification\_engine.py

```



\## 6. Testing



Unit tests are implemented using Python's unittest framework.



\*\*Current test result: 8 passed, 0 failures.\*\*



Testing covers:

\- Evidence schema validation.

\- Invalid evidence rejection.

\- Retrieval-score range validation.

\- Integer and string page references.

\- Verification request validation.

\- Empty evidence handling.

\- Claim-status aggregation.



\## 7. Current Scope and Limitations



This phase establishes the verification data structures and deterministic aggregation logic.



The following capabilities are not yet implemented:

\- Automated factual claim extraction.

\- Semantic claim-to-evidence verification.

\- Grounding-score calculation.

\- Risk classification.

\- Human-review routing.

\- Response verification integration with the FastAPI query endpoint.



These capabilities are planned for subsequent development phases.



\## 8. Design Principle



Retrieval similarity is not treated as proof of factual correctness. Verification outcomes must ultimately be based on the relationship between individual claims and retrieved evidence.