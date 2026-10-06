# TF

**Latest update:** [October 6 independent CH review — current code, Ya's plan, fixes and statistical next steps](reviews/2026-10-06/CH/README.md). The guides below preserve their October 5 snapshot.

I am building a Feature Tokenizer Transformer for transaction fraud scoring at the bank and testing whether account history improves its predictions.

I keep the two independent documentation sets separate: **CH** and **CL**. My public copies use first-person project-owner wording, **Ya** for my collaborator, and **My** / **Ya’s** for workspace ownership. Exact dataset figures, scores and model details are retained as approved for public release.

## Start here

| What I need | CH | CL |
|---|---|---|
| A direct business explanation | [Business use case](CH/BUSINESS_USE_CASE.md) | [Business use case](CL/BUSINESS_USE_CASE.md) |
| Project structure and models | [Project guide](CH/PROJECT_GUIDE.md) | [Project guide](CL/PROJECT_GUIDE.md) |
| Findings and limitations | [Audit findings](CH/AUDIT_FINDINGS.md) | [Audit findings](CL/AUDIT_FINDINGS.md) |
| Proposed next steps | [Next steps](CH/NEXT_STEPS.md) | [Recommendations](CL/RECOMMENDATIONS.md) |
| Detailed explanations | [Review index](CH/README.md) | [Learning guide and ten chapters](CL/learning/README.md) |

I can answer the [open questions](QUESTIONS.md) by number. They were published first and remain open until I supply the missing facts. CL’s [meeting questions and answers](CL/learning/parts/07_questions_and_answers.md) explain how I can discuss the project; those are different from the unanswered project questions.

## What I can say today

- I am testing a single-transaction transformer, a flat history model and a hierarchical history model.
- My best recorded h5 / 100-training-file validation AP is **0.141585**. Evaluation scoring was pending in the last bank-run snapshot, **2026-10-04 at 23:09, pod-log time**.
- The immediate use is offline fraud scoring and comparison. Beating the bank’s existing model, a benefit from pretraining and a live deployment have not been demonstrated.
- Counts and scores are documented observations. A started run is not a current job check, and a proposed fix is not an applied fix.

## How I read the reviews

CH and CL are independent verifiers. I keep their accounts and disagreements visible. CH’s [independent review](CH/INDEPENDENT_REVIEW.md) addresses CL’s **v2** proposals. CL’s later **v3** checks are reported by CL; this publication does not establish a new CH verification of v3.

These are adapted copies; the original findings remain unchanged. Source paths used in the guides are illustrative aliases, not runnable public-repository paths.

The [copy register](PUBLICATION_NOTES.md) records the scope and wording changes. This documentation set was prepared on **2026-10-05**.
