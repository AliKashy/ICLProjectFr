# TF

**New, October 20:** [Questions for the data owner](DATA_OWNER_QUESTIONS_2026-10-20.md): the eight questions only a person can answer, and the handful of items for me. Everything else that used to be a question will be measured by a read-only script I run on the bank's machine (written and tested; not yet run there).

**New, October 19:** [V2 proposed features workbook](features/README.md): 29 candidate inputs, each with the raw fields and transactions it is derived from.

**New, October 19:** [CL’s questions for me](CL_QUESTIONS_2026-10-19.md), 21 numbered questions, with CL’s recommendation where there was a choice; most are answered or decided.

**New learning course, October 15:** [Tabular Fraud, Explained — download the offline course ZIP](explained/README.md). Twenty beginner chapters, interactive practice and recall, CPU labs, a project atlas, oral-defence exercises, and a complete PDF. Extract the ZIP and open `index.html`.

**Earlier review:** [October 6 independent CH review — code, Ya's plan, fixes and statistical next steps](reviews/2026-10-06/CH/README.md). The guides below preserve their October 5 snapshot.

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

I can answer the [open questions](QUESTIONS.md) by number. Most of the 5 October list is now answered or overtaken; the answer lines say so. CL’s [meeting questions and answers](CL/learning/parts/07_questions_and_answers.md) explain how I can discuss the project; those are different from the unanswered project questions.

## Where things stand (20 October 2026)

- **Version 1 is complete.** The 15 planned runs finished on 12-13 October: the single-transaction model, the flat history model and the hierarchical history model at several history lengths, all trained on ten training files and scored on the same eleven evaluation files (3,818 fraud cases). The best evaluation APs were 0.095459 and 0.095465 for the hierarchical model at history 15 and 20 (both hand-retrained and scored separately); the all-position variant at history 5 scored 0.0927 when scored in chunks and 0.093420 when scored on windows ending at each transaction, against 0.0674 for the plain history-5 run.
- **Version 2 has a governing plan** agreed by CL and CH on 19 October (one rule's recorded risk acceptance, proposed by CL, still awaits CH's confirmation): test the chunk-trained model against the plain history model at the same history (5 and 20), without the month input, then grow the winner to a hundred training files, with one optional slot for a new feature family. Its decision rules are written down before any result is read.
- **Nothing in version 2 has run on the bank's machine yet.** The version 2 code is packaged as [`007_20261020_062736_UTC.zip`](releases/007_20261020_062736_UTC.zip) in `releases/` (the tools only; nothing runs by itself, and no result is claimed). The first step is a read-only script that measures the machine and the data (counts, types and names only); I run it and return the files it lists. Training follows only on my word.
- **The questions that remain for a person** are in [one short list](DATA_OWNER_QUESTIONS_2026-10-20.md).
- The immediate use is offline fraud scoring and comparison. Beating the bank’s existing model, a benefit from pretraining and a live deployment have not been demonstrated.

## What I could say on 5 October (kept as history)

- I am testing a single-transaction transformer, a flat history model and a hierarchical history model.
- My best recorded h5 / 100-training-file validation AP is **0.141585**. Evaluation scoring was pending in the last bank-run snapshot, **2026-10-04 at 23:09, pod-log time**.
- Counts and scores are documented observations. A started run is not a current job check, and a proposed fix is not an applied fix.

## How I read the reviews

CH and CL are independent verifiers. I keep their accounts and disagreements visible. CH’s [independent review](CH/INDEPENDENT_REVIEW.md) addresses CL’s **v2** proposals. CL’s later **v3** checks are reported by CL; this publication does not establish a new CH verification of v3.

These are adapted copies; the original findings remain unchanged. Source paths used in the guides are illustrative aliases, not runnable public-repository paths.

The [copy register](PUBLICATION_NOTES.md) records the scope and wording changes. This documentation set was prepared on **2026-10-05**.
