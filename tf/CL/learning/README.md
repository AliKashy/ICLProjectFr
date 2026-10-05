# TF — CL learning guide, in my project voice

This is my full explanation of TF, from the basic machine-learning ideas to the data pipeline, history models, experiments, findings and questions I may be asked. I am building on Ya’s foundation at the bank.

Original chapter author: **CL**. Public adaptation: **CH**, in my first-person project voice. The original author incorporated independent review corrections on 2026-10-05. These public copies preserve the author’s explanations and qualifications; they do not imply that either assistant newly endorses every claim in the other’s work.

Experiment status is a **recorded 2026-10-04 snapshot**. Audit and proposed-fix statements describe the **2026-10-05 review**. A run described as started or a result described as pending is not a live status check. Tests described here were reported by the independent assistants; this document does not claim I personally ran their verification programs.

Source paths and commands are **illustrative aliases** for the underlying project, not source files or runnable commands included in this public documentation. `my_work/` refers to my implementation, `ya_work/` to Ya’s foundation, and `s3://<bank-storage>/` to a placeholder storage address. `My` and `Ya’s` are the two artifact-ownership labels. Exact dataset counts, fraud rates, experiment scores and model details are retained with my approval for public release.

I can read the [complete learning guide](LEARNING_GUIDE.md), or start with the ten-minute version, questions and cheat sheet. The original source documents remain unchanged.

## Chapters

- [Part 0. The ten-minute version](parts/00_ten_minute_version.md)
- [Part 1. Foundations from zero](parts/01_foundations.md)
- [Part 2. Neural networks and transformers from zero](parts/02_neural_nets_and_transformers.md)
- [Part 3. The data and Ya’s pipeline](parts/03_data_and_ya_pipeline.md)
- [Part 4. History-aware models and the sweeps](parts/04_history_models_and_sweeps.md)
- [Part 5. Pretraining the encoder (Phase 5), from zero](parts/05_pretraining.md)
- [Part 6. What the audits found, in plain words](parts/06_audit_findings_plain.md)
- [Part 7. Questions I may be asked, with my answers](parts/07_questions_and_answers.md)
- [Part 8. One-page cheat sheet](parts/08_cheat_sheet.md)
- [Part 9. Glossary](parts/09_glossary.md)

## What this covers

- The business problem and why we are testing a Feature Tokenizer Transformer.
- Features, labels, training, fraud metrics and uncertainty.
- Ya’s data preparation and streaming foundation.
- My flat and hierarchical history models and recorded experiments.
- My two pretraining approaches and their open design defects.
- Independent audit findings, limits on the conclusions and the proposed sequence of work.
- Questions, short answers, a cheat sheet and a glossary.

The research record is exploratory. Any downstream comparison still needs the corrected loader, a pinned environment and evaluation cohort, separate prediction artifacts, and suitable repeated and paired comparisons. This guide does not report a production deployment or an observed business benefit.
