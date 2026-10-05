# TF: business-use-case guide

**CL’s guide · First-person public adaptation by CH · 2026-10-05**

## What we are building

I am building a fraud-risk score for each debit-card transaction, produced by a transformer model that reads the transaction's
own fields and, in the history-aware version, the account's recent transactions as well.

- A **transformer** is the neural-network design behind modern language models. Here it reads a table row
  instead of a sentence: each field becomes a learned vector ("token") and the model learns how the fields
  relate.
- The specific design is the **FT-Transformer** (Feature Tokenizer + Transformer, Gorishniy et al. 2021). The
  history models extend that idea to a window of transactions; they are built in-house, not copied from the
  paper.
- Output: one uncalibrated fraud score per transaction, used to rank transactions for review.

## Why

- The production fraud model is an XGBoost tree model that sees one transaction at a time through hand-built
  features. Account context ("is this normal for this account?") has to be engineered by hand.
- A history-aware transformer reads the last N transactions directly and can learn account-relative and
  velocity patterns itself.
- Business goal: more fraud caught within the same review capacity, fewer false alerts, and a challenger model
  for model-risk comparison. Beating the production XGBoost has not been demonstrated yet; that comparison is
  on the plan.

## The data

- Historical debit-card transactions in parquet files on S3. Training and validation files `part_1` to
  `part_5000`; evaluation files `part_8000` to `part_10000`. One file is about 840,000 transactions from about
  3,300 accounts.
- Label: `frd_tag` (1 = confirmed fraud). Fraud rate 0.1654% in training files 1 to 100 (143,338 of
  86,683,008 transactions), about 1 in 600.
- 23 features known at authorization time: 6 numeric (amount, external fraud score, available balance, three
  daily-limit fields), 13 categorical (transaction type, segment, PIN, CVV, cardholder present and similar) and
  4 derived from the timestamp (month, day, hour, weekday). No identifiers, no merchant ids, and no earlier
  fraud labels are inputs.

## What we did to the data

1. Selected the 23 features and derived the 4 calendar fields.
2. Froze all preprocessing statistics on training files 1 to 5: medians, means and standard deviations for
   numbers; category vocabularies with reserved codes for "unseen" and "missing". The same frozen statistics
   are applied at training, validation and scoring, so nothing is refit on held-out data.
3. Added a missing-value flag per numeric field.
4. For history models, built a window per transaction: the target plus its N preceding transactions on the
   same account (timestamp order, transaction id as tie-break), left-padded when fewer exist, with a log
   time-gap feature per slot. Only the target's label is used.
5. Streamed files from S3 one at a time with a 65,536-row shuffle buffer, because 100 files do not fit in
   memory.

## Experiments and recorded status

| Experiment | Data | Validation PR-AUC | Status |
|---|---|---:|---|
| Single transaction, FT-Transformer (Ya) | 100 files | 0.1247 | done; reported 11-file evaluation 0.0940 |
| History 5, Ya’s defaults | 10 files | 0.1293 | done |
| History 5, design-choice sweep (8 runs), best | 10 files | 0.1365 | done; two runs scored 0.1215 and 0.1224, below the baseline |
| History 10 / history 16, best choices | 10 files | 0.1399 / 0.1383 | done; gaps small, one seed, unresolved |
| History 5, best choices | 100 files | 0.1416 | trained; evaluation scoring pending |
| Hierarchical architecture (one vector per transaction, then a sequence model) | 10 files | pending | sweep started 2026-10-04 23:09, no results documented |
| Encoder pretraining (masked autoencoder, supervised contrastive) | | none | written; no result documented; two code defects found first |

PR-AUC (average precision) is the ranking metric; a random model scores about 0.0016; higher is better. The
numbers above are validation numbers from the documented runs, one training seed each.

## What the two independent audits found (2026-10-05)

CL and CH independently audited my project and cross-checked each other. Their documented checks found that the code parses and all 28 offline tests pass. They found no transcription-specific failure; this does not certify every photographed line or every runtime path. Defects found,
in the recovered project logic: a data-loader sharding bug that made each training epoch read about two thirds of
the files (validation unaffected); a "causal" attention option that does not meet its contract; an evaluation
range that is one file in the configs but eleven in the documents; sweep runs overwriting each other's
prediction files; two pretraining-loss defects; and an uninstallable package definition. Proposed fixes were
tested in disposable copies. CH’s independent v2 verdict was 7 confirmed within scope and 4 partial;
CL’s later v3 EMA and runner revisions await independent verification. None is applied to the project code yet.

## What stands and what does not

- Stands: the best history-aware runs score above the single-transaction baseline on held-out validation
  (0.1247 to about 0.14). The window construction uses only preceding transactions and no labels.
- Not yet established: how much of that lift is history itself (the history models also change the encoder;
  a no-history control is proposed); whether longer history beats 5; whether the hierarchical design wins;
  whether pretraining helps; any evaluation-set number until the one-file versus eleven-file question is
  settled; how much the loader defect moved the scores.

## Benefits if it works

- A ranking signal that uses account history without hand-built velocity features.
- A challenger to the production XGBoost, or an additional feature for it (the transformer score fed into
  XGBoost), measured at the review budget (precision and recall in the top 0.1% and 0.5% of transactions).
- An encoder whose transaction embeddings can serve other tools (clustering, case review, drift monitoring).

## Downstream uses and what they need

- Offline scoring of an evaluation cohort: the Phase 4 scoring path exists and uses a checkpoint plus frozen preprocessing. I still need to resolve the cohort, environment and output-retention issues before relying on a comparison.
- Comparison with production XGBoost: needs both models' scores on the same identified transactions.
- Near-real-time scoring: needs a store of each account's last N transactions with the frozen preprocessing
  applied; not built; latency and retrieval cost unmeasured.
- Alert-budget thresholds: can be selected from ranked held-out scores. Probability or expected-loss thresholds need measured calibration; no calibration result is documented yet.

## Decisions to make

1. I need verified loader coverage, a reproducible environment, an identified evaluation cohort and unique prediction outputs before new comparative runs. Other fixes gate the paths that use them: causal attention, EMA or Phase 5.
2. I need to pin the evaluation cohort (files 8000 to 8010, recorded by transaction id) and recover the pod logs that
   say which range produced the recorded evaluation scores.
3. I need to fund the controlled rerun: history 0, 5, 10, 16, flat and hierarchical, 3 to 5 seeds, corrected loader,
   paired comparison on the pinned cohort, then the XGBoost head-to-head.

## Success criteria for the next round

- A seeded, paired comparison shows the history model above the no-history control on the pinned cohort.
- Precision and recall at the review budget reported next to PR-AUC, for the transformer and for production
  XGBoost on the same transactions.
- Every run's predictions kept, every number traceable to a config, a commit and a log.

## Public copy notes

> Public adaptation prepared by CH from CL’s dated 2026-10-05 documents. CL remains the original author; this adaptation does not imply CL approved its wording. I use **TF** for my project, **Ya** for my collaborator, **My** for my storage ownership and **Ya’s** for Ya’s storage ownership. Dataset counts, rates, scores and model details are retained with my approval.
>
> This is a documentation-only copy. Code paths such as `my_work/` and `ya_work/`, storage addresses, and references to private proofs are illustrative aliases or evidence descriptions—not files or commands available in this public folder. Test execution and findings are credited to the assistant that reported them. Status is historical as of the dated review.

Original author: CL, 2026-10-05. This is my public business summary based on that dated edition, separate from [CH’s version](../CH/BUSINESS_USE_CASE.md). Recorded outcomes and unresolved questions are kept distinct.
