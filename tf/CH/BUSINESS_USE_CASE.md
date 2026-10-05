# TF — my business-use-case guide

**Originally prepared by CH · 2026-10-05 · My public first-person adaptation · Separate from CL’s guide**

## What I am building with Ya

**I am working with Ya to build a Feature Tokenizer Transformer (FT-Transformer) that scores bank transactions for fraud risk.** We are also testing whether the account’s recent transactions improve that score.

- **Feature:** a field such as transaction amount, available balance or cardholder presence.
- **Tokenizer:** turns each field into a learned list of numbers, called a token.
- **Transformer:** learns relationships between those tokens.
- **Output:** a fraud-risk score for the transaction being assessed.

Our history models extend the FT-Transformer idea rather than exactly reproduce the research model.

## Business purpose

- Catch more fraud within the same investigation capacity.
- Reduce unnecessary alerts while retaining fraud detection.
- Give fraud operations better alert rankings and model developers a challenger to evaluate.

The comparator is the bank's existing **XGBoost debit-card fraud model**. Beating it has not yet been demonstrated.

## The data I use

- Historical bank transaction tables stored as **Parquet files in S3**.
- Fraud label: `frd_tag`, with 1 meaning fraud and 0 meaning non-fraud.
- **23 selected features:** six numeric fields, thirteen categorical fields and four calendar fields derived from transaction time.
- Examples: amount, external Visa fraud score, available balance and limits, transaction type, PIN/CVV indicators, cardholder presence and hour of day.
- Reported 100-file training set: **86.7 million transactions; 143,338 fraud cases (0.1654%)** — about 17 in 10,000.
- Account IDs organize history; transaction IDs identify predictions. IDs and previous fraud labels are not model predictors.

These counts come from project documentation.

## How my pipeline prepares the data

For the current streaming and history pipeline:

1. Selected fields and derived month, day, hour and weekday.
2. Filled missing numbers using training-set medians, put numbers on comparable scales and retained missing-value flags.
3. Gave categories fixed numeric codes, including missing and unknown values.
4. Saved these preparation rules and reused them for validation and scoring.
5. Grouped transactions by account and ordered them by timestamp and transaction ID.
6. Built history windows, added time gaps and masked empty history positions.

**History 5 means five previous transactions plus the current transaction. It does not mean five days.** The default history input has 30 feature tokens per transaction after the added fields.

History does not carry across files. Account grouping and equal-time ordering must preserve information available when scoring.

## The models and why we are testing them

| Model or method | What it does | Reason and potential benefit |
|---|---|---|
| Single-transaction FT-Transformer | Scores the current transaction alone. | Reference before adding history. |
| **A: flat history (Phase 4a)** | Reads all fields across the transaction window together. | Direct interactions across transactions; longer windows need more memory. |
| **B: hierarchical history (Phase 4b)** | Summarizes each transaction, then combines the summaries. | More compact history attention; actual accuracy and speed need measurement. |
| **Masked pretraining** | Learns to reconstruct hidden fields without fraud labels in that objective. | Tests whether learning transaction structure helps later classification. |
| **Supervised contrastive pretraining** | Uses fraud labels to organize learned representations. | Tests a different starting point for fraud classification. |

The current pretraining fine-tuner handles single transactions. Connecting it to A or B remains additional work.

## What the experiments mean

- **Initial A sweep:** eight combinations of how numbers are encoded, how tokens interact and how training updates the model.
- **Selected settings:** linear encoding, bidirectional attention and AdamW, selected using validation results.
- **History sweep:** compare 5, 10 and 16 previous transactions.
- **A versus B:** compare two ways of processing history, including accuracy, memory and speed.
- **100-file confirmation:** increase training data from 10 files to 100 files.
- **Phase 5:** compare starting from scratch against pretraining, after correcting the relevant objectives.

**“100 files” describes training volume. “h5” describes history length. Neither describes the number of evaluation files.**

## My latest documented run status

**Latest recorded bank-run status: 2026-10-04 at 23:09, pod-log time. This is a documented snapshot, not a live job check.**

| Experiment | Training files | Best recorded validation AP | Recorded status |
|---|---:|---:|---|
| Single-transaction reference | 100 | 0.124700 | Result documented. |
| Flat A, h5, selected settings | 10 | 0.136500 | Training and scoring documented. |
| Flat A, h10 | 10 | 0.139942 | Training and scoring documented. |
| Flat A, h16 | 10 | 0.138256 | Training and scoring documented. |
| Flat A, h5 confirmation | 100 | **0.141585** | Training complete; evaluation scoring pending. |
| Hierarchical B, h5/h10/h16 | 10 planned | No result recorded | h5 had started; h10 and h16 were queued. |
| Phase 5 pretraining | Configuration-dependent | No result recorded | Code exists; no outcome documented in this snapshot. |

**AP means average precision:** a summary of how well the model ranks fraud. Higher is better on the same evaluation population. AP 0.14 does **not** mean 14% accuracy or 14% of fraud caught.

These scores are promising. Controlled comparisons are still needed to separate history's contribution from architecture, training and data differences.

## Why I am doing this and the benefits I am testing

- **Feature interactions:** learn combinations such as amount, balance and transaction type.
- **Account context:** test whether recent behavior helps identify unusual activity.
- **Scale:** process files or batches instead of loading the whole dataset into memory.
- **Reusable outputs:** use scores and learned representations in downstream experiments.

Reduced fraud losses and fewer false alerts are benefits to demonstrate, not results already achieved.

## How I can use the outputs downstream now

**Available now: offline research scoring.** Prediction files contain transaction/account IDs, amount, the known fraud label and a score.

```text
Historical transactions → saved preprocessing + model → scores → offline comparison
```

| Use | What we can do | Current position |
|---|---|---|
| Challenger comparison | Compare both models on the same transactions. | Research use with matching predictions and preparation rules. |
| Alert-budget simulation | Measure fraud caught within a fixed number of alerts. | Helper code exists; needs verified outputs and corrected reporting. |
| Error analysis | Investigate misses, false alerts and model disagreements. | Can use prediction files and outcomes. |
| Combined model | Feed transformer scores or representations into another model. | Proposed; needs separate training and evaluation without leakage. |
| Analyst queue / shadow scoring | Add candidate scores alongside the existing system. | Future integration. No live service or analyst integration established. |

**Illustration:** the top 0.1% of one million transactions is 1,000 alerts. Compare fraud caught within those same 1,000 slots. This is an example, not a measured result.

## What I need next

1. Fix training-file coverage; keep separate outputs and exact run settings.
2. Resolve **one evaluation file versus eleven**: configs select 8000; documentation describes 8000–8010.
3. Obtain the h5/100-file evaluation and Architecture B results with their logs.
4. Test the same architecture with/without history and time gaps; repeat promising settings to check consistency.
5. Compare with XGBoost at the same alert budget, then confirm on untouched later data. Measure probability reliability, latency and cost before operational use.

Candidate fixes exist; the reviewed application source does not show them applied. Pretraining fixes gate those experiments, not unrelated supervised comparisons.

## My short meeting explanation

“We are building a transformer that scores transaction fraud risk. We are testing whether account history helps, using flat and hierarchical designs. Results are promising, but we still need reliable comparisons against our existing model. The immediate use is offline scoring; the business goal is more fraud caught for the same review effort.”

### Sources

CH checked these facts against the private project snapshot available on 2026-10-05. Exact source identity remains in the private review record.

The private predictor list, data inventory, dated run-status document and scoring source support this guide. My public [project guide](PROJECT_GUIDE.md), [audit findings](AUDIT_FINDINGS.md) and [independent v2 review](INDEPENDENT_REVIEW.md) provide the detailed qualifications. I retain those qualifications where older source documents overstate conclusions.

The [public copy notes](../PUBLICATION_NOTES.md) explain identity aliases, first-person wording and the original review scopes. CH performed the independent checks attributed to CH.
