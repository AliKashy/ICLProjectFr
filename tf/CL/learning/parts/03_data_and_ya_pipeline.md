# Part 3. The data and Ya’s pipeline

Original chapter author: CL. Public adaptation: CH, in my first-person project voice.
The original author incorporated an independent CH review on 2026-10-05; this adaptation is not a new joint endorsement.

Status note: experiment status is the recorded 2026-10-04 snapshot; audit and proposed-fix statements describe the 2026-10-05 review. They are not live job or deployment status.
Source paths and commands are illustrative aliases, not files or runnable commands supplied by this public guide. `my_work/` means my implementation; `ya_work/` means Ya’s foundation; `s3://<bank-storage>/` replaces the actual storage address.

Parts 1 and 2 gave the ideas. This part walks through what Ya actually built, in the order the data flows,
so I can answer "where does the number come from" at every step.

---

## 3.1 Where the data lives

The bank's transaction data sits in an object store (S3) as **parquet** files, a compressed column-oriented
format that Python reads directly. Two prefixes matter:

| Prefix | Files | Role |
|---|---|---|
| `s3://<bank-storage>/training/` | `part_1.parquet` to `part_5000.parquet` | training and validation |
| `s3://<bank-storage>/evaluation/` | `part_8000.parquet` to `part_10000.parquet` | evaluation |

One file holds about 840,000 transactions from about 3,300 accounts, and an account's transactions are all in
the same file (this is a design assumption recorded in Ya’s decisions, and it is what makes history windows
buildable one file at a time). Ya’s notebooks measured the first file: 841,428 rows, 3,334 accounts, median
133 transactions per account, 10% of accounts with 3 or fewer, 10% with 670 or more.

If asked "how big is the data?": thousands of files of about 840,000 transactions each; the experiments so far
used 10 or 100 training files, 5 validation files, and 1 or 11 evaluation files.

---

## 3.2 One row is one transaction

Each row has dozens of columns. Ya’s exploratory notebook (`ya_work/notebooks/eda.ipynb`) lists the candidates with
their missing rates and number of distinct values. The columns that are **not** features but are still needed:

- `fi_transaction_id`: unique id of the transaction (used to join predictions back, and as a tie-break when
  two transactions share a timestamp).
- `Nmap_account_reference_xid`: the account (used to group history).
- `transaction_dttm`: the timestamp (used to order history and to derive calendar features; never fed to the
  model raw).
- `frd_tag`: the label, 1 for fraud.

If asked "what is the unit of prediction?": one transaction; the label is whether that transaction was later
confirmed as fraud.

---

## 3.3 Choosing the 23 features

Ya kept features that are (a) known at the moment the transaction is authorized, (b) low-cardinality enough
to embed, and (c) not identifiers. He wrote them down in a versioned file, `ya_work/configs/predictors.csv`, with
columns `feature, type, source, version, description`, so that adding a feature later means adding a row with
`version 2`, and every script derives its column lists from that file instead of hard-coding them.

| Group | Count | Examples |
|---|---|---|
| Continuous, raw | 6 | amount; Visa's external fraud score (0 to 99); available balance; three daily-limit fields |
| Categorical, raw | 13 | transaction type (12 values), segment (6), category code (6), POS condition, AVS, PIN verified, cardholder present, CVV result, cryptogram valid, off-premises, unattended terminal |
| Categorical, derived from the timestamp | 4 | month, day of month, hour, day of week |

Excluded on purpose: merchant id (288,000 distinct values in two files), postal codes, device IP, network id,
`transaction_advice_xcd` (95% missing), and the labels of earlier transactions. The two highest-missing kept
fields are `avs_request_xcd` (65% missing) and `pos_security_xflg` (82% missing). Missing categorical values
get their own reserved code; missing continuous values also receive an explicit binary flag (next section), so
missingness is a signal either way.

If asked "why these features?": available at authorization time, low cardinality, non-identifying; the list is
versioned so it can grow without breaking older models.

---

## 3.4 Frozen preprocessing

A model trained on numbers in one scale must be scored on numbers in the same scale, forever. For the
streaming and history paths, Ya therefore computes every preprocessing statistic **once** (the default build
uses training files 1 to 5) and saves it as **artifacts** (JSON files on S3 under
`s3://<bank-storage>/Ya/artifacts/1/`, one manifest plus one file per feature). Streaming training, validation and
scoring all load the same artifacts. Nothing is ever recomputed on validation or evaluation data, which is the
leakage-safe way. (The earlier in-memory baseline of Phase 2 is the exception: it fits its own imputer and the
library's preprocessing on its own training split and saves them with the model.)

Per continuous feature the artifact stores the mean, standard deviation and median. At use time: fill missing
values with the median, then standardize `(x - mean) / std`. Separately, a `<feature>_isna` flag records
whether the value was missing before filling (6 flags, one per continuous feature, treated as categorical
features with 2 values).

Per categorical feature the artifact stores a **vocabulary**: a mapping from each value seen in training to an
integer, starting at 2. Index **0** is reserved for a value never seen in training ("unknown"), index **1**
for missing. So a transaction type that appears for the first time in production does not crash anything; it
is embedded as "unknown".

The four calendar features are derived from the timestamp (`ya_work/data/datetime_features.py`) before encoding;
an unparseable timestamp becomes the category "-1".

Result: each transaction becomes 6 standardized numbers plus 23 integers (13 + 4 categorical codes and 6
flags). With the history model's time-gap feature that is the 30 tokens of Part 2.

An open question for me, raised by both audits: my configs read artifacts from the storage prefix labeled `My`, while Ya built the original artifacts under the prefix labeled `Ya’s`. I need to confirm whether my artifacts were copied unchanged or refitted. If they were refitted, my results and Ya’s are not based on identical transformed inputs.

My explanation of preventing preprocessing skew: every statistic is frozen from training files
1 to 5, versioned, and applied identically everywhere; unknown and missing categories have reserved codes.

---

## 3.5 Streaming instead of loading

Ten files are 8.4 million rows; one hundred are 84 million. They do not fit in memory, and the library's
`TabularModel.fit` wants one big table. So Phase 3 (`ya_work/docs/phase3_streaming_plan.md`) built a **streaming
dataset** (`ya_work/data/parquet_loader.py`): a PyTorch `IterableDataset` that opens one parquet file at a time,
reads it in blocks of 65,536 rows, applies the frozen artifacts to the block, shuffles the rows inside the
block, and yields batches of 8,192 rows as tensors. Several **workers** (4) do this in parallel, each on its
own slice of the file list.

Two details matter for the meeting:

- **Row shuffling inside the block was decisive.** Rows in a file are clustered by account and time; without
  shuffling, each batch was nearly one account, and training collapsed to a coin flip (ROC-AUC 0.5) at batch 512.
- **The worker slices were not a clean partition** when file shuffling was on; this is the sharding defect in
  Part 6. It affected training only.

My explanation of training beyond memory limits: a streaming dataset reads parquet blocks from
S3, applies frozen preprocessing per block, shuffles within the block, and feeds batches to the trainer; four
workers in parallel.

---

## 3.6 The single-transaction model and the training loop

Two code paths train the single-transaction FT-Transformer:

- **Baseline (Phase 2, `ya_work/scripts/train_baseline.py`)**: loads two or three files into memory and uses the
  `pytorch_tabular` library end to end (its `TabularModel` fits its own encoders and runs the trainer). Used for
  the first smoke test.
- **Streaming (Phase 3, `ya_work/scripts/train_streaming.py`)**: builds the library's model directly from the
  artifact dimensions (`ya_work/streaming_model.py`) and drives it with a plain PyTorch Lightning `Trainer` fed by
  the streaming dataset. This is the path that scales.

The two paths select differently. The baseline monitors validation loss during training and computes PR-AUC
and ROC-AUC over the full split after fitting. The streaming and history classifiers compute PR-AUC and
ROC-AUC over the **whole** validation set at the end of each epoch (`StreamingBinaryMetrics` in
`ya_work/callbacks.py`), write one CSV row per epoch (`training_metrics.csv`), early-stop on validation PR-AUC
with patience 5, and save `best.ckpt`. Scoring scripts (`score_eval.py`, `score_streaming.py`) reload the
checkpoint and the preprocessing, score an evaluation range file by file, write parquet files of (ids, label,
score) to S3, and print the range's PR-AUC and ROC-AUC. A consistency check matters here: scoring `part_8000`
with the saved baseline model reproduced that run's test numbers (ROC-AUC 0.9521, PR-AUC 0.0940). That is a
useful check on that saved model and file; it does not verify every model or configuration, and it does not
show that the artifacts under the two users' S3 prefixes are identical.

If asked "what framework?": PyTorch Lightning with a streaming data loader; the model is the `pytorch_tabular`
FT-Transformer for single transactions and a hand-built transformer for the history models.

---

## 3.7 Phases 0 to 3: the results

| Phase | Setup | Validation | Evaluation |
|---|---|---|---:|
| 0 (public data) | UCI Bank Marketing, FT-Transformer vs MLP vs SAINT | ROC-AUC 0.935 vs 0.921 vs 0.82 to 0.85 | not fraud data |
| 2 baseline | 2 train files, 1 epoch | ROC-AUC 0.9275, PR-AUC 0.1051 (part_4950) | ROC-AUC 0.9521, PR-AUC 0.0940 (part_8000) |
| 3 streaming | 10 files, 2 epochs | PR-AUC 0.1085 | PR-AUC 0.0732 (part_8000) |
| 3 streaming | 10 files, 10 epochs | PR-AUC 0.1131 (epoch 7) | |
| 3 streaming | 100 files, early-stopped | PR-AUC 0.1247 (epoch 2), ROC-AUC 0.9493 | PR-AUC 0.0940, ROC-AUC 0.9514 (parts 8000 to 8010) |

What Ya concluded, and why it set up Phase 4: going from 10 to 100 files moved validation PR-AUC from
about 0.11 to about 0.125; more epochs did not help beyond epoch 2 to 5; a bigger model did not help. When
data, time and capacity all stall, a natural reading is that the single-transaction model is close to
**feature-limited**. That plateau motivated looking for new information rather than a bigger model; it did not
prove the 23 columns were exhausted (optimization settings or data coverage can also produce a plateau). The
obvious new information is what the account did before.

If asked "what did the single-transaction model achieve?": validation PR-AUC about 0.125 and ROC-AUC about
0.95 at 100 files, plateauing with more data or capacity, which motivated adding history.

---

## 3.8 Ya’s documentation, where I can find the supporting documentation

- `ya_work/docs/fraud_detection_plan_overall.md`: goals and why the library was forked.
- `ya_work/docs/fraud_detection_plan_detailed.md`: the 23 features, preprocessing decisions, smoke results.
- `ya_work/docs/phase3_streaming_plan.md`: streaming design and the 10- and 100-file results.
- `ya_work/docs/phase4_sequential_plan.md` and `ya_work/docs/phase4_design_decisions.md`: the history model design (Part 4).
- `ya_work/docs/monitoring_and_eval.md`: how to watch a run and score a range.
- `ya_work/docs/tabular_transformer_models.md`: the Phase 0 benchmark.
