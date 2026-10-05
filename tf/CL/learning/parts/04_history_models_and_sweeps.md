# Part 4. History-aware models and the sweeps

Original chapter author: CL. Public adaptation: CH, in my first-person project voice.
The original author incorporated an independent CH review on 2026-10-05; this adaptation is not a new joint endorsement.

Status note: experiment status is the recorded 2026-10-04 snapshot; audit and proposed-fix statements describe the 2026-10-05 review. They are not live job or deployment status.
Source paths and commands are illustrative aliases, not files or runnable commands supplied by this public guide. `my_work/` means my implementation; `ya_work/` means Ya’s foundation; `s3://<bank-storage>/` replaces the actual storage address.

This is the part most of the meeting will be about. It covers how a history window is built, the two
architectures, the knobs I tested, and how to read every number in the results tables.

---

## 4.1 Why history

A 900 dollar purchase at 3 a.m. is suspicious for one account and routine for another. A single-transaction
model cannot know which, because it sees one row. Fraud teams have always encoded this with hand-built
"velocity" features: spend in the last hour, number of transactions today, time since the last one. The history
model lets the network read the last N transactions directly and learn which patterns matter. The data
supports it: the median account has 133 transactions, so many accounts can supply histories of 5 to 16. How
often a window needs padding depends on the history length and on where the target falls in the account
(every account's first transactions need padding). Ya’s h5 smoke test checked padding correctness on 30
deliberately selected accounts; its 150 padded windows out of 28,054 is not a population estimate for h5 or
h16.

If asked "why would history help?": account-relative behavior and velocity are classic fraud signals that a
per-transaction model cannot see without hand-built features.

---

## 4.2 How a window is built

`ya_work/data/sequence_builder.py` (Ya’s code, used unchanged by me) does this for every file:

1. Encode every row once with the frozen artifacts (Part 3.4).
2. Group rows by account; sort each account by timestamp, then by transaction id to break ties.
3. For each transaction (the **target**), take the N transactions immediately before it in that account.
4. Build a window of `T = N + 1` slots with the target in the last slot. If fewer than N earlier transactions
   exist, fill the leftmost slots with **padding** (all zeros) and mark them in a **mask**.
5. Add one extra continuous feature per slot: `time_delta = log(1 + seconds between that transaction and the
   target)`; it is 0 for the target itself (and for any earlier row that shares the target's timestamp).
6. The window's label is the target's `frd_tag` only. Earlier transactions contribute features, never labels.

Example, N = 2, an account with transactions A (Mon 09:00), B (Mon 09:05), C (Tue 14:00):

| Target | Slot 0 | Slot 1 | Slot 2 (target) | time_delta per slot |
|---|---|---|---|---|
| A | pad | pad | A | -, -, 0 |
| B | pad | A | B | -, log(1+300), 0 |
| C | A | B | C | log(1+104,400), log(1+104,100), 0 |

Within one file, every transaction is a target exactly once per pass (`stride 1`); a `stride` above 1 would
skip some non-fraud targets during training but never a fraud one. (Across a whole multi-worker training
epoch that "exactly once" is broken by the sharding defect of Part 6.2.) Windows are shuffled in a buffer of 65,536 before batching so that one
batch mixes many accounts (windows of the same account overlap heavily and would otherwise correlate a batch).

Correctness evidence: nine unit tests on synthetic data (shapes, every transaction once, target last, no
future leakage via time delta, left padding, account isolation, empty first history, fraud never skipped by
stride) and Ya’s notebook check against a manual reconstruction on real data. Both audits also verified
that changing the padded slots' values changes no prediction.

If asked "is the history strictly causal?": within a complete account file with valid, sortable timestamps,
yes: the builder uses only preceding rows in timestamp-then-id order and labels only the target; no label of a
prior transaction is used. Two qualifications: rows with an identical timestamp are ordered by transaction id,
which is a stable tie-break rather than a guarantee of availability order (such rows enter the history with
time delta 0); and the timestamp column must be of a type that sorts chronologically, which the tests cover
only for valid, ordered synthetic data.

---

## 4.3 Architecture A: flat

Every slot's 30 tokens are laid out in one long sequence with CLS first. A learned **position embedding** for
the slot is added to all 30 tokens of that slot (so the model knows "these belong to the most recent
transaction"); there is no ordering inside a slot. Four pre-norm transformer blocks (embed 48, 8 heads), then
the CLS vector goes through a linear head to two logits.

```
[CLS] [slot 0: 30 tokens] [slot 1: 30 tokens] ... [slot N: 30 tokens, the target]
        one transformer; every token attends to every other token (bidirectional)
                                  CLS -> linear -> fraud probability
```

Sequence length is `1 + (N + 1) * 30`: **181** tokens at N = 5, **331** at N = 10, **511** at N = 16. Attention
cost grows with the square of that, so N = 16 is about 8 times the attention cost of N = 5. At N = 10 and 16 a
batch of 4,096 windows did not fit in GPU memory; I kept the same effective batch by processing 1,024
windows 4 times (N = 10) or 256 windows 16 times (N = 16) before each weight update. This is **gradient
accumulation**; it approximates one 4,096-window update. It matches the large-batch average exactly when the
micro-batches are full and equally sized and the loss normalization and update timing match; the streaming
loader's short final chunks, dropout and numerical precision can make it differ slightly. The model has no
batch normalization, so nothing in it depends on the micro-batch size directly.

Masks: padded slots' tokens are hidden from attention. The **bidirectional** default lets every token see
every token. A **causal** option was meant to let a slot see only earlier slots; Part 6 explains why the option
as written did not do that.

If asked "what is the sequence length and why does it matter?": 1 + 30 per transaction; 181 to 511 tokens;
attention cost is quadratic in it, which forced gradient accumulation at history 10 and 16.

---

## 4.4 Architecture B: hierarchical

I implemented the alternative Ya had deferred (`my_work/phase4b/sequential_hier.py`):

```
slot 0 tokens  -> [transaction encoder, 2 blocks] -> v0   (one 48-number summary vector)
slot 1 tokens  -> [same encoder, shared weights]  -> v1
...
slot N tokens  -> [same encoder]                  -> vN  (the target)
[CLS] v0 v1 ... vN  -> [sequence encoder, 4 blocks] -> CLS -> linear -> fraud probability
```

Stage 1 compresses each transaction into one vector independently (a per-transaction CLS token does the
summarizing). Stage 2 attends over the `N + 2` vectors (N + 1 slots plus CLS) with position embeddings added.
Padded slots are encoded in stage 1 like any other (they are zeros) and hidden in stage 2.

Trade-offs:

- Stage 2's sequence is 7 to 18 vectors instead of 181 to 511 tokens. Its attention is still quadratic, but
  over only N + 2 vectors, so it is small. The committed B sweep requests batch 4,096 without accumulation,
  including at N = 16; whether that fits and what it costs is not documented yet, and longer histories (32, 64)
  need actual profiling.
- Stage 1 runs once per slot, so its cost grows linearly with N. Overall the design is substantially cheaper
  than flat attention over 1 + 30(N + 1) tokens.
- A field of one transaction can no longer attend directly to a field of another; the interaction must pass
  through the summary vector. Whether that loses signal is exactly what the A-vs-B comparison tests.
- Capacity differs: 2 + 4 blocks against A's 4. So the comparison is "matched knobs", not "matched capacity".

This mirrors the design of the bank's Payment Foundation Model and of published transaction-sequence models
(a field-level encoder, then a row-level encoder).

Recorded status as of 2026-10-04: the sweep `history-h5`, `history-h10`, `history-h16` started on 2026-10-04 at 23:09 pod time; no
results are documented yet.

If asked "flat versus hierarchical?": flat lets any field attend to any field across transactions but pays
quadratic attention over all 1 + 30(N + 1) tokens; hierarchical summarizes each transaction first, so the
quadratic part runs over only N + 2 vectors and long histories become cheap; it is the design used by
sequence-of-transactions models in the literature. Results pending.

---

## 4.5 The knobs I swept

| Knob | Options | Why it might matter |
|---|---|---|
| Numerical embedding | linear (value times a learned vector) vs periodic (sines and cosines at 16 learned frequencies, then a projection) | Periodic embeddings help when the relationship between a number and fraud is not monotone (Gorishniy et al., 2022). |
| Attention mask | bidirectional vs causal | Meant to test whether restricting history-to-target attention changes anything. |
| Optimizer | AdamW vs RAdam | RAdam stabilizes early training; usually a small effect. |
| Weight averaging (EMA) | off (never turned on) | Averaging weights over training often gives a small, free gain. |
| History length | 5, 10, 16 | The main question. |
| Training size | 10 files vs 100 files | Does more data help the history model as it helped the single-transaction one? |

The periodic variant has two implementation choices worth knowing: one projection layer shared across the
seven numeric channels of the history model, the six raw numeric features plus time delta (the paper uses one
per feature), and a fixed frequency scale (the paper tunes it). So
"linear beat periodic" is a statement about this variant.

My answer on tuned hyperparameters: embedding type, attention direction, optimizer, history
length and training size, one seed each; learning rate, depth and width are still at Ya’s defaults
(3e-4, 4 blocks, 48).

---

## 4.6 How a sweep runs

`my_work/phase4a/scripts/run_ablations.py` reads a spec file listing axes of config overrides, forms every
combination, and for each one: trains with `train_sequential.py` into its own folder, reads the best
validation PR-AUC and ROC-AUC from `training_metrics.csv`, scores the best checkpoint on the evaluation range
with `score_sequential.py`, parses the printed metrics, and appends a row to `results.csv`. Three specs exist:

- `ft_decisions.yaml`: embedding x mask x optimizer at history 5 (8 runs).
- `ft_decisions_flat_long.yaml`: winner knobs at history 10 and 16 (2 runs, with accumulation).
- `ft_decisions_hier.yaml`: winner knobs on Architecture B at 5, 10, 16 (3 runs).

All use training files 1 to 10, validation files 4950 to 4954, 10 epochs maximum, seed 42. The evaluation
range in the base configs is file 8000 only; see Part 6.5.

If asked "how reproducible is the sweep?": every run is a config plus overrides checked into the repository;
`bash my_work/run_all.sh` reruns the 13 sweep runs; the 100-file confirmation and scoring are separate commands.

---

## 4.7 The results, and how to read them

**Finder at history 5** (validation PR-AUC, best epoch; one seed):

| embedding | mask | optimizer | validation PR-AUC | max validation ROC-AUC over epochs (may not be the AP checkpoint's epoch) | eval PR-AUC (range uncertain) |
|---|---|---|---:|---:|---:|
| linear | bidirectional | AdamW | **0.1365** | 0.9494 | 0.0893 |
| linear | bidirectional | RAdam | 0.1305 | 0.9485 | 0.0743 |
| linear | causal | AdamW | 0.1330 | 0.9507 | 0.0941 |
| linear | causal | RAdam | 0.1303 | 0.9456 | 0.0787 |
| periodic | bidirectional | AdamW | 0.1215 | 0.9463 | 0.1011 |
| periodic | bidirectional | RAdam | 0.1274 | 0.9444 | 0.0930 |
| periodic | causal | AdamW | 0.1224 | 0.9486 | 0.0801 |
| periodic | causal | RAdam | 0.1261 | 0.9438 | 0.0935 |

Averages by knob: linear 0.1326 vs periodic 0.1244 (gap 0.008); bidirectional 0.1290 vs causal 0.1280;
AdamW 0.1283 vs RAdam 0.1286. (The standing document's averages differ in the fourth decimal; these are
recomputed.)

**Long history at the winner knobs** and **the 100-file confirmation**:

| Run | Training | validation PR-AUC (epoch) | eval PR-AUC |
|---|---|---:|---:|
| history 5 | 10 files | 0.1365 | 0.0893 |
| history 10 (batch 1024 x 4) | 10 files | 0.1399 (6) | 0.0902 |
| history 16 (batch 256 x 16) | 10 files | 0.1383 (8) | 0.1171 |
| history 5 | 100 files | **0.1416** (1), stopped at epoch 6 | not yet scored |
| Ya’s history 5 defaults | 10 files | 0.1293 (5) | not reported |
| Ya’s single transaction | 100 files | 0.1247 (2) | 0.0940 (11 files) |

How to read them, honestly:

1. **History looks promising.** The best documented history result, 0.1416, is well above the documented
   100-file single-transaction result of 0.1247, and most history runs are above it. Not every one is: the two
   periodic/AdamW finder runs scored 0.1215 and 0.1224. The comparison also changes the encoder (hand-built,
   with a time-delta feature, different batch and optimizer details) and used the faulty loader, so the clean
   test is still owed: the same history model with history length 0, time delta on and off, and reruns on the
   corrected loader.
2. **Linear exceeded periodic in all four matched one-seed pairs** (average gap 0.008). With one seed this is
   "probable", not proven, and it is a statement about this periodic variant (4.5).
3. **Mask and optimizer: unresolved.** The gaps (0.001 and 0.0003) are too small for one-seed runs to call a
   real effect, no effect, or equivalence. For the mask there is an extra reason for caution: the causal option
   did not meet its contract (Part 6.3), so the ablation did not compare what it meant to compare; the two
   options still differ, so the bug does not prove they must perform the same.
4. **History 5, 10, 16: unresolved.** The gaps (0.0034 and 0.0017) are about the size of one model's own
   sampling wobble (Part 1.8), with one seed each and no paired test.
5. **The 100-file run peaked at epoch 1** and then declined for five epochs. That pattern (best epoch very
   early) deserves a look: with 10 times the data, one epoch is 21,000 updates, and the learning rate of
   3e-4 with no warm-up may be too aggressive at that scale.
6. **Evaluation PR-AUC columns are not comparable yet** (one file vs eleven, Part 6.5). A one-file number rests
   on 290 fraud cases, so the 0.1171 for history 16 is not evidence that 16 is best; confirm the cohort and
   recover the predictions before reading anything into those columns.

My best documented result: validation PR-AUC 0.1416 with five transactions of history on 100
training files, against 0.1247 for the single-transaction model on the same files (the two runs differ in more
than history; the controlled comparison is planned); evaluation scoring of that checkpoint is pending.

---

## 4.8 What is not known yet

- Architecture B results (sweep started, no rows).
- Whether 10 or 16 history beats 5 (needs seeds and a paired comparison).
- The history-0 control and the time-delta ablation from Ya’s plan (not documented).
- A capacity sweep for the history model (Ya’s decision 6; the Optuna tuner exists but has no recorded
  trials).
- The evaluation number for the 100-file checkpoint on the pinned 11-file cohort.

What would change my conclusion about history 10 vs 16: three to five seeds each and a paired
bootstrap by account on the same evaluation transactions; until then the question is open: the documented
scores are close, and that is all that can be said.
