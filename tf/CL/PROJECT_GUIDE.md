# TF: the complete project guide

> Public adaptation prepared by CH from CL’s dated 2026-10-05 documents. CL remains the original author; this adaptation does not imply CL approved its wording. I use **TF** for my project, **Ya** for my collaborator, **My** for my storage ownership and **Ya’s** for Ya’s storage ownership. Dataset counts, rates, scores and model details are retained with my approval.
>
> This is a documentation-only copy. Code paths such as `my_work/` and `ya_work/`, storage addresses, and references to private proofs are illustrative aliases or evidence descriptions—not files or commands available in this public folder. Test execution and findings are credited to the assistant that reported them. Status is historical as of the dated review.

Original author: CL, 2026-10-05. I use this public adaptation to explain what Ya built, what I built, how the pieces fit together and what our results do—and do not—show. CL based the original on a full project review and related bank documentation. The dated test results below are CL’s or CH’s reported work, not tests I am claiming to have rerun for this publication.

Companion documents:

- [Business use case](BUSINESS_USE_CASE.md): the short business explanation.
- [Audit findings](AUDIT_FINDINGS.md): defects, checks and review limitations.
- [Recommendations](RECOMMENDATIONS.md): the proposed next steps.
- [Open questions](../QUESTIONS.md): facts I still need to confirm.

Private source code, patches, proof scripts, raw logs and bank data are not included in this public documentation set.

---

## 1. One-paragraph summary

The bank wants a better fraud detector for debit-card transactions. The current production model is an
XGBoost (gradient-boosted trees) model that scores one transaction at a time from hand-engineered
features. This project tests whether a transformer, the neural-network architecture behind large
language models, can do better by (a) reading the raw transaction fields directly and (b) reading the
account's recent transaction history alongside the transaction being scored. Ya built the data pipeline and the first four phases: feature selection, preprocessing, a
single-transaction model, a streaming trainer for the full dataset, and a first history-aware model.
I own the work under `my_work/`: a systematic sweep of the history-aware model's design
knobs (Phase 4a), a second "hierarchical" architecture for long histories (Phase 4b), encoder
pretraining recipes (Phase 5), and a scoring toolkit. The best validation result so far is a PR-AUC of
0.1416 (history of 5, 100 training files), against Ya’s single-transaction benchmark of 0.1247.

---

## 2. The data

| Item | Value | Where it is defined |
|---|---|---|
| Training bucket | `s3://<bank-storage>/training/part_1.parquet` to `part_5000.parquet` | `ya_work/docs/fraud_detection_plan_detailed.md` |
| Evaluation bucket | `s3://<bank-storage>/evaluation/part_8000.parquet` to `part_10000.parquet` | same |
| Rows per file | roughly 840,000 (part_1 has 841,428) | `ya_work/notebooks/sequence_smoke.ipynb` |
| Accounts per file | roughly 3,300; each account is fully contained in one file | same; `ya_work/docs/phase4_design_decisions.md` |
| Transactions per account | median 133, 10th percentile 3, 90th percentile 670, max 2,296 (part_1) | `ya_work/notebooks/sequence_smoke.ipynb` |
| Label | `frd_tag`, 1 = fraud | `ya_work/fraud/configs/*.yaml` |
| Fraud rate | 0.1654% in training files 1 to 100; 0.155% in eval files 8000 to 8010 | `my_work/docs/my_roadmap.md` |
| Time column | `transaction_dttm` (orders the history and derives the four calendar features and the time gap; the raw timestamp itself is not a token) | `ya_work/fraud/configs/predictors.csv` |
| Account column | `Nmap_account_reference_xid` | same |
| Transaction id | `fi_transaction_id` (tie-break for equal timestamps) | `ya_work/fraud/configs/sequential.yaml` |

"Fraud rate 0.165%" means about 1 fraud in every 600 transactions. This drives most of the design: a
model that says "not fraud" for everything is 99.8% accurate and useless, so accuracy is unsuitable as the headline measure;
ranking metrics guide model selection instead (Section 7).

### 2.1 The 23 features (version 1 of `ya_work/fraud/configs/predictors.csv`)

A **feature** is one column the model is allowed to look at. Features come in two types. A
**continuous** feature is a number with magnitude (an amount). A **categorical** feature is a code
from a fixed set (a transaction type). The predictor list is a versioned CSV so the feature set can
grow without breaking older artifacts.

- Continuous (6): `transaction_amt`, `external_1_scr` (Visa's own fraud score), three daily-limit
  amounts, `total_available_balance_amt`.
- Categorical, low cardinality (13): transaction type, segment, category code, POS condition, AVS
  request, POS security flag, PIN verification, cardholder present, card capture, CVV result,
  cryptogram validity, off-premises flag, unattended-terminal flag.
- Datetime-derived (4, categorical): `month`, `day`, `hour`, `day_of_week`, computed from
  `transaction_dttm`.

Deliberately excluded: all identifiers, the raw timestamp, high-cardinality fields (merchant id,
postal codes, device IP), and the historical fraud label of earlier transactions (Section 5.2).

---

## 3. My working vocabulary

- **Model**: a function with adjustable numbers (weights) that maps inputs to a fraud score.
  Training adjusts the weights to make scores high on known fraud and low elsewhere.
- **Embedding / token**: neural networks work on vectors of numbers. Each feature value is turned into
  a vector of fixed length `d` (here 32 to 64). That vector is a token. A categorical value gets a
  learned lookup vector; a continuous value `x` gets `x * w + b` with learned `w, b` (a "linear"
  embedding) or a sine/cosine expansion (a "periodic" embedding, Section 6.2).
- **Transformer / attention**: a stack of layers in which every token looks at every other token and
  decides how much to borrow from each. This is what lets the model relate "amount" to "hour" to
  "the previous five transactions" without anyone hand-coding those interactions.
- **CLS token**: an extra learned token added to the sequence. After the transformer layers, its
  vector is read out and fed to a small linear layer that produces the fraud score. It is the model's
  summary slot.
- **FT-Transformer** (Gorishniy et al., 2021): "Feature Tokenizer plus Transformer." Each feature
  becomes one token, a CLS token is added, a transformer runs, the CLS output is classified. The team
  chose it after a benchmark on a public bank dataset (Section 4.0).
- **PyTorch Lightning**: a training framework. It runs the loop (forward, loss, backward, optimizer
  step), validation, checkpointing and early stopping. The project's models are Lightning modules.
- **Epoch**: one pass over the training data. **Batch**: the rows processed per step (4,096 here).
- **Validation set vs evaluation set**: validation (files 4950 to 4954, from the training bucket) is
  used during training to pick the best epoch and to compare designs. Evaluation (files 8000 and up,
  a separate bucket) is meant to be looked at rarely, as the final honest number.
- **Early stopping**: stop training when the validation metric has not improved for `patience`
  epochs (5 here) and keep the best epoch's weights.
- **Checkpoint**: a saved copy of the weights (`best.ckpt`).

---

## 4. Ya’s work (the `ya_work/fraud/`, `ya_work/docs/`, `ya_work/src/`, `ya_work/tests/`, `ya_work/notebooks/` trees)

### 4.0 Phase 0: choosing the architecture

Before touching bank data, four models were compared on the public UCI Bank Marketing dataset
(`ya_work/docs/tabular_transformer_models.md`). FT-Transformer from the `pytorch_tabular` library won (test
AUROC 0.935) over a plain MLP (0.921) and two SAINT variants (0.82 to 0.85). No tuning was done; the
ranking was taken as directional. Decision: build on `pytorch_tabular`'s FT-Transformer and fork the
library (`ya_work/src/pytorch_tabular`, pinned to upstream v1.1.0 plus the stacking model from upstream main)
because the planned history-aware model would need changes the library does not support.

Note for precision: the fork is unmodified upstream code. The `ft_transformer_seq` package it was
forked for is an empty placeholder with a TODO. The history-aware models were instead written from
scratch with PyTorch's `nn.TransformerEncoder` (Section 5.3). Only the single-transaction models of
Phases 1 to 3 use the library's `FTTransformerModel`.

### 4.1 Phase 1: predictors, preprocessing, artifacts

- `ya_work/fraud/data/feature_list.py` loads the predictor CSV into typed lists.
- `ya_work/fraud/data/datetime_features.py` derives month/day/hour/day_of_week (categorical by default; a
  sine/cosine "cyclical" mode exists but is unused).
- `ya_work/fraud/data/preprocessing.py` has two pieces. `ContinuousImputer` (median fill plus a `<col>_isna`
  flag) serves the in-memory baseline. `PreprocessingArtifacts` is the frozen, versioned statistics
  store for everything after: per continuous feature it stores mean, std, median; per categorical
  feature it stores a vocabulary with reserved indices `0 = unknown`, `1 = missing`, known values from
  2 upward. Artifacts are JSON files on S3 under `.../artifacts/<version>/`.
- `ya_work/fraud/scripts/build_preprocessing_artifacts.py` fits them on files 1 to 5 in full.
- Key finding recorded in the plan: `pytorch_tabular` does not impute continuous NaNs on its own, so
  imputation had to be done upstream. Decision: median plus missing flag.

### 4.2 Phase 2: the in-memory baseline

`ya_work/fraud/scripts/train_baseline.py` loads a few parquet files into pandas, runs the imputer, hands the
frame to `pytorch_tabular.TabularModel` with `FTTransformerConfig` (embed 32, 8 heads, 3 blocks,
roughly 142K weights), trains one epoch, then computes PR-AUC and ROC-AUC over the whole validation and
test splits with scikit-learn. `ya_work/fraud/scripts/score_eval.py` re-scores a saved model file by file.

Result (2026-09-03, H200 GPU): validation ROC-AUC 0.9275 / PR-AUC 0.1051 on `part_4950`; test ROC-AUC
0.9521 / PR-AUC 0.0940 on `part_8000`. PR-AUC of 0.094 is roughly 60 to 75 times the fraud prevalence,
which is what a random model would score.

### 4.3 Phase 3: streaming at scale

Thousands of files cannot fit in memory and `TabularModel.fit` needs a DataFrame, so Phase 3
(`ya_work/docs/phase3_streaming_plan.md`) bypasses it:

- `ya_work/fraud/data/parquet_loader.py` (`ParquetStreamingDataset`) is a PyTorch `IterableDataset`. Each
  DataLoader worker takes a slice of the file list, reads parquet in 65,536-row blocks, applies the
  frozen artifacts (impute, standardize, encode), shuffles rows within the block, and yields whole
  batches as tensors.
- `ya_work/fraud/streaming_model.py` builds the library's `FTTransformerModel` by hand-constructing the
  `InferredConfig` (dimensions and cardinalities) that `TabularModel.fit` would normally infer.
- `ya_work/fraud/callbacks.py` has `StreamingBinaryMetrics`, which accumulates every validation prediction and
  computes PR-AUC/ROC-AUC once per epoch (per-batch metrics are meaningless at 0.15% prevalence), and
  `CSVMetricsCallback`, which writes one row per epoch to `training_metrics.csv`.
- `ya_work/fraud/scripts/train_streaming.py` and `score_streaming.py` run it; `show_progress.py` prints the CSV.

Results (2026-09-09): 100 files, early-stopped: best validation PR-AUC 0.1247 (epoch 2), ROC-AUC
0.9493; eval on files 8000 to 8010 (2,466,461 rows, 3,818 fraud): PR-AUC 0.0940, ROC-AUC 0.9514.
Findings: shuffling rows inside each read block matters (without it, batch 512 collapsed to ROC-AUC
0.5); more data helps modestly (0.11 to 0.125 from 10 to 100 files); metrics plateau by epoch 2 to 5,
which suggests (but does not prove) that the single-transaction model is close to **feature-limited**;
optimization settings or data coverage can also cause a plateau. Ya’s conclusion: the next lever is history.

### 4.4 Phase 4: the history-aware model (Architecture A, "flat")

Design (`ya_work/docs/phase4_sequential_plan.md`, `ya_work/docs/phase4_design_decisions.md`, dated 2026-09-25):

| # | Decision | Value |
|---|---|---|
| 1 | Architecture v1 | flat token sequence; hierarchical deferred |
| 2 | Historical fraud label as input | excluded (may not be known at inference time) |
| 3 | Sample | target = transaction i; history = the N transactions strictly before it, same account; N = 5, stride 1; never skip a fraud target |
| 4 | Time | add time-delta features |
| 5 | Short histories | left-pad with PAD slots plus a mask |
| 6 | Capacity | embed 48 to 64, 4 blocks, then sweep |
| 7 | Attention direction | `bidirectional` default; `causal` option for a future "score every transaction at once" mode |
| 8 | Position | one learned embedding per transaction slot, added to all of that transaction's tokens; none within a transaction |
| 9 | Correlation | windows of one account overlap heavily; use a 65,536-window shuffle buffer and large batches |
| 10 | Stride | `train_stride` may exceed 1; `eval_stride` is always 1 |
| 11 | Memory | bf16, start at batch 4096, watch GPU memory |

Implementation:

- `ya_work/fraud/data/sequence_builder.py` (`SequenceBuilder`): per file, encode every row once with the
  artifacts; group rows by account; sort by time then transaction id; for each target emit a window
  of `T = N + 1` slots with the target last; left-pad; add `time_delta = log(1 + seconds before the
  target)` as a seventh continuous feature; yield tensors `continuous [B, T, 7]`, `categorical
  [B, T, 23]`, `mask [B, T]`, `target [B, 1]`. The 23 categoricals are 17 artifact categoricals plus
  6 missing flags.
- `ya_work/fraud/sequential_model.py` (`FTTransformerSeqModel`): tokenize every slot's 30 features (7
  continuous + 23 categorical) into 30 tokens, add the slot's position embedding, flatten all slots
  into one sequence of `1 + T * 30` tokens with CLS first, run `num_attn_blocks` pre-norm
  `nn.TransformerEncoderLayer`s (GELU feed-forward, 4x width), mask padded slots' tokens, read CLS,
  classify with a 2-way linear head and cross-entropy. For N = 5 the sequence is 181 tokens; N = 10
  gives 331; N = 16 gives 511.
- `ya_work/fraud/scripts/train_sequential.py` and `score_sequential.py`, config `ya_work/fraud/configs/sequential.yaml`.
- Verified in `ya_work/notebooks/sequence_smoke.ipynb` on real data: 28,054 windows from 30 accounts, every
  transaction is a target exactly once, padding is left-aligned, time deltas are zero at the target and
  non-increasing toward it, and one account's window matched a manual pandas reconstruction.

Preliminary result (2026-09-28): history 5, 10 files, 10 epochs, early-stopped: best validation
PR-AUC **0.1293** (epoch 5), ROC-AUC 0.9509, versus 0.1131 for the single-transaction model at the
same scale. This is promising for the history-aware model, but it does not isolate history from the encoder and training changes; I still need the history-0 control.

Naming note: the plans and some docstrings speak of "SEP tokens" between transactions. The implemented
model has none; transaction boundaries are carried by the per-slot position embeddings and the fixed
token count per transaction. Both audits classify this as a naming discrepancy, not a defect.

### 4.5 Ya’s documentation set

`ya_work/docs/fraud_detection_plan_overall.md` (goals, why fork), `fraud_detection_plan_detailed.md` (features,
preprocessing, smoke results), `phase3_streaming_plan.md`, `phase4_sequential_plan.md`,
`phase4_design_decisions.md`, `monitoring_and_eval.md` (how to watch and score runs),
`tabular_transformer_models.md` (Phase 0 benchmark). `ya_work/tests/test_preprocessing.py` covers the feature
list, datetime derivation, imputer and artifact round trip (6 tests).

---

## 5. My work (the `my_work/` tree)

### 5.1 My rule: preserve Ya’s code

`my_work/docs/my_roadmap.md` sets the rule. Anything of Ya’s used unchanged (data loaders,
`SequenceBuilder`, preprocessing, `StreamingBinaryMetrics`, `CSVMetricsCallback`, logging) is
imported from `ya_work/fraud/`. Anything changed is copied into the phase folder that uses it, so `phase4a/`
and `phase4b/` each carry their own `tokenizer.py`, `numerical_embeddings.py`, `optim.py`,
`callbacks.py`, `factory.py`. CL’s historical review found no changes by me under `ya_work/fraud/`. The
copies were compared line by line in this audit: they are identical apart from import paths and
docstrings.

The project is framed as one model improved along five levers (roadmap table): architecture (A flat vs
B hierarchical), input representation (linear vs periodic numerical embedding), context (history 5,
10, 16; time encoding), training recipe (optimizer, EMA, sampling), and pretraining (from scratch vs
masked autoencoder vs supervised contrastive). The baseline to beat is the internal XGBoost model.

### 5.2 Phase 4a: the flat model with knobs

`my_work/phase4a/sequential.py` is Ya’s flat model with three additions: the tokenizer can use a
periodic numerical embedding (`numerical_embeddings.py`, after Gorishniy et al. 2022), the optimizer is
selectable (`optim.py`: AdamW, Adam, NAdam, RAdam, RMSprop, SGD; Muon/Lion/SOAP/Adan if
`pytorch-optimizer` is installed), and an optional weight-EMA callback (`callbacks.py`). `factory.py`
builds the model from config. `scripts/train_sequential.py` and `score_sequential.py` are Ya’s with
the factory swapped in (TensorBoard logging was dropped). `scripts/tune.py` is an Optuna search harness
(unused so far). `scripts/run_ablations.py` is the sweep driver: it reads a spec file with named axes
of config overrides, takes their cartesian product, and for each combination trains, reads the best
`valid_ap` from the metrics CSV, scores the checkpoint on the eval range, parses PR-AUC/ROC-AUC from the
log with a regular expression, and appends a row to `results.csv`.

Sweep 1, `configs/ft_decisions.yaml`: embedding {linear, periodic} x mask {bidirectional, causal} x
optimizer {AdamW, RAdam}, 8 runs, history 5, files 1 to 10, validation 4950 to 4954, 10 epochs, batch
4096, seed 42. Results (`my_work/docs/my_current_standing.md`):

| embedding | mask | optimizer | best valid PR-AUC | eval PR-AUC |
|---|---|---|---:|---:|
| linear | bidir | adamw | **0.1365** | 0.0893 |
| linear | bidir | radam | 0.1305 | 0.0743 |
| linear | causal | adamw | 0.1330 | 0.0941 |
| linear | causal | radam | 0.1303 | 0.0787 |
| periodic | bidir | adamw | 0.1215 | 0.1011 |
| periodic | bidir | radam | 0.1274 | 0.0930 |
| periodic | causal | adamw | 0.1224 | 0.0801 |
| periodic | causal | radam | 0.1261 | 0.0935 |

Reading: averaged over the other knobs, linear beats periodic by 0.008; mask and optimizer differ by
under 0.001. Winner locked: linear + bidirectional + AdamW. Two finder rows (periodic/AdamW, 0.1215 and 0.1224) sit
below the single-transaction 0.1247, so not every history setting beat the baseline. (The audit explains why
the mask knob did not test what it meant to: Section 6.3 and [audit findings](AUDIT_FINDINGS.md); the 0.001 gap is
unattributed, not explained.)

Sweep 2, `configs/ft_decisions_flat_long.yaml`: winner knobs, history 10 and 16. Because the flat
sequence grows to 331 and 511 tokens and attention cost grows with the square of that, batch 4096 ran
out of GPU memory; the runs use batch 1024 x 4 accumulation steps and 256 x 16, which keeps the
effective batch at 4096.

| history | best valid PR-AUC | eval PR-AUC |
|---|---:|---:|
| 5 | 0.1365 | 0.0893 |
| 10 | **0.1399** | 0.0902 |
| 16 | 0.1383 | 0.1171 |

Confirmation run: history 5, winner knobs, files 1 to 100: best valid PR-AUC **0.1416** at epoch 1,
early-stopped at epoch 6. Scoring on the eval set had not been run at the time of the standing document.

My reproduction of Ya’s setup (Ya’s defaults, 10 files): 0.1333 (epoch 4), compared with Ya’s 0.1293.

### 5.3 Phase 4b: the hierarchical model (Architecture B)

This is Ya’s deferred "approach (b)", implemented in `my_work/phase4b/sequential_hier.py`
(`FTTransformerHierModel`). Section 6.1 explains it in depth. Configs: `sequential_hier.yaml` (base,
2 transaction-level blocks + 4 sequence-level blocks, embed 48) and `ft_decisions_hier.yaml` (winner
knobs, history 5 / 10 / 16, full batch 4096, no accumulation requested in the config; completed-run memory performance is not yet documented). Status at the last snapshot: the
first run (`history-h5`) started at 2026-10-04 23:09; no results yet.

### 5.4 Phase 5: pretraining the encoder

`my_work/phase5_pretrain/` holds single-transaction encoder pretraining (no history):

- `encoder.py` (`TabularEncoder`): the same tokenizer + CLS + pre-norm transformer, returning the CLS
  vector as a transaction embedding.
- `ssl_pretrain.py` (`MaskedTabularAutoencoder`): hide 30% of a transaction's feature tokens behind a
  learned mask token and train the encoder to reconstruct them (mean squared error for continuous,
  cross-entropy for categorical). Uses no labels, so it can learn from every transaction.
- `ssl_finetune.py` (`SSLFinetuneClassifier`): the same encoder plus a 2-way head; loads the
  pretrained encoder weights, trains on labels; `freeze_encoder` trains only the head (encoder dropout
  stays active as written, so it is not yet a deterministic linear probe).
- `supcon.py` (`SupConPretrainer`, `SupConLoss`): supervised contrastive pretraining (Khosla et al.
  2020). Two corrupted views of each transaction (30% of features swapped with another row's values,
  SCARF-style) are embedded and projected; same-label views are pulled together, different-label
  views pushed apart. The audit found the loss implementation misaligned (proposed fix in the private patch collection, independently
  checked by CH, not applied).
- Scripts: `pretrain_ssl.py`, `train_ssl_finetune.py`, `score_ssl_finetune.py`,
  `pretrain_contrastive.py`, `train_contrastive.py`, `score_contrastive.py`, `export_embeddings.py`.
  Configs `ssl.yaml` (files 1 to 100, embed 64, 4 blocks, batch 8192) and `contrastive.yaml`.

No Phase 5 result is documented in the repository snapshot (a pod-side run cannot be excluded). The phase
guide itself proposes a balanced sampler for SupCon (at 0.16% fraud, batch 8192 contains about 13 fraud rows);
after the alignment fix each transaction's two views already give it one positive, so the sampler is an
efficiency experiment, not a prerequisite. Note also that SupCon uses the fraud labels; only the masked
autoencoder is label-free.

### 5.5 Scoring toolkit (`my_work/scoring/`)

All four read prediction parquet files (`id, frd_tag, score`):

- `calibrate_and_report.py`: fit isotonic or Platt calibration on validation predictions, apply to
  test, report PR-AUC, ROC-AUC, Brier score, and precision/recall at alert budgets (top 0.1%, 0.5%,
  1%).
- `ensemble.py`: rank-average several models' scores with weights searched on validation.
- `leaderboard.py`: ranked table of models with bootstrap 95% confidence intervals and a seeds column.
- `temporal_eval.py`: bucket predictions by month/week/day to see drift. It is a view of existing
  predictions, not a train/holdout protocol.

### 5.6 Tests (`my_work/tests/`)

22 tests, all passing in this audit (CPU, torch 2.14.1, Lightning 2.6.6): forward/backward for the
flat model with both embeddings and both masks, the hierarchical model, the encoder, the masked
autoencoder, finetune weight loading, the optimizer factory, the EMA callback, SupCon loss shape and
gradients, and nine tests of `SequenceBuilder` (shapes, every transaction scored once, target slot
last, no future leakage via time delta, contiguous left padding, account isolation, empty first
history, fraud never skipped by stride, window count). Ya’s 6 preprocessing tests also pass.

### 5.7 How the code got here

After my laptop broke, my code was reconstructed from photographs of my screen. CH helped transcribe the images; the private reconstruction records describe a subsequent pixel-level audit process. CL’s audit found no transcription-specific failure in its parsing, import, test and duplicate-module checks. Neither assistant re-inspected all original photographs, so these checks do not certify every reconstructed line. The
Phase 5 files were pixel-verified; Phase 4a/4b files were not re-audited to that standard, but their
shared modules match the pixel-verified Phase 5 copies.

---

## 6. The two architectures, explained

### 6.1 Flat (A) versus hierarchical (B)

Both receive the same window: the target transaction plus its N prior transactions, each as 30 tokens.

**Flat (A).** All `(N + 1) * 30` tokens plus CLS go into one transformer. Every token can attend to
every other token: the amount of transaction 3 can look directly at the hour of transaction 1. Cost of
attention grows with the square of the sequence length: 181 tokens at N = 5, 331 at N = 10, 511 at
N = 16, which is why N = 16 needed a 16x smaller per-step batch.

```
[CLS] [txn-5: 30 tokens] [txn-4: 30 tokens] ... [txn-1: 30 tokens] [target: 30 tokens]
  |______________________ one transformer, 4 blocks, every token sees every token ____|
                                   |
                              CLS -> Linear -> fraud / not fraud
```

**Hierarchical (B).** Stage 1: each transaction is encoded on its own by a shared small transformer
(2 blocks) over its 30 tokens plus a per-transaction CLS, giving one 48-dimensional vector per
transaction. Stage 2: the N + 1 vectors, plus position embeddings and a sequence CLS, go through a
second transformer (4 blocks). Padded slots are still encoded in stage 1 (they are all-zero inputs)
and masked out in stage 2. Attention cost in stage 2 grows with `(N + 2)^2`, which is small; the committed B sweep requests
batch 4,096 at N = 16 without accumulation, but no completed B run is documented, so memory and cost at
longer histories still need measurement. The price is that a field of one
transaction can no longer attend directly to a field of another; it must go through the per-transaction
summary vector first.

```
txn-5 tokens -> [stage-1 transformer] -> v5
txn-4 tokens -> [stage-1 transformer] -> v4        (same weights for every transaction)
...
target tokens -> [stage-1 transformer] -> v0
                 [CLS] v5 v4 v3 v2 v1 v0  -> [stage-2 transformer, 4 blocks] -> CLS -> Linear
```

This mirrors how the bank's Payment Foundation Model (a related bank model repository) treats one transaction as
one unit, and it is the reason my roadmap gives long histories to B.

### 6.2 Linear versus periodic numerical embeddings

The linear embedding turns a standardized amount `x` into the vector `x * w + b`. The periodic
embedding (Gorishniy et al. 2022) turns `x` into `[sin(2 pi c_k x), cos(2 pi c_k x)]` for 16 learned
frequencies `c_k`, then a linear layer. Periodic embeddings help when the relationship between a number
and the target is wiggly rather than monotone. In the sweep, linear won by 0.008 PR-AUC. The audit
notes two implementation details that may explain this (one shared projection across all seven
numeric channels, the six features plus time delta, instead of one per feature; no ReLU after it; and an
untuned frequency scale), see [recommendations](RECOMMENDATIONS.md).

### 6.3 Bidirectional versus causal attention

Bidirectional: every token sees every token. This is safe for this task because the only label used is
the target's, and all history features are legitimately known at scoring time. Causal was meant to be
"block-causal": a transaction's tokens may see earlier transactions but not later ones, as groundwork
for a future mode that scores every transaction in a window with one forward pass. The audit shows the
mask as written lets every token read the CLS token, and CLS has already read the target, so from the
second layer on the target leaks backward into history anyway. So the causal option does not meet its contract. It is
still not identical to the bidirectional model (the attention graphs differ, and CH measured different
logits from the same weights), so the 0.001 gap is unattributed rather than explained. A corrected mask is
proposed in the private patch collection.

---

## 7. How success is measured

- **PR-AUC** (average precision, `valid_ap` / `eval_ap`): the area under the precision-recall curve.
  At 0.16% prevalence a random model scores about 0.0016; the single-transaction model scores 0.125 on
  validation; the best history model 0.1416. This is the model-selection metric everywhere in the
  project, by design.
- **ROC-AUC**: the chance a random fraud scores above a random non-fraud. It is 0.93 to 0.95 for
  every model here and barely moves; it is reported but not used for decisions.
- **Computed over the full split**, never per batch: `StreamingBinaryMetrics` accumulates every
  validation prediction and computes once per epoch.
- **Early stopping** on `valid_ap` with patience 5; the saved `best.ckpt` is the best epoch.
- **How precise are these numbers?** A published binomial-style approximation for the standard error of
  one model's AP is `sqrt(AP * (1 - AP) / number_of_fraud_cases)` (Boyd, Eng and Page, 2013). It is an
  illustrative marginal approximation, not a project confidence interval: its coverage is unvalidated for
  correlated account windows, it is not an interval for the difference between two models, and it ignores
  training-seed variance. On the 5 validation files (about 6,600 fraud cases) it gives about 0.004; on a
  single eval file (290 fraud cases) about 0.017. The history 10 versus history 16 gap on validation
  (0.0017) and the history 5 versus 10 gap (0.0034) remain unresolved without seeds and a paired, account-blocked bootstrap on the same identified
  predictions. The illustrative formula does not establish statistical equivalence or the significance of either gap. Linear exceeded periodic in all four matched one-seed pairs (0.008 average gap), which is
  plausible but unproven. Any eval PR-AUC that came from one file rests on 290 fraud cases; confirm the
  cohort before reading anything into differences between eval columns.
- **Validation versus eval discipline**: the standing document correctly refuses to select on eval
  PR-AUC because the 11-file eval set has been looked at repeatedly. Note that every `my_work/` config
  sets `eval.file_end: 8000` (one file), so unless a run overrode it, the sweep's eval numbers are
  single-file numbers.

---

## 8. Where things stand (as of the 2026-10-04 23:09 snapshot)

1. Flat model knob sweep: done; winner linear + bidirectional + AdamW at 0.1365.
2. Flat long history: done; h10 0.1399, h16 0.1383 (close; not resolved at one seed).
3. Flat h5 at 100 files: trained, best 0.1416 at epoch 1; eval scoring not documented.
4. Hierarchical sweep h5/h10/h16: started 2026-10-04 23:09; no rows yet.
5. Next per the standing document: score the 100-file checkpoint; finish B; pick the architecture and
   history; run the winner at 100 files; a temporal holdout confirmation.
6. Not documented from Ya’s verification plan: the history 0 versus 5 ablation, the time-delta on/off
   ablation, the leakage check, and the capacity sweep. No Phase 5 pretraining result is documented.

The audit's view on what to do first is in [recommendations](RECOMMENDATIONS.md): fix the data sharding defect
before any further training, because it affects the documented multi-worker training configurations.

---

## 9. My operational workflow

I keep executable commands in the private implementation. The public paths are aliases, so this documentation cannot be run as a checkout of the training system.

1. I build or verify the frozen preprocessing artifacts.
2. I run the single-transaction baseline and record its checkpoint, validation metrics and evaluation cohort.
3. I run the flat, long-history and hierarchical sweeps with a resolved configuration saved for every run.
4. I run Phase 5 pretraining and fine-tuning only after the relevant defects are fixed and independently checked.
5. I score each selected checkpoint into its own retained prediction artifact.
6. I run the offline test suite and targeted regression checks before committing a training budget.

The private configuration system accepts explicit overrides such as `sequence.history_len=10` and `training.batch_size=1024`. Those values must be recorded in the run manifest.

---

## 10. Illustrative path glossary

| Path | Owner | Role |
|---|---|---|
| `ya_work/fraud/configs/predictors.csv` | Ya | the 23 features, versioned |
| `ya_work/fraud/data/preprocessing.py` | Ya | imputer and frozen artifacts |
| `ya_work/fraud/data/parquet_loader.py` | Ya | streaming single-transaction dataset |
| `ya_work/fraud/data/sequence_builder.py` | Ya | streaming history-window dataset |
| `ya_work/fraud/streaming_model.py` | Ya | builds the library FT-Transformer for streaming |
| `ya_work/fraud/sequential_model.py` | Ya | flat history model (original) |
| `ya_work/fraud/callbacks.py` | Ya | full-stream metrics, CSV metrics |
| `ya_work/fraud/scripts/*.py` | Ya | train/score/build/progress scripts |
| `ya_work/src/pytorch_tabular/` | upstream | unmodified library fork, v1.1.0 |
| `my_work/phase4a/` | My work | flat model with knobs, sweep driver, tuner |
| `my_work/phase4b/` | My work | hierarchical model |
| `my_work/phase5_pretrain/` | My work | masked autoencoder, SupCon, finetune, export |
| `my_work/scoring/` | My work | calibration, ensemble, leaderboard, drift |
| `my_work/tests/` | My work | 22 offline tests |
| `my_work/docs/` | My work | roadmap, phase guides, current standing |
| `reconstruction/` | process | photo-to-text provenance records |
| CL’s private review | CL | source of this public adaptation |
