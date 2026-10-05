# Part 8. One-page cheat sheet

Original chapter author: CL. Public adaptation: CH, in my first-person project voice.
The original author incorporated an independent CH review on 2026-10-05; this adaptation is not a new joint endorsement.

Status note: experiment status is the recorded 2026-10-04 snapshot; audit and proposed-fix statements describe the 2026-10-05 review. They are not live job or deployment status.
Source paths and commands are illustrative aliases, not files or runnable commands supplied by this public guide. `my_work/` means my implementation; `ya_work/` means Ya’s foundation; `s3://<bank-storage>/` replaces the actual storage address.

## The project in three lines

Debit-card fraud scoring at 0.165% prevalence. Baseline: production XGBoost on one transaction. Test:
FT-Transformer on raw features, with and without the account's last N transactions. Metric: PR-AUC
(average precision) on held-out files; higher is better; random is about 0.0016.

## Numbers I use to explain the project

| What | Value |
|---|---|
| Fraud rate, training files 1 to 100 | 0.1654% (143,338 of 86,683,008) |
| Fraud rate, eval files 8000 to 8010 | 0.155% (3,818 of 2,466,461) |
| Rows per parquet file | about 840,000; about 3,300 accounts; median 133 transactions per account |
| Features | 23 (6 numeric, 13 category codes, 4 calendar) + 6 missing flags = 29 tokens per transaction; + 1 time gap = 30 in the history models |
| Single-transaction model, 100 files | validation PR-AUC 0.1247, ROC-AUC 0.949; eval (11 files) 0.0940 |
| History 5, Ya’s defaults, 10 files | 0.1293 |
| History 5, best knobs, 10 files | 0.1365 (eval, range uncertain, 0.0893) |
| History 10, 10 files | 0.1399 (eval 0.0902) |
| History 16, 10 files | 0.1383 (eval 0.1171) |
| History 5, best knobs, 100 files | 0.1416 at epoch 1 (eval not yet scored) |
| Illustrative marginal standard error of one model's PR-AUC (binomial approximation; not a project interval; not for differences between models) | about 0.004 on validation (about 6,600 fraud cases); eval one file about 0.017 (290 cases) |
| Hierarchical (Architecture B) | started 2026-10-04 23:09, no results yet |
| Pretraining (Phase 5) | written; no result documented; two objective bugs with proposed repairs (one tested, one designed), none applied |
| Model size | about 1 to 2 x 10^5 weights; embed 48, 8 heads, 4 blocks (B: 2 + 4) |
| Training | AdamW, lr 3e-4, wd 1e-5, batch 4,096, bf16, 10 epochs max, early stop patience 5 on valid PR-AUC, seed 42 |
| Flat sequence length | 1 + (N+1) x 30 tokens: 181 (N=5), 331 (N=10), 511 (N=16) |

## Vocabulary in one breath each

- **Token**: a feature value turned into a vector of 48 learned numbers.
- **Attention**: every token takes a weighted look at every other token; weights learned, input-dependent.
- **CLS**: an extra token that collects the summary; the classifier reads it.
- **FT-Transformer**: Feature Tokenizer + Transformer + CLS head (Gorishniy et al., 2021).
- **Flat (A)** vs **hierarchical (B)**: all tokens of all transactions in one transformer, vs one vector per
  transaction then a transformer over the vectors.
- **Position embedding**: tells tokens which transaction slot they belong to. **Time delta**: log seconds
  before the target.
- **Bidirectional / causal mask**: whether tokens may look at later transactions; bidirectional is the
  safe default here.
- **PR-AUC / AP**: average precision, precision averaged over recall levels; rewards a clean top of the
  ranking; show precision at the alert budget beside it. **ROC-AUC**: probability a fraud outranks a legit;
  near 0.95 for every model here.
- **Frozen artifacts**: training-time medians, means, standard deviations and category vocabularies
  (0 = unknown, 1 = missing) applied identically everywhere.
- **Early stopping / checkpoint**: stop when validation PR-AUC stalls for 5 epochs; keep the best weights.
- **SSL / masked autoencoder**: learn by reconstructing hidden fields, no labels. **SupCon**: learn by pulling
  same-label examples together.

## The audits in one breath

Two independent AI audits agree: the code parses and its 28 offline tests pass; targeted checks found no
transcription damage; the best history runs beat the baseline, with controls still owed. Defects: multi-worker
file sharding skips about a third of files per epoch (multi-worker shuffled training; unshuffled validation unaffected; two-line fix
proposed and tested, not yet applied); "causal" mask leaks through CLS (default unaffected; ablation
weakened); SupCon label misalignment and masked-SSL field identity (no pretraining result documented); eval
range one file vs eleven; sweep overwrote prediction files; package does not install; vendored library one
version too old. Agreed order: fix loader, pin environment, pin cohort, unique prediction paths, then rerun
with seeds and a no-history control.

## Where things are

| Path | What |
|---|---|
| `ya_work/configs/predictors.csv` | the 23 features |
| `ya_work/data/sequence_builder.py` | builds history windows (Ya) |
| `ya_work/sequential_model.py` | Ya’s flat history model |
| `my_work/phase4a/` | flat model with knobs, sweep driver, configs |
| `my_work/phase4b/sequential_hier.py` | hierarchical model |
| `my_work/phase5_pretrain/` | masked autoencoder, SupCon, finetune |
| `my_work/scoring/` | calibration, ensemble, leaderboard, drift |
| `my_work/docs/my_current_standing.md` | the results snapshot (2026-10-04 23:09) |
| `ya_work/docs/phase4_design_decisions.md` | Ya’s 11 design decisions |
| Public assistant bundles | separate CL and CH documentation; this learning guide belongs to the CL bundle |

## Illustrative commands behind my experiments

```bash
python -m pytest my_work/tests tests -q                       # 28 offline tests
python my_work/phase4a/scripts/run_ablations.py --config my_work/phase4a/configs/ft_decisions.yaml   # 8-run knob sweep
python my_work/phase4b/scripts/run_ablations.py --config my_work/phase4b/configs/ft_decisions_hier.yaml  # B sweep
python my_work/phase4a/scripts/score_sequential.py --config my_work/phase4a/configs/sequential.yaml \
    eval.file_start=8000 eval.file_end=8010 eval.checkpoint=<path>/best.ckpt                          # 11-file eval
```

## Three sentences to open with

"We built a history-aware FT-Transformer for debit-card fraud on top of Ya’s frozen preprocessing and
streaming pipeline. On held-out validation our best runs lift PR-AUC from about 0.125 for a single-transaction
model to about 0.14 with five prior transactions; two independent audits confirmed the history window uses
only preceding transactions and no labels, and found a data-loader sharding bug plus several latent issues,
with proposed fixes under review. Next is a seeded rerun on the corrected loader and a pinned evaluation
cohort, with a no-history control, the hierarchical architecture and a head-to-head with production XGBoost."
