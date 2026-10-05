# Part 0. The ten-minute version

Original chapter author: CL. Public adaptation: CH, in my first-person project voice.
The original author incorporated an independent CH review on 2026-10-05; this adaptation is not a new joint endorsement.

Status note: experiment status is the recorded 2026-10-04 snapshot; audit and proposed-fix statements describe the 2026-10-05 review. They are not live job or deployment status.
Source paths and commands are illustrative aliases, not files or runnable commands supplied by this public guide. `my_work/` means my implementation; `ya_work/` means Ya’s foundation; `s3://<bank-storage>/` replaces the actual storage address.

This is my short explanation when someone asks "what is TF?" Every sentence here is expanded later in the guide.

---

## The problem we are solving

The bank scores every debit-card transaction for fraud. The production model is an XGBoost tree model that
looks at one transaction at a time through hand-engineered features. Fraud is rare: about 1 transaction in 600
(0.165%). We are testing whether a **transformer**, the neural-network design behind modern language models,
can score fraud better by reading the raw transaction fields directly and by reading the account's recent
transaction history alongside the transaction being scored.

## Who did what

- **Ya** built the foundation: picked the 23 features, built frozen preprocessing so training and
  scoring treat data identically, built a streaming pipeline that trains on millions of rows from S3 without
  loading them into memory, trained the single-transaction model, and built the first history-aware model.
- **I** built on top without touching Ya’s code: a systematic sweep of the
  history model's design choices, a second "hierarchical" architecture for long histories, two encoder
  pretraining recipes, and a scoring toolkit (calibration, ensembles, leaderboard, drift view).
- Two AI assistants (CL and CH) independently reviewed the project within their recorded scopes on
  2026-10-05, found overlapping defects, and cross-check selected findings through a separate review workspace.

## What the models are

- **Single-transaction FT-Transformer** (Ya, Phases 1 to 3). Each of the 23 features (plus 6 missing-value
  flags, 29 tokens in all) becomes a small vector ("token"); a transformer lets every feature look at every
  other; a summary token is classified.
- **History-aware flat model** (Ya’s Phase 4, my Phase 4a). The target transaction plus its N previous
  transactions on the same account, each as 30 tokens, all in one transformer. N tried: 5, 10, 16.
- **Hierarchical model** (my Phase 4b, "Architecture B"). First summarize each transaction into one
  vector with a small transformer, then run a second transformer over the N+1 summaries. Its second-stage
  attention is much smaller; total memory and speed still need measurement. No result appears in the recorded status.
- **Pretraining** (my Phase 5). Train the encoder before the final fraud classifier is fitted. The masked
  autoencoder reconstructs hidden fields and uses no labels; supervised contrastive uses the fraud labels to
  decide which transactions to pull together. Written; no result is documented in the repository.

## The numbers, and what they mean

The metric is **PR-AUC** (average precision), because accuracy is useless at 0.165% fraud and ROC-AUC
saturates near 0.95 for every model. A random model scores about 0.0016. Higher is better.

| Model | Training data | Validation PR-AUC |
|---|---|---:|
| Single transaction (Ya) | 100 files | 0.1247 |
| History 5 (Ya’s defaults) | 10 files | 0.1293 |
| History 5, my best knobs | 10 files | 0.1365 |
| History 10, same knobs | 10 files | 0.1399 |
| History 16, same knobs | 10 files | 0.1383 |
| History 5, my best knobs | 100 files | **0.1416** |

Reading: the best documented history runs score higher than the documented single-transaction baseline
(0.1247 to 0.1416), which is promising evidence for history. It is not yet a clean comparison: the history
models also change the encoder and training details, and the documented multi-worker shuffled training
runs used a data loader with a known defect (below). Not every history setting beat the baseline: two of the eight finder runs scored 0.1215 and
0.1224. The gaps between history 5, 10 and 16 are small, single-seed and unresolved; neither a real difference
nor equivalence has been shown. Among my design knobs, the plain linear embedding beat the periodic one in
all four matched one-seed pairs; the attention-direction and optimizer gaps are small and unresolved. The
hierarchical results and the 100-file evaluation score are still pending.

## What the audits found

Both AI audits, independently, found the same main defects. All are in the original logic; the targeted checks
found no damage from the photo recovery (they do not certify every line against the original screenshots).

1. **Training data coverage.** With 4 data-loader workers and shuffling on, each worker shuffles the file list
   on its own before taking its share, so about a third of the training files are skipped in any given epoch
   and others are read twice. The documented multi-worker shuffled training runs used this path;
   zero-worker smoke runs and the in-memory baseline are exceptions. Unshuffled validation and evaluation are unaffected. A two-line fix exists and is tested in a disposable copy; it is not yet applied to the
   project code.
2. **The "causal" attention option is not causal.** Information flows from the target back into history via
   the summary token from the second layer on, so the option does not meet its contract and the
   causal-vs-bidirectional comparison is weakened (the two options still differ, so the bug does not prove
   they must score alike). Fix proposed and tested in a copy. The default (bidirectional) is unaffected.
3. **The supervised-contrastive pretraining loss pairs the wrong rows with the wrong labels** and would push
   same-class examples apart. No run is documented. One-line fix proposed, tested, and independently checked by
   CH; not yet applied.
4. **The masked-autoencoder pretraining cannot tell two hidden numeric fields apart.** No run is documented.
   Fix designed, not yet written.
5. **The evaluation range in the configs is one file (290 fraud cases)** while the documents quote eleven-file
   numbers (3,818 fraud cases). Which one produced the recorded evaluation scores needs the pod logs.
6. **Each sweep run overwrote the previous run's prediction file on S3**, so under the configured paths only
   the last-scored run's predictions survive (S3 versioning or backups were not inspected). Fix proposed and
   tested.
7. Smaller items: the saved checkpoint would not contain the averaged weights if EMA were turned on (it never
   was); the package does not install because of a typo-level error in `pyproject.toml`; the vendored library
   is one minor version older than Ya’s streaming code expects; the training-loss column in the status
   document is the last mini-batch, not the epoch average.

## What still stands

The best history runs beat the single-transaction baseline on validation, with the caveats above. The
validation numbers used the right metric, computed over the whole validation set on held-out files. The code
parses, its 28 offline tests pass, and targeted checks found no transcription damage. What is **not**
established yet: how much of the lift is history itself, that 10 or 16 history beats 5, that the flat
architecture is "saturated", that Architecture B wins, how much the loader defect moved the scores, or that
pretraining helps. Those need the fixes above, a same-architecture no-history control, several random seeds, a
fixed evaluation cohort, and the hierarchical results.

## What happens next

1. Review and apply the proposed fixes (data sharding first) before any new GPU run; they are tested in
   copies, not yet in the project code.
2. Pin the evaluation cohort to files 8000 to 8010 and record the exact ids.
3. Rerun the key configurations with 3 to 5 seeds: history 0, 5, 10, 16; flat vs hierarchical; time delta on
   and off.
4. Compare against the production XGBoost on the same transactions, and try the transformer's score as an
   extra XGBoost feature.
5. Hold out a later time period that nobody has looked at for the final number.

If asked for the one-sentence summary: "The best history-aware transformer runs lift validation PR-AUC from
about 0.125 to about 0.14 over a single-transaction model on this data; two independent audits found a
data-loading defect and several latent bugs, with proposed fixes under review; the next step is a seeded,
fixed-cohort rerun on the corrected loader, with a no-history control, before anyone declares an architecture
winner."
