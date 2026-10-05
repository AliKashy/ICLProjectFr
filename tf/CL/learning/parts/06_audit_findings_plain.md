# Part 6. What the audits found, in plain words

Original chapter author: CL. Public adaptation: CH, in my first-person project voice.
The original author incorporated an independent CH review on 2026-10-05; this adaptation is not a new joint endorsement.

Status note: experiment status is the recorded 2026-10-04 snapshot; audit and proposed-fix statements describe the 2026-10-05 review. They are not live job or deployment status.
Source paths and commands are illustrative aliases, not files or runnable commands supplied by this public guide. `my_work/` means my implementation; `ya_work/` means Ya’s foundation; `s3://<bank-storage>/` replaces the actual storage address.

On 2026-10-05 two AI assistants, CL and CH, independently reviewed the project within their recorded scopes,
ran the tests, and wrote small programs to check specific behaviors. They recorded separate audits and independent cross-checks. Here I explain their findings, which numbers those findings affect, and what remains supported. Where the two audits differ in
emphasis, this part says so.

---

## 6.1 First, what was checked and found correct

- The independent assistants reported that, at the audited snapshot, all 146 Python files parse, the targeted imports succeed, and the critical lint
  checks pass. The photo-to-text recovery shows no sign of damage: the four modules duplicated across my
  phase folders match each other line for line apart from import paths, and the pixel-verified Phase 5 copies
  match the Phase 4 copies. This is strong evidence, not a certification of every runtime path or of every line
  against the original screenshots.
- The independent assistants reported that all 28 offline tests pass (22 of mine, 6 of Ya’s) in two independent audit environments, each of which
  had to add dependencies the requirements file omits.
- The history window is built as designed: preceding transactions of the same account in timestamp-then-id
  order, target in the last slot, left padding masked out, time deltas zero at the target and non-negative
  before it (zero for a row sharing the target's timestamp). Both audits confirmed this with synthetic data;
  Ya’s notebook confirmed it on 28,054 real windows from 30 selected accounts.
- Padding never changes a prediction: both audits changed the padded slots' values arbitrarily and got
  identical outputs for every model and mask setting.
- No identifiers and no past fraud labels are inputs. Preprocessing is frozen from training files and never
  refit.
- Validation PR-AUC is computed once over the whole validation set per epoch, which is the correct way for a
  1-in-600 event, and the best-epoch checkpoint is chosen on it.
- My three sweep specifications expand to exactly the 8, 2 and 3 runs the documents describe, and the
  training command receives the overrides in the intended precedence.
- My scripts never edited Ya’s `ya_work/` code, as my isolation rule promised.

If asked "was the code reviewed?": yes, by two independent audits, which cross-checked selected findings; the tests pass; the reported defects concern the recovered project logic,
rather than a demonstrated transcription error.
The checks cover parsing, imports, tests and targeted behaviors, not every runtime path.

---

## 6.2 The defect in multi-worker shuffled training: data-loader sharding

**What happens.** Training uses 4 parallel data-loader workers and shuffles the order of files each epoch.
Each worker shuffles the *whole* file list on its own, then takes every fourth file. Because the four
shuffles differ, the four slices are not a clean division: in a 10-file epoch about 3 files are never read
and 2 or 3 are read twice. Over 10 epochs, individual files were read anywhere between 4 and 16 times instead
of exactly 10. Validation and evaluation loaders do not shuffle, so they are unaffected.

**Where.** Ya’s `ya_work/fraud/data/sequence_builder.py` and `ya_work/fraud/data/parquet_loader.py`. The documented
multi-worker shuffled runs by Ya (Phases 3 and 4) and my sweeps use this path. Zero-worker smoke runs and the in-memory baseline are exceptions.

**What it means for the numbers.** Nothing was "faster" or "cheaper" than reported: each worker still read
its share of files. The exposure to data was randomized, so each run effectively trained on a resample of the
files, and because files differ in row count even the number of rows per epoch varied. Every configuration
went through the same code with the same seed, but that does not make the effective samples identical across
architectures, and it does not guarantee that the ranking of configurations survives a corrected loader. We do
not yet know how much the defect moved the scores or their order; the corrected-loader reruns will tell.

**Fix.** Two lines: shuffle with one shared per-epoch seed, then shard. Tested in a disposable copy (each file
is read exactly once per epoch afterwards) and reproduced by CH; not yet applied to the project code. It
must land before new comparisons that use multi-worker shuffled training.

If asked: "a multi-worker sharding bug made each epoch a random two-thirds sample of the training files;
unshuffled validation was unaffected; the documented multi-worker shuffled configurations share the defect, but we do not yet know how much it moved them;
the fix is two lines, tested, and pending review and application."

---

## 6.3 The "causal" attention option is not causal

**What happens.** In causal mode a transaction's tokens are allowed to read the summary (CLS) token, and the
summary token has already read every transaction including the target. So from the second transformer layer
onward, information about the target flows backward into the history tokens, and the option does not meet its
contract. It is still not the same as the bidirectional model: direct transaction-to-transaction edges differ,
and CH measured different outputs from the same weights under the two masks. So the bug weakens the
ablation without proving the two options must score alike.

**Where.** Ya’s `ya_work/sequential_model.py`, copied faithfully into my flat and hierarchical models.

**What it means.** My mask ablation (bidirectional vs causal, within 0.001 of each other) did not compare
what it meant to compare, so its small gap has not been attributed to anything. The defect changes the
attention computation that feeds the target's classification, but it does not bring in any transaction after
the target or any label; the default bidirectional model is unaffected. It would matter for Ya’s future plan
to score every transaction in a window at once.

**Fix.** Ordinary tokens may not read the CLS key, and the mask is merged with the padding mask so that a padded
slot is never left with nothing to attend to (the first version of the fix had exactly that flaw and produced
not-a-number scores for short histories; CH's review caught it). Tested in a disposable copy, and
CH's own harness confirmed the second version: zero leakage at every depth, finite scores for padded
histories. Not yet applied to the project code.

If asked: "the causal option had a backchannel through the summary token; the default bidirectional model is
unaffected and remains the correct choice for scoring the last transaction."

---

## 6.4 Two Phase 5 pretraining defects (no Phase 5 results are documented, so none are affected)

**Supervised contrastive loss pairs the wrong rows.** The two augmented views of each transaction are laid
out sample by sample, but the label mask is laid out view by view. The loss therefore rewards the wrong
clustering: in a test, perfectly class-separated embeddings received a high loss and deliberately mixed ones a
low loss. One-line fix proposed and tested; it matches the reference formulation and CH checked it
independently; not yet applied.

**Masked autoencoder cannot tell two hidden numeric fields apart.** Hidden fields are replaced by one shared
"mask" token and decoded by one shared numeric head, with no field identity left, so two hidden numeric
fields in the same transaction receive identical predictions. CH found this; CL confirmed it. Fix:
keep a learned per-field identity when masking, or use per-field decoders. Not yet written.

If asked: "no pretraining result is documented in the repository; two objective bugs were found first; one
has a tested proposed fix, the other a designed fix not yet written; neither is applied yet."

---

## 6.5 The evaluation range is ambiguous

Every configuration file in `my_work/` evaluates on one file (`part_8000`, 290 fraud cases), and the sweep
driver does not widen it. My status document quotes eleven-file numbers (files 8000 to 8010, 3,818 fraud
cases) and compares them with Ya’s eleven-file number of 0.0940. The repository cannot tell which range
produced the `eval_ap` values in my tables; the pod's `score.log` files can (they print `n=` and
`positives=`). Until that is settled, treat the eval numbers as development numbers and quote validation
PR-AUC, which is unambiguous.

If asked: "the configs say one file, the document says eleven; I need to recover the logs; validation numbers
are unaffected."

---

## 6.6 The sweep overwrote its own prediction files

Every run in a phase wrote its predictions to the same S3 object name. So the eight finder runs, the two
long-history runs and the 100-file run share one object name, and under the configured paths only the
last-scored run's predictions survive (S3 object versioning or external backups were not inspected). Scalar
results and checkpoints survived; per-run predictions, which the leaderboard, ensemble and calibration tools
need, were overwritten. Fix: a per-run prediction path in the sweep driver. Proposed and tested; CH found
two remaining collision cases (identical directory basenames across experiments, and reruns), which the next
version addresses.

---

## 6.7 Smaller items worth knowing

| Item | Who found it | Plain meaning |
|---|---|---|
| EMA checkpoint | both | If weight averaging were switched on, the saved checkpoint would hold the raw weights while validation had scored the averaged ones. It was never switched on. Fix proposed and tested; CH found one remaining gap (the final partial accumulation group) for the next version. |
| Package does not install | both | `pyproject.toml` names a build backend that does not exist and omits two dependencies, so `pip install -e .` fails. Scripts work from a checkout anyway. Fix proposed; a wheel builds and installs in a disposable environment. |
| Vendored library one version too old | CH found, CL traced | Ya’s streaming code uses a field that first appears in `pytorch_tabular` 1.1.1; the repository pins 1.1.0, so Ya’s Phase 3 scripts cannot start from this checkout. Re-vendor 1.1.1 (that alone does not fix the separate checkpoint-loading issue under newer PyTorch). |
| Train-loss column | CH | The `train_loss` column in my status document is the last mini-batch's loss, not the epoch average, which is why it jumps around. Validation columns are correct. |
| Best AP and best AUROC from different epochs | CH | My results table takes the maximum of each column separately; the AUROC shown is not always from the epoch whose AP is shown. |
| Train/score override order | both | The sweep passes overrides to the scorer in the opposite order from the trainer; harmless today because no key overlaps, dangerous later. Fix proposed and tested. |
| Class-weight checkpoints | CH | Ya’s scorers and the Phase 5 scorers cannot load a model trained with a non-default class weight. My Phase 4 scorers can. All runs used weight 1. |
| Ensemble tie handling, leaderboard intervals, cohort checks | CH | The scoring toolkit ranks ties arbitrarily, reports a confidence interval for only the last seed, and does not check that models were scored on the same transactions. |
| Marginal means arithmetic | CH | Two averages in my status document are off in the fourth decimal (linear 0.132579, not 0.132604). Conclusion unchanged. |
| Periodic embedding design | both | My periodic embedding shares one projection across the seven history-model numeric channels (six raw features plus time delta) and uses a fixed frequency scale, unlike the paper's per-feature layers; this may be why it lost to linear. |

---

## 6.8 Which conclusions still stand, and which do not yet

Stand:

- The best documented history runs beat the documented single-transaction baseline on validation (0.1247 to
  about 0.14). This is promising evidence for history, not a controlled result: two finder runs scored below
  the baseline, the models differ in more than history, and all runs used the faulty loader.
- The window construction uses only preceding transactions and no labels, and preprocessing is frozen (with
  the tie-order and timestamp-type qualifications of Part 4.2).
- Linear numerical embeddings did at least as well as the periodic variant as implemented.

Not yet established:

- How much of the lift is history itself (no same-architecture no-history control yet).
- How much the loader defect moved the scores or their ordering.
- That history 10 or 16 beats history 5 (small gaps, one seed, no paired test).
- That the flat architecture is "saturated", or that Architecture B wins (no B results yet).
- That attention direction does not matter (the causal option did not meet its contract; the small gap is
  unattributed).
- That pretraining helps (no result documented).
- Any evaluation-set number, until the one-file vs eleven-file question is answered.

What I can conclude: "the best history runs beat the baseline; isolating history's
contribution, and everything else, needs the fixes, a no-history control, several seeds, and a pinned
evaluation cohort, which is the plan."

---

## 6.9 The agreed order of work

The independent reviewers recorded the following agreed sequence: before any new comparative training,
(1) fix and verify the loader sharding, (2) make the environment installable with pinned versions, (3) pin
the evaluation cohort by transaction id, (4) give every run its own prediction path. The pretraining fixes
gate only the pretraining experiments; measurement planning can proceed now. Existing results are kept as
exploratory observations. Patches are adopted only after the other assistant has independently verified them.
Status as of this guide: all patches are proposals tested in copies; none is applied to the project code.
