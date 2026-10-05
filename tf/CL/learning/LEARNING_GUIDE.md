# TF — my learning guide

This is my full explanation of TF, from the basic machine-learning ideas to the data pipeline, history models, experiments, findings and questions I may be asked. I am building on Ya’s foundation at the bank.

Original chapter author: **CL**. Public adaptation: **CH**, in my first-person project voice. The original author incorporated independent review corrections on 2026-10-05. These public copies preserve the author’s explanations and qualifications; they do not imply that either assistant newly endorses every claim in the other’s work.

Experiment status is a **recorded 2026-10-04 snapshot**. Audit and proposed-fix statements describe the **2026-10-05 review**. A run described as started or a result described as pending is not a live status check. Tests described here were reported by the independent assistants; this document does not claim I personally ran their verification programs.

Source paths and commands are **illustrative aliases** for the underlying project, not source files or runnable commands included in this public documentation. `my_work/` refers to my implementation, `ya_work/` to Ya’s foundation, and `s3://<bank-storage>/` to a placeholder storage address. `My` and `Ya’s` are the two artifact-ownership labels. Exact dataset counts, fraud rates, experiment scores and model details are retained with my approval for public release.

## Contents

- [Part 0. The ten-minute version](#part-0-the-ten-minute-version)
- [Part 1. Foundations from zero](#part-1-foundations-from-zero)
- [Part 2. Neural networks and transformers from zero](#part-2-neural-networks-and-transformers-from-zero)
- [Part 3. The data and Ya’s pipeline](#part-3-the-data-and-yas-pipeline)
- [Part 4. History-aware models and the sweeps](#part-4-history-aware-models-and-the-sweeps)
- [Part 5. Pretraining the encoder (Phase 5), from zero](#part-5-pretraining-the-encoder-phase-5-from-zero)
- [Part 6. What the audits found, in plain words](#part-6-what-the-audits-found-in-plain-words)
- [Part 7. Questions I may be asked, with my answers](#part-7-questions-i-may-be-asked-with-my-answers)
- [Part 8. One-page cheat sheet](#part-8-one-page-cheat-sheet)
- [Part 9. Glossary](#part-9-glossary)

---

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

---

# Part 1. Foundations from zero

Original chapter author: CL. Public adaptation: CH, in my first-person project voice.
The original author incorporated an independent CH review on 2026-10-05; this adaptation is not a new joint endorsement.

Status note: experiment status is the recorded 2026-10-04 snapshot; audit and proposed-fix statements describe the 2026-10-05 review. They are not live job or deployment status.
Source paths and commands are illustrative aliases, not files or runnable commands supplied by this public guide. `my_work/` means my implementation; `ya_work/` means Ya’s foundation; `s3://<bank-storage>/` replaces the actual storage address.

These are the ideas I use to explain TF. Every later part builds on this vocabulary.

---

## 1.1 What a model is

A **model** is a rule that turns inputs into an output number. For us the inputs are the facts about one
debit-card transaction (its amount, the hour, whether the PIN was verified, and so on) and the output is a
**score** between 0 and 1: how likely the model thinks this transaction is fraud.

The rule has adjustable numbers inside it called **weights**. Nobody writes the rule by hand. We show the model
millions of past transactions where we already know the answer, and a procedure adjusts the weights so that
the scores come out high on the fraud ones and low on the rest. That procedure is **training**. A model with
its weights fixed after training is just a calculator: same input, same score, every time.

The smallest model in this project has about 142,000 weights. The biggest history model has roughly a hundred
thousand to two hundred thousand. For comparison, a large language model has hundreds of billions. Ours are
small on purpose: the data has only 23 columns.

If asked "what is the model, really?": a function with roughly 10^5 learned weights that maps one transaction,
plus optionally its account's recent history, to a fraud probability.

---

## 1.2 Features, labels, rows

A **row** is one transaction. A **feature** is one column the model is allowed to look at. A **label** is the
column we are trying to predict. Here the label is `frd_tag`: 1 if the transaction was later confirmed as
fraud, 0 otherwise.

Features come in two kinds and the model treats them differently:

- **Continuous** features are numbers with magnitude. Example: `transaction_amt` (the amount), or
  `external_1_scr` (Visa's own fraud score, 0 to 99).
- **Categorical** features are codes from a fixed list. Example: `transaction_type_xcd` has 12 possible
  values; `cardholder_present_xflg` is a yes/no flag. The number 7 for a category does not mean "more than 6".

This project uses 23 features: 6 continuous, 13 categorical codes, and 4 calendar fields (month, day, hour,
day of week) derived from the timestamp and treated as categories. The list lives in one file,
`ya_work/configs/predictors.csv`, so everyone uses exactly the same columns.

Things deliberately **not** used as features: the account id, the transaction id, the raw timestamp
(it only orders transactions), merchant ids and postal codes (too many distinct values), and the fraud labels
of earlier transactions (Section 1.9 explains why).

If asked "what does the model see?": 23 columns per transaction, no identifiers, no past labels.

---

## 1.3 Training is guess, measure, nudge, repeat

Training works like this, millions of times:

1. Take a **batch** of rows (here 4,096 at a time).
2. Run them through the model with the current weights to get scores. This is the **forward pass**.
3. Measure how wrong the scores are with a formula called the **loss**. For a fraud row the loss is small if
   the score is near 1 and large if it is near 0; for a legitimate row the reverse. We use **cross-entropy**,
   which is the standard choice for yes/no questions.
4. Compute, for every weight, which direction would reduce the loss a little. This is the **backward pass**
   (the mathematics is called backpropagation; this explanation does not need its derivation).
5. Nudge every weight a small step in that direction. The step size is the **learning rate**
   (0.0003 here). The rule that decides how to nudge is the **optimizer**; ours is called **AdamW**.

One trip through all the training rows is an **epoch**. With 10 training files of about 840,000 rows each,
one epoch is about 8.4 million rows, or about 2,100 batches of 4,096. We train for up to 10 epochs.

Nothing here is specific to fraud. Every neural network in the world trains this way.

If asked "how is it trained?": mini-batch gradient descent with AdamW on cross-entropy loss, batch 4,096,
learning rate 3e-4, up to 10 epochs, early-stopped.

---

## 1.4 Overfitting, and why we hold data back

A model with enough weights can memorize its training rows and score them perfectly while being useless on
new rows. That is **overfitting**. The only honest test of a model is data it never trained on.

So the data is split by file. The S3 bucket holds parquet files named `part_1` through `part_5000`; a
separate "EVAL" bucket holds `part_8000` through `part_10000`. The project uses:

| Split | Files | Used for |
|---|---|---|
| Training | `part_1` to `part_10` (small sweeps) or `part_1` to `part_100` (confirmation runs) | adjusting the weights |
| Validation | `part_4950` to `part_4954` | deciding when to stop, which epoch to keep, and which design is better |
| Evaluation | `part_8000` (one file) or `part_8000` to `part_8010` (eleven files) | the final number reported |

After every epoch the model is scored on the validation files. If the validation score has not improved for
5 epochs in a row we stop (**early stopping**, patience 5) and keep the weights from the best epoch. Those
saved weights are the **checkpoint**, a file called `best.ckpt`.

A subtle point the quants will care about: validation is used to *choose*, so a validation score is slightly
optimistic. The evaluation files are supposed to be looked at rarely. The project's own standing document
admits the eleven-file evaluation set has been "peeked at repeatedly", which is why it refuses to pick winners
on evaluation scores. That is the right discipline.

My explanation of overfitting controls: separate validation files drive early stopping and model
selection; a separate evaluation bucket is reported; nothing is tuned on the evaluation files.

---

## 1.5 Why accuracy is useless here

In the training files, 0.1654% of transactions are fraud: about 1 in 600. A model that says "not fraud" to
everything is 99.83% accurate and catches nothing. So nobody in this project uses accuracy. The training logs
print an accuracy near 0.998 and the documentation treats it as uninformative for model selection.

This is the single most important fact about the problem. It shapes the metric (1.7), the loss (1.9), the
batch size (per-batch statistics are meaningless when a batch of 4,096 holds about 7 fraud cases), and the
way results are compared (1.8).

If asked "what is the base rate?": 0.1654% in training files 1 to 100 (143,338 fraud in 86,683,008 rows);
0.155% in evaluation files 8000 to 8010 (3,818 in 2,466,461).

---

## 1.6 Scores, thresholds, precision and recall

The model outputs a score per transaction. A bank uses it by **ranking** transactions and reviewing the top
ones, or by setting a **threshold** and alerting on everything above it. Two questions follow:

- **Precision**: of the transactions we alerted on, what fraction were really fraud?
- **Recall**: of all the fraud, what fraction did we alert on?

Worked example with round numbers. Take 1,000,000 transactions with 1,654 fraud among them. Suppose the
fraud team can review 0.1% of traffic, so we alert on the top 1,000 scores. If 300 of those are fraud:

- precision = 300 / 1,000 = 30%
- recall = 300 / 1,654 = 18%

Raise the threshold and precision usually rises while recall falls; lower it and the reverse. There is no
single right threshold; it depends on how many analysts the bank has and what a missed fraud costs. The
scoring toolkit in `my_work/scoring/` reports precision and recall at alert budgets of 0.1%, 0.5% and 1% for
this reason.

My recommendation on the operating point: that is a business decision; the model provides a ranking
and we report precision/recall at the bank's alert budget.

---

## 1.7 PR-AUC and ROC-AUC, the two curves

Because the threshold is not fixed, we judge the whole ranking at once with two summary numbers.

**ROC-AUC** (area under the receiver operating characteristic curve) answers: if I pick one random fraud
transaction and one random legitimate transaction, how often does the model score the fraud one higher?
0.5 is a coin flip; 1.0 is perfect. Every model in this project scores about 0.93 to 0.95. That sounds
excellent, and it barely moves between designs, which is exactly why it is not used for decisions: with 600
legitimate transactions per fraud, ranking a fraud above *most* legitimate ones is easy; what matters is
ranking it above *almost all* of them.

**PR-AUC** is what this project calls its selection metric. What the code computes is scikit-learn's
**average precision** (**AP**): precision averaged along the ranking, with each fraud case's step in recall as
the weight. It summarizes the ranking across all recall levels and differs slightly from the trapezoidal area
under the precision-recall curve. It rewards a clean top of the ranking, but it does not by itself report
precision at the bank's alert budget (the top 0.1% or 0.5%), so that number should be shown beside it. A
random model scores AP equal to the fraud rate, about 0.0016. The
single-transaction model scores about 0.125 on validation; the best history model 0.1416. That is roughly
85 times random. In the code the numbers are called `valid_ap` (on validation files) and `eval_ap` (on
evaluation files).

Why does PR-AUC look so low if the model is good? Because at 1-in-600 prevalence even a strong model's top
alerts contain many legitimate transactions. A PR-AUC of 0.14 does **not** mean 14% accuracy or 14% recall; it
is an average of precision across recall levels. Translate it into precision at a budget when talking to the
business.

Both numbers are computed over the **whole** validation set at the end of an epoch, never per batch. The
code that does this is `StreamingBinaryMetrics` in `ya_work/callbacks.py`. Early stopping and checkpoint
selection use `valid_ap`.

If asked "why PR-AUC and not ROC-AUC?": in this project ROC-AUC sits near 0.95 for every model tried and can
hide poor precision at very low alert rates; average precision summarizes the ranking where the fraud team
operates, and precision at the alert budget should be shown beside it.

---

## 1.8 How sure I can be about a number

Every PR-AUC is computed on a finite sample of fraud cases and has sampling wobble. A published binomial-style
approximation for the standard error of **one model's** AP is `sqrt(AP * (1 - AP) / number_of_fraud_cases)`
(Boyd, Eng and Page, 2013). It is useful to illustrate the scale of that wobble. Three limits: its coverage has
not been validated for this data, where windows from the same account are correlated; it is an interval for one
model, not for the difference between two models; and it says nothing about training randomness.

| Set | Fraud cases (approximate, from the documented rates) | AP about | Illustrative marginal standard error |
|---|---:|---:|---:|
| 5 validation files (4950 to 4954) | ~6,600 | 0.135 | 0.004 |
| 11 evaluation files (8000 to 8010) | 3,818 | 0.09 | 0.005 |
| 1 evaluation file (8000) | 290 | 0.09 | 0.017 |

This table is an illustrative marginal approximation, not a project confidence interval.

Two consequences for this project's tables:

- The documented history 10 and history 16 validation scores (0.1399 vs 0.1383) are close. Whether they differ
  has not been tested; the table only says the gap is about the size of one model's own wobble.
- Any evaluation number from one file rests on 290 fraud cases and is far less stable than the validation
  numbers. Confirm which cohort produced it before reading anything into differences between eval columns.

The proper way to compare two models is a **paired** comparison: score both on the same identified
transactions, then resample by a justified unit (accounts, not single transactions) many times and look at the
distribution of the difference. That needs both models' predictions on the same cohort, which the sweep did
not keep (Part 6 explains why). It is on the plan.

There is a second source of noise: training itself is random (the order of batches, the initial weights).
Every sweep in this project used one random **seed** (42). Repeating the best configurations with 3 to 5 seeds
shows how much a score moves between runs; together with the paired comparison, that is what would support a
claim that one configuration is better. It has not been done yet.

If asked "is 0.1399 vs 0.1383 significant?": not shown either way; both are single-seed runs and no paired test
has been run. The honest statement is "the documented scores are close; their reproducibility and statistical
separation have not been established; a seeded, paired comparison is the next step".

---

## 1.9 Rare events: weighting, downsampling, calibration, leakage

Four ideas come up whenever the positive class is rare.

**Class weighting.** Multiply the loss on fraud rows by a factor so the model pays more attention to them.
The code supports it (`pos_weight`) but every reported run used weight 1. Weighting changes the probabilities
the model outputs but not necessarily the ranking.

**Downsampling.** Keep all fraud rows and throw away most legitimate rows during training, so the model sees,
say, 12% fraud instead of 0.16%. The bank's Payment Foundation Model team has used a version of this (their
configs keep 1 in 100 legitimate rows, which gives about 14% fraud). My roadmap proposes it as a future
experiment. It is **not** yet implemented here.

**Calibration.** A model trained on 12% fraud will output probabilities that are far too high when deployed
on 0.16% fraud. Calibration is the correction back to true probabilities (Platt scaling, isotonic regression,
or a prior-shift formula). Training at the natural rate avoids that sampling shift, but it does not by itself
guarantee calibrated probabilities; calibration has to be measured on held-out predictions, and none is
documented yet. Two things to get right in the meeting: a strictly increasing recalibration leaves the
ranking, and therefore PR-AUC and ROC-AUC, unchanged; isotonic regression is only non-decreasing and can
create ties, which can move those metrics slightly, so report ranking on raw scores and calibration quality
separately. Calibration matters for thresholds and expected-cost decisions. (My roadmap treats
calibration as a prerequisite for comparing ranking metrics; both AI audits flagged that.)

**Leakage.** Any information in the inputs that would not be available at decision time inflates the score
dishonestly. The clearest example here: the fraud labels of an account's earlier transactions. Fraud is
confirmed days or weeks after it happens, so at scoring time those labels are unknown. The design excludes
them (decision 2 in `ya_work/docs/phase4_design_decisions.md`). The bank's Confluence notes record another team
hitting exactly this leak and getting a meaningless 99% AUC.

My answer on past fraud labels: no, by design, because they are not available at
decision time; the history model uses only past transactions' features.

---

## 1.10 The baseline to beat: XGBoost

The bank's production fraud model is **XGBoost**, a gradient-boosted decision-tree model. Picture a tree of
yes/no questions ("amount > 500? hour between 1 and 5? PIN verified?") ending in a score, then hundreds of
such trees added together, each one correcting the errors of the previous ones.

Trees are the workhorse of tabular prediction for good reasons: fast to train, robust, handle missing values,
need little tuning, and interpretable. Their weakness is that they see one row at a time. If I want a tree
to know "this account normally spends 40 and this is 900", a person must compute that ratio and add it as a
column. Production fraud XGBoost models are fed dozens to hundreds of such hand-built features.

The bet of this project is that a transformer can (a) learn feature interactions itself from the 23 raw
columns and (b) read the account's recent transactions directly instead of through hand-built summaries. The
published literature says trees still win or tie on most plain tabular benchmarks, and that transformers pull
ahead when there is structure trees cannot see, such as sequences. So the honest framing is: "we are testing
whether history-aware transformers add signal over a per-transaction tree model", not "transformers beat
trees".

If asked "why not just improve the XGBoost?": the two are complementary; the cheapest win may be adding the
transformer's score as one more feature in XGBoost, and that comparison has not been run yet.

---

# Part 2. Neural networks and transformers from zero

Original chapter author: CL. Public adaptation: CH, in my first-person project voice.
The original author incorporated an independent CH review on 2026-10-05; this adaptation is not a new joint endorsement.

Status note: experiment status is the recorded 2026-10-04 snapshot; audit and proposed-fix statements describe the 2026-10-05 review. They are not live job or deployment status.
Source paths and commands are illustrative aliases, not files or runnable commands supplied by this public guide. `my_work/` means my implementation; `ya_work/` means Ya’s foundation; `s3://<bank-storage>/` replaces the actual storage address.

Part 1 said a model is a function with learned weights. This part opens the box for the particular kind of
function used here, one small idea at a time. Nothing beyond arithmetic is needed.

---

## 2.1 A neuron, a layer, a network

A **neuron** takes some input numbers, multiplies each by a weight, adds them up, adds a constant (the
**bias**), and passes the result through a simple bend such as "negative numbers become zero" (the
**activation**). That bend is what lets networks learn curves instead of straight lines. Ours uses a smooth
bend called GELU; the name does not matter.

A **layer** is many neurons reading the same inputs in parallel; its output is a list of numbers. A
**network** is layers stacked so each reads the previous one's output. "Deep learning" just means several
layers.

If asked "what activation?": GELU in the feed-forward blocks, standard for transformers.

---

## 2.2 Embeddings: turning a value into a vector

A network only eats lists of numbers of a fixed length. A transaction type code like `"P"` or a category
like "Tuesday" is not a number, and an amount like 21.71 is one number where the network wants many.
**Embeddings** solve both.

- For a **categorical** feature, keep a table with one row per possible value, each row a list of `d` learned
  numbers (`d` is 48 in the history models). Looking up the row for "Tuesday" gives Tuesday's vector. The
  numbers in the table are weights, learned like any other, so the model decides for itself that, say,
  Saturday and Sunday should end up with similar vectors if that helps predict fraud.
- For a **continuous** feature, first standardize it (subtract the training mean, divide by the training
  standard deviation, so most values land between about -3 and 3), then multiply the single number by a
  learned vector of length `d` and add a learned bias vector. This is the **linear** embedding. The project
  also tried a **periodic** embedding, which first expands the number into sines and cosines at 16 learned
  frequencies and then projects to `d`; it can capture "wiggly" relationships. In the sweep, linear won.

The output of embedding one transaction is a set of `d`-dimensional vectors, one per feature. Each such vector
is a **token**. One transaction here becomes **30 tokens**: 7 continuous (the 6 numeric features plus a
time-gap feature in the history model) and 23 categorical (13 codes, 4 calendar fields, and 6 "was this
numeric value missing" flags).

If asked "how are categoricals handled?": learned embedding tables with reserved indices 0 for unknown and
1 for missing; continuous features are standardized with frozen training statistics and linearly embedded.

---

## 2.3 Attention: every token looks at every other token

The transformer's one big idea. Given a set of tokens, **attention** lets each token build a new version of
itself as a weighted average of all the tokens, with weights it computes on the fly.

Mechanically, each token produces three vectors through three small learned matrices: a **query** ("what am I
looking for"), a **key** ("what do I contain"), and a **value** ("what I will contribute"). For token A, take
the dot product of A's query with every token's key; the better they match, the bigger the number. Pass those
numbers through a **softmax**, which turns them into positive weights that sum to 1. A's new vector is the
sum of every token's value times its weight.

What this buys us: the "amount" token can learn to look hard at the "merchant category" token and the "hour"
token when deciding what to pass on, and the strength of that look depends on the actual values. That is a
feature interaction learned from data, no human needed. Trees have to find such interactions by splitting;
attention computes them directly.

**Multi-head** attention runs several of these lookups in parallel with different matrices (8 heads here),
so one head can track "amount vs limit" while another tracks "hour vs card-present". A **block** is one
attention step followed by a small two-layer network applied to each token (the **feed-forward** part), with
two housekeeping tricks: a **residual connection** (add the input back to the output so information is never
lost) and **layer normalization** (rescale each token's numbers so training stays stable). The models here
stack 4 blocks (2 plus 4 in the hierarchical variant).

Attention cost grows with the **square** of the number of tokens, because every token looks at every other.
30 tokens is trivial; 511 tokens (history of 16) is 290 times more work per layer. That single fact drives
most of Part 4.

If asked "what is attention, in one sentence?": a learned, input-dependent weighted average over all tokens,
so the model decides at run time which features and which past transactions matter for this one.

---

## 2.4 The CLS token and the head

After the blocks, we have 30 updated token vectors and need one score. The trick, borrowed from BERT, is to
add one extra learned token called **CLS** (for "classification") before the blocks. It has no feature of its
own; through attention it absorbs whatever the other tokens offer. After the last block, we throw away all
tokens except CLS and feed its 48 numbers to the **head**: a single linear layer producing 2 numbers (the
**logits**, one for "legit", one for "fraud"). A softmax turns them into two probabilities that sum to 1; the
fraud probability is the score.

My explanation of producing one score from many tokens: a CLS token aggregates through attention; a linear head
on its final vector gives two logits; softmax gives the probability.

---

## 2.5 FT-Transformer: the architecture this project chose

**FT-Transformer** stands for Feature Tokenizer plus Transformer (Gorishniy, Rubachev, Khrulkov and Babenko,
"Revisiting Deep Learning Models for Tabular Data", NeurIPS 2021). It is exactly Sections 2.2 to 2.4: embed
each feature as a token, add CLS, run transformer blocks, classify from CLS. Two properties make it a good
fit for tables:

1. Features are a **set**, not a sequence. Nothing says "amount comes before hour". FT-Transformer uses no
   position information inside a transaction, which is correct: each token already knows which feature it is
   from its own embedding.
2. It is the strongest general deep model for tables in that paper, roughly matching gradient-boosted trees
   across many datasets, while a plain multi-layer network (an **MLP**) trails.

Ya confirmed this on a public dataset before touching bank data (`ya_work/docs/tabular_transformer_models.md`):
on the UCI Bank Marketing set, FT-Transformer reached ROC-AUC 0.935 versus 0.921 for an MLP and 0.82 to 0.85
for a model called SAINT. That benchmark is a term-deposit prediction task, not fraud, and used no tuning, so
treat it as "a reasonable reason to start here", not as proof.

One precise point for a PhD audience: the single-transaction models (Ya’s Phases 1 to 3) use the
`pytorch_tabular` library's FT-Transformer implementation. The history models (Ya’s Phase 4 and all of my
work) are hand-built with PyTorch's standard `nn.TransformerEncoder` using the same tokenizer idea. They are
FT-Transformer in design, not the library's code, and differ in small internals (pre-norm vs post-norm,
GELU vs a gated activation, head width). The documentation calls both "FT-Transformer".

If asked "why FT-Transformer?": per-feature tokenization fits tabular data, it was the best deep tabular
model in the 2021 benchmark, it matched trees there, and it extends naturally to sequences of transactions,
which trees cannot do.

---

## 2.6 From one transaction to a history: tokens, positions, masks

To add history we embed the target transaction and each of its N previous transactions (same account,
preceding it in timestamp order with the transaction id as a tie-break; rows sharing the target's timestamp can
be included) the same way, giving `(N + 1) * 30` tokens, and let attention run over all of them plus CLS.
Three additions make this work:

- **Transaction position embedding.** Attention is order-blind, so without help the model cannot tell "this
  token belongs to the most recent transaction" from "this belongs to the oldest". A learned vector per slot
  (slot 0 = oldest, slot N = target) is added to every token of that slot. There is no position *within* a
  transaction, by the set argument above.
- **Time delta.** A continuous feature per transaction, `log(1 + seconds before the target)`, so the model knows
  whether the previous five transactions happened in one hour or one month. Velocity is a classic fraud signal.
- **Masks.** Accounts with fewer than N previous transactions get **padding**: empty slots on the left, and a
  mask telling attention to ignore them. A second kind of mask, **causal**, would stop a transaction's tokens
  from looking at later transactions; the default is **bidirectional** (everything sees everything), which is
  safe here because all inputs are legitimately known at scoring time and only the target's label is used.

Note on names: the plans talk about "SEP tokens" separating transactions. The code has none; the position
embeddings mark the boundaries. Both audits noted this as a naming discrepancy, not a defect.

If asked "how does the model know the order of transactions?": a learned per-slot position embedding plus an
explicit log time-gap feature.

---

## 2.7 What a GPU training run actually does

Put together, one run of the history model with N = 5 on 10 files:

1. A data loader reads parquet files from S3 one at a time, groups rows by account, sorts by time, builds one
   window per transaction (target plus 5 prior, left-padded), applies the frozen preprocessing, shuffles
   windows in a 65,536-window buffer, and hands over batches of 4,096 windows.
2. Each window is 6 slots of 30 tokens plus CLS: 181 tokens of 48 numbers. The GPU runs the forward pass for
   the 4,096 windows at once, computes the cross-entropy loss, runs the backward pass, and AdamW nudges the
   roughly 120,000 weights. Arithmetic is done in **bf16**, a 16-bit number format, to halve memory and
   speed things up; it is the standard choice for transformers, and its effect on these results has not been
   measured separately.
3. About 2,100 such steps make one epoch. After each epoch the model scores all validation windows (about
   4.4 million) and `valid_ap` is computed once over all of them.
4. Repeat for up to 10 epochs; stop when `valid_ap` has not improved for 5; keep the best checkpoint.
5. Score the evaluation files with the checkpoint and write a parquet of (id, label, score) to S3.

On an NVIDIA H200 this takes on the order of an hour for 10 files and several hours for 100.

If asked "what hardware and precision?": one NVIDIA H200, bf16 mixed precision, batch 4,096 windows,
gradient accumulation used for the long-history flat runs to keep the effective batch constant.

---

## 2.8 Capacity, and what "feature-limited" means

The number of weights is the model's **capacity**. More capacity can fit more complex patterns but needs more
data and overfits more easily. Ya’s Phase 3 finding was that making the single-transaction model bigger,
or training it longer, or giving it 10x more data, moved validation PR-AUC from about 0.11 only to about 0.125
and then flat. When capacity, epochs and data all fail to help, one natural reading is that the model has
extracted most of what the 23 columns contain: that it is **feature-limited**. The plateau motivates adding new
information (history) rather than a bigger model, but it does not prove the inputs are exhausted; optimization
settings, capacity choices and data coverage can also cause a plateau.

My answer on model capacity: yes at the single-transaction stage; it plateaued at about 0.125, which
is why Phase 4 adds history instead of capacity. A capacity sweep for the history model is still on the list.

---

## 2.9 Optimizer, learning rate schedule, regularization

**AdamW** is the standard optimizer for transformers: it adapts the step size per weight and applies **weight
decay** (a gentle pull of weights toward zero, 1e-5 here) to discourage overfitting. **ReduceLROnPlateau**
halves the learning rate when validation loss stops improving for 3 epochs. **Dropout** (randomly zeroing 10%
of activations during training) is another anti-overfitting trick used in the attention and feed-forward
parts. My sweep also tried **RAdam**, a variant; its gap to AdamW was tiny and is unresolved with one
seed. **EMA** (an
exponential moving average of the weights) exists in the code as an option but was never turned on.

If asked "what optimizer and schedule?": AdamW, learning rate 3e-4, weight decay 1e-5, ReduceLROnPlateau on
validation loss, dropout 0.1, early stopping on validation PR-AUC.

---

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

---

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

---

# Part 5. Pretraining the encoder (Phase 5), from zero

Original chapter author: CL. Public adaptation: CH, in my first-person project voice.
The original author incorporated an independent CH review on 2026-10-05; this adaptation is not a new joint endorsement.

Status note: experiment status is the recorded 2026-10-04 snapshot; audit and proposed-fix statements describe the 2026-10-05 review. They are not live job or deployment status.
Source paths and commands are illustrative aliases, not files or runnable commands supplied by this public guide. `my_work/` means my implementation; `ya_work/` means Ya’s foundation; `s3://<bank-storage>/` replaces the actual storage address.

Phase 5 is written, has two main objective defects with proposed repairs (not yet applied to the project code),
and has no result documented in this repository snapshot. I explain what it is for and how I would test it.

---

## 5.1 The idea

Fraud labels are rare (1 in 600) and arrive late. Transactions themselves are plentiful. **Pretraining**
trains the encoder on a first task before the final fraud classifier is fitted (**fine-tuning**). Of the two
recipes here, the masked autoencoder is self-supervised and ignores fraud labels; SupCon is supervised
pretraining and uses the fraud labels to decide which transactions should look alike. (Both scripts still read
the label column through the shared loader, so neither runs on an unlabeled archive as written.) The analogy
for the self-supervised recipe: learn the language first, then study for the exam. In text and vision this is
the standard recipe; on tabular data the evidence is mixed, which is why Phase 5 is framed as an experiment,
not a plan.

The object being pretrained is the **encoder**: the tokenizer plus transformer that turns one transaction into
a vector (`my_work/phase5_pretrain/encoder.py`, `TabularEncoder`). After pretraining, a 2-way classification head
is attached and the whole thing is fine-tuned on fraud labels (`ssl_finetune.py`). Both pretraining recipes
produce the same kind of encoder, so they can be compared head to head against training from scratch.

If asked "why pretrain?": labels are rare and late, transactions are abundant; a pretrained encoder may need
fewer labels to reach the same ranking quality, and it gives an embedding other tools can use. The label-free
argument applies to the masked autoencoder; SupCon needs the labels.

---

## 5.2 Recipe 1: the masked autoencoder (self-supervised)

**Self-supervised** means the training signal comes from the data itself, no labels. Hide some of a
transaction's fields and ask the model to reconstruct them from the rest: hide the amount and the hour, show
the merchant category and the time of day, predict the amount and the hour. To do that well the model must
learn how fields relate (small amounts at grocery merchants, large at electronics), which is the kind of
knowledge a fraud classifier can reuse.

In `ssl_pretrain.py`: 30% of a transaction's tokens are replaced by a learned "mask" token; the encoder runs;
for each hidden continuous field a small head predicts the number (squared-error loss) and for each hidden
categorical field a head predicts the category (cross-entropy). Early stopping is on validation
reconstruction loss. Then `ssl_finetune.py` loads the encoder weights (`load_pretrained`) and trains the
classifier; `freeze_encoder: true` trains only the head while holding the encoder weights fixed, a cheap
approximation of a "linear probe". As written the encoder's dropout stays active during that training; the
standard deterministic linear-probe baseline needs the encoder put in evaluation mode.

The audit found a design flaw: because the mask token is the same for every field and the numeric decoder is
shared, two hidden numeric fields in one transaction get identical predictions (there is nothing left to say
which field is which). Fix: keep a learned per-field identity when masking, or use one decoder per numeric
field; this fix is designed, not yet written. Until then the objective cannot tell simultaneously hidden
numeric fields apart, which blunts what it is meant to teach; other learning can still happen.

If asked "what is the masked autoencoder?": hide 30% of fields, reconstruct them from the rest, no labels;
the encoder learns field relationships; a classifier is then fine-tuned on top.

---

## 5.3 Recipe 2: supervised contrastive learning (SupCon)

**Contrastive** learning teaches an encoder by comparison: pull representations of things that should be alike
together, push the rest apart. The **supervised** version (Khosla et al., 2020) defines "alike" by label: two
fraud transactions should land near each other, far from legitimate ones. For a rare class this is attractive
because every fraud example is pulled toward every other fraud example in the batch, concentrating the rare
class in one region of the representation space.

In `supcon.py`: each transaction is corrupted twice (30% of its fields swapped for another random row's values,
the SCARF recipe), both views are encoded and projected to 128 numbers, and the SupCon loss, with temperature
0.07, is applied over the batch using fraud labels as the grouping. Then the same fine-tuner as Recipe 1
trains the classifier.

The audit found that the loss attached the labels to the wrong rows (the views were laid out sample by sample
while the label mask assumed view by view), so it would have pulled the wrong pairs together. A one-line fix is
proposed; it is tested on synthetic data, matches the reference formulation, and CH checked it
independently. It is not yet applied to the project code.

A design question the Phase 5 guide raises itself: at 0.165% fraud, a batch of 8,192 holds about 13 fraud
transactions. The guide proposes a balanced sampler (keep all fraud, downsample legitimate rows). Note that
after the fix each transaction's two views already give it one positive partner, so the sampler is an
experiment about learning efficiency, not a requirement for the loss to work.

If asked "what is SupCon?": a loss that pulls same-label representations together and pushes others apart,
using two augmented views per transaction; attractive for rare classes; a bug in the label alignment has a
proposed, independently checked fix.

---

## 5.4 What a fair Phase 5 experiment looks like

Three encoders, identical architecture, data, preprocessing, validation files and evaluation cohort:

1. **From scratch**: train the classifier directly.
2. **Masked-autoencoder pretrained** (after the field-identity fix), then fine-tuned.
3. **SupCon pretrained** (after the alignment fix), then fine-tuned.

Report validation PR-AUC with 3 seeds each, plus the linear-probe variant for the two pretrained encoders.
Only then compare. Two mismatches exist in the current configs and must be fixed first: the SSL config
validates on one file (4950) while SupCon validates on five (4950 to 4954); both evaluate on one file (8000).

Also note what Phase 5 is **not** yet: it pretrains a single-transaction encoder of width 64 and 4 blocks,
while the Phase 4 history models use width 48; no Phase 4 script loads a pretrained encoder. The natural
connection is Architecture B, whose stage-1 transaction encoder could be initialized from a pretrained
encoder of matching shape. Matching width alone is not enough: B defaults to 2 transaction blocks against Phase
5's 4, and the history encoder has the extra time-gap channel, so checkpoint keys and shapes would not line
up without a deliberate mapping. That is a future experiment.

If asked "has pretraining helped?": no result is documented in the repository, and the snapshot cannot tell
whether a bank-side run happened; two objective defects were found first and have proposed repairs; the fair
test is pretrained vs from-scratch on identical architecture, data and cohort with seeds.

---

## 5.5 Vocabulary from this part

- **Encoder**: the part of the model that turns a transaction into a vector.
- **Pretraining / fine-tuning**: learn general structure first, then the labeled task.
- **Self-supervised**: training signal from the data itself (reconstruction), no labels.
- **Masked autoencoder**: hide fields, reconstruct them.
- **Contrastive / SupCon**: learn by pulling alike things together; "alike" defined by label in SupCon.
- **Augmentation / view**: a corrupted copy of a transaction (here, 30% of fields swapped with another row).
- **Projection head**: a small network mapping the encoder vector into the space where the contrastive loss is
  computed; discarded after pretraining.
- **Linear probe**: train only a linear head on a frozen encoder; measures how good the representation is on
  its own (deterministic only if the encoder is in evaluation mode).

---

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

---

# Part 7. Questions I may be asked, with my answers

Original chapter author: CL. Public adaptation: CH, in my first-person project voice.
The original author incorporated an independent CH review on 2026-10-05; this adaptation is not a new joint endorsement.

Status note: experiment status is the recorded 2026-10-04 snapshot; audit and proposed-fix statements describe the 2026-10-05 review. They are not live job or deployment status.
Source paths and commands are illustrative aliases, not files or runnable commands supplied by this public guide. `my_work/` means my implementation; `ya_work/` means Ya’s foundation; `s3://<bank-storage>/` replaces the actual storage address.

These are my spoken answers. Additional background explains the reasoning. Where the evidence is incomplete, I state the limitation.

---

**Q1. Why a transformer at all? Trees win on tabular data.**

On plain tables, yes, boosted trees usually win or tie, and our production XGBoost is the baseline to beat. The
reason to try a transformer is history: a tree sees one row, so any account-level context has to be hand-built
as features. A transformer can read the last N transactions directly and learn which ones matter. Our
single-transaction transformer sits at validation PR-AUC about 0.125 and plateaued when we added capacity,
epochs or data; our best runs with five prior transactions reach about 0.14. That lift is the case for
history, with one caveat: the history runs also changed the encoder and training details, so the
same-architecture no-history control is planned.

*Background:* the published FT-Transformer paper (Gorishniy et al., 2021) reports it roughly matching boosted
trees across many tabular benchmarks. We have not yet compared our transformer to our XGBoost on the same
transactions; that is on the plan.

---

**Q2. Why FT-Transformer specifically?**

Each feature becomes one token, so features are treated as an unordered set, which is right for a table, and
attention learns the interactions between them. In Ya’s benchmark on a public bank dataset it beat an MLP
and SAINT, and it extends to sequences of transactions by simply adding more tokens. The single-transaction
model uses the `pytorch_tabular` library implementation; the history models are the same design hand-built on
PyTorch's standard transformer encoder.

---

**Q3. How do I prevent leakage?**

Three ways. First, no identifiers and no historical fraud labels are inputs; the labels of earlier
transactions are not known at decision time, so using them would be the classic leak. Second, the history
window uses only preceding transactions on the same account in timestamp-then-id order; rows sharing the
target's timestamp can enter with a zero time gap, and we still need to confirm that order matches what is
available at scoring time. The code has nine tests on that contract plus a notebook check on real data. Third,
all preprocessing statistics are frozen from the first five training files and never refit on validation or
evaluation data.

*Background:* both audits confirmed the window construction uses only preceding rows and no labels. The caveat
they raised: the "causal attention" option lets information pass through the summary token into history tokens
in deeper layers, so it does not meet its contract and the mask ablation is weakened. It does not bring in any
transaction after the target or any label. It was never the default.

---

**Q4. Why PR-AUC and not ROC-AUC or accuracy?**

At 0.165% fraud, accuracy is 99.8% for a model that flags nothing. ROC-AUC is about 0.95 for every model we
have tried and barely moves; it can hide poor precision at very low alert rates. What we call PR-AUC is average
precision: precision averaged across recall levels, which rewards a clean top of the ranking; a random model
scores only 0.0016 on it. We select on it and report both, and the number the business needs beside it is
precision and recall at the actual alert budget.

---

**Q5. Is 0.14 good?**

It is about 85 times a random ranking and about 13% above the single-transaction model. It is not 14%
accuracy or 14% recall. For the business the number that matters is precision and recall at the review
budget: for example, how many of the top 0.1% of transactions are fraud. We have the tooling to report that
and it should be shown next to the PR-AUC on the same cohort as the production model.

---

**Q6. How do I know history helps rather than the model change?**

Honestly, not yet in a controlled way. The two models share the features, frozen preprocessing, metric and
validation files, but the history model is a different encoder with a time-gap feature and different batch
settings, and both ran on a loader with a known defect. The lift from 0.125 to 0.14 is large relative to one
model's sampling wobble, which is what makes it promising. The clean controls are the same history model with
history length zero, time delta on versus off, and reruns on the corrected loader. All are in the plan.

---

**Q7. One seed? No confidence intervals?**

Correct, and that is the main methodological gap. Every sweep used seed 42. A published binomial-style
approximation puts one model's validation sampling wobble around 0.004 (Boyd et al. 2013), but that is a
marginal illustration, not an interval for the difference between two models, and its coverage is unvalidated
for correlated account windows. We have no estimate of training-seed variance and no paired comparison. So the
honest statements are: the best history runs are well above the baseline; linear exceeded periodic in all four
matched one-seed pairs (0.008 average gap); the history-length, attention-direction and optimizer gaps are
unresolved, neither shown real nor shown absent. The next round runs the finalists with three to five seeds on
the same identified cohort and reports paired intervals.

---

**Q8. My evaluation numbers: one file or eleven?**

The configuration files in the repository score one evaluation file with 290 fraud cases; the documents quote
numbers for eleven files with 3,818 fraud cases. The audit flagged the discrepancy and I need to recover the
exact logs. Until then I treat the evaluation scores as development numbers and report validation PR-AUC,
which is unambiguous. My plan is to pin the cohort to files 8000 to 8010 and record it by transaction id.

---

**Q9. The audits found a data-loading bug. Does that invalidate the results?**

We do not yet know how much. With four loader workers, each epoch saw a random subset of roughly two thirds of
the training files, with some read twice. Validation and evaluation were unaffected. Every configuration went
through the same defect, but that does not guarantee the ordering survives a corrected loader. The fix is two
lines, tested in a copy and independently reproduced; it will be applied before the finalists are rerun on
the corrected loader.

---

**Q10. What is the difference between my two architectures?**

Flat: all tokens of all transactions in one transformer, so a field of one transaction can attend directly to
a field of another; cost grows with the square of (history times fields), which is why history 16 needed a
sixteen-fold smaller per-step batch. Hierarchical: a shared small transformer compresses each transaction to
one vector, then a second transformer attends over the N+1 vectors; cost grows with the square of N only, so
long histories are cheap, at the price of no direct field-to-field attention across transactions. It mirrors
how the bank's Payment Foundation Model treats one transaction as one unit. The status recorded on 2026-10-04 says the hierarchical sweep started; that snapshot contains no results.

---

**Q11. Why not longer history? Accounts have hundreds of transactions.**

Median 133 transactions per account in the data, so there is plenty of history. The flat model runs out of GPU
memory past about 16 because attention cost is quadratic in tokens. The hierarchical model exists for exactly
this reason; once it is validated at 16 we can test 32 and 64 cheaply.

---

**Q12. What did the pretraining phase show?**

Nothing yet; it is implemented and no results are documented. Both audits found a bug in the contrastive loss
(labels misaligned with the two augmented views) and a design flaw in the masked autoencoder (two hidden
numeric fields are indistinguishable). The first has a tested proposed fix; the second has a designed fix not
yet written; neither is applied yet. When run, pretraining will be compared against a from-scratch encoder
with identical architecture and data before anyone claims a benefit.

---

**Q13. How is class imbalance handled?**

Currently by metric choice and by not downsampling: training sees the natural 0.165% rate, which avoids an
artificial prevalence shift (it does not by itself guarantee calibrated probabilities; that has to be
measured), and selection uses PR-AUC computed over the full validation set, never per batch. Class weighting
and the Payment Foundation Model team's downsampling recipe (keep all fraud, keep a fraction of legitimate
rows) are implemented or planned as experiments. If we downsample, the probabilities need recalibration for
thresholds; a strictly increasing recalibration leaves ranking metrics unchanged, while isotonic calibration
can create ties and move them slightly, so we report ranking on raw scores.

---

**Q14. How do I handle missing values and unseen categories?**

Continuous: median imputation from training statistics plus a binary "was missing" flag per feature, so
missingness itself is a signal. Categorical: a vocabulary frozen from training with two reserved codes,
0 for a value never seen in training and 1 for missing. The artifacts are versioned, which helps
reproducibility; the pairing of a checkpoint with its artifact version still has to be recorded and checked,
because the scorer loads the two separately.

---

**Q15. What about calibration and operating thresholds?**

Scores come from a softmax trained at natural prevalence, so there is no sampling-prior shift to undo, but no
calibration measurement is documented yet. The scoring toolkit fits isotonic or Platt calibration on validation predictions and reports Brier score and
precision/recall at alert budgets of 0.1%, 0.5% and 1%. The audits noted that isotonic calibration can create
ties that slightly change ranking metrics, so we report ranking on raw scores and calibration separately.

---

**Q16. How does this relate to the Payment Foundation Model?**

That is a separate team's large tokenized sequence model with discrete token ids and a 2,048-token cap; it is
not weight-compatible with ours. We borrow two things from it: the hierarchical one-vector-per-transaction
idea, and its 17-transaction window as a target for history length. It is a comparison point, not an
ingredient.

---

**Q17. What would production look like?**

Scoring one transaction needs the account's last N transactions, which means a feature store keyed by account
with the frozen preprocessing applied. The model itself is small (about 10^5 weights), but attention cost
grows with the window, so end-to-end latency and history retrieval both need measurement; nothing has been
benchmarked. Serving is explicitly out of scope so far; nothing has been built.

---

**Q18. What is the temporal holdout story?**

The training and evaluation buckets are different file ranges, but we have not yet confirmed they are
different time periods, and the eleven-file evaluation set has been looked at repeatedly. The plan is to
define a later, untouched cohort with mature labels for the final number. Until then every number is a
development number.

---

**Q19. What would I do with another month of GPU time?**

In order: apply the fixes; rerun history 0, 5, 10, 16 flat and hierarchical with five seeds on the fixed
loader and pinned cohort; run time-delta on and off; compare with production XGBoost on the same transactions
and test the transformer score as an XGBoost feature; then pretraining versus scratch; then longer histories
on the hierarchical model.

---

**Q20. Who wrote the code, and can we trust the repository?**

Ya wrote the base; I wrote everything under `my_work/`. The repository was reconstructed from screen
photographs after a laptop failure, with a documented pixel-level audit. Two independent AI audits then
checked that all 146 Python files parse, targeted imports succeed, all 28 offline tests pass in supplemented
environments, and duplicated modules match each other exactly; the defects they raise are logic issues present
in the original code, not transcription errors. Those checks are strong evidence about the reconstruction;
they do not certify every runtime path or a clean install.

---

**If I am asked something I cannot verify:** I say, "I do not have that verified number yet; I will check the source evidence and follow up." I record the question and its eventual answer in the public questions document without treating either assistant’s answer as automatically correct.

---

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

---

# Part 9. Glossary

Original chapter author: CL. Public adaptation: CH, in my first-person project voice.
The original author incorporated an independent CH review on 2026-10-05; this adaptation is not a new joint endorsement.

Status note: experiment status is the recorded 2026-10-04 snapshot; audit and proposed-fix statements describe the 2026-10-05 review. They are not live job or deployment status.
Source paths and commands are illustrative aliases, not files or runnable commands supplied by this public guide. `my_work/` means my implementation; `ya_work/` means Ya’s foundation; `s3://<bank-storage>/` replaces the actual storage address.

- **Ablation** [4]: an experiment that changes one design choice at a time to see what it contributes.
- **Accuracy** [1.5]: fraction of correct yes/no calls; useless at 0.165% fraud.
- **AdamW** [2.9]: the optimizer that adjusts weights during training; "W" means it also applies weight decay.
- **AP, average precision** [1.7]: the metric the code computes (scikit-learn average precision); what this
  project calls PR-AUC.
- **Architecture A / flat** [4]: one transformer over all tokens of the target and its history.
- **Architecture B / hierarchical** [4]: a transformer per transaction, then a transformer over transactions.
- **Artifacts, frozen** [3]: preprocessing statistics and vocabularies computed once on training files and
  reused everywhere.
- **Attention** [2.3]: each token forms a learned, input-dependent weighted average over all tokens.
- **AUROC, ROC-AUC** [1.7]: probability that a random fraud scores above a random legitimate transaction.
- **Backpropagation / backward pass** [1.3]: computing how each weight should change to reduce the loss.
- **Batch** [1.3]: rows processed together in one training step (4,096 here).
- **bf16** [2.7]: a 16-bit number format used on the GPU to halve memory.
- **Bidirectional mask** [2.6]: every token may attend to every other; the default here.
- **Bootstrap** [1.8]: estimating uncertainty by resampling the data many times.
- **Calibration** [1.9]: making output probabilities match real frequencies; a strictly increasing
  recalibration does not change ranking, isotonic can create ties.
- **Causal mask** [2.6]: tokens may not attend to later transactions; the option in this code leaks via CLS.
- **Checkpoint** [1.4]: a saved copy of model weights (`best.ckpt`).
- **CLS token** [2.4]: the extra token whose final vector is classified.
- **Class weighting** [1.9]: counting fraud rows more heavily in the loss; `pos_weight`, unused so far.
- **Continuous feature** [1.2]: a number with magnitude, such as amount.
- **Categorical feature** [1.2]: a code from a fixed list, such as transaction type.
- **Cross-entropy** [1.3]: the loss used for yes/no predictions.
- **Downsampling** [1.9]: dropping most legitimate rows during training to raise the fraud share.
- **Dropout** [2.9]: randomly zeroing some activations during training to reduce overfitting.
- **Early stopping** [1.4]: stop training when validation performance stalls (patience 5).
- **EMA** [2.9]: exponential moving average of weights; optional, never enabled.
- **Embedding** [2.2]: turning a value into a learned vector.
- **Epoch** [1.3]: one pass over the training data.
- **Evaluation set** [1.4]: files 8000 and up, meant for the final honest number.
- **Feature** [1.2]: a column the model may use; 23 here.
- **Feature-limited** [2.8]: a plateau when more capacity, epochs or data stop helping; suggests, but does not
  prove, that the inputs are exhausted.
- **FT-Transformer** [2.5]: Feature Tokenizer + Transformer, the model family used.
- **Fraud rate / prevalence** [1.5]: fraction of fraud rows; 0.1654% in training.
- **GELU** [2.1]: the smooth activation used in the feed-forward blocks.
- **Gradient accumulation** [4]: several small batches per weight update so a large batch fits in memory.
- **Head** [2.4]: the final linear layer producing two logits.
- **History length N** [2.6]: number of prior transactions in the window; 5, 10 or 16 tried.
- **Isotonic regression / Platt scaling** [1.9]: two calibration methods.
- **Label** [1.2]: the answer column, `frd_tag`.
- **Leakage** [1.9]: using information not available at decision time.
- **Learning rate** [1.3]: step size of weight updates (3e-4).
- **Linear embedding** [2.2]: value times a learned vector plus a bias; the winning numeric embedding.
- **Logit** [2.4]: a raw score before softmax.
- **Loss** [1.3]: the number training tries to minimize.
- **Mask (padding)** [2.6]: marks empty history slots so attention ignores them.
- **Masked autoencoder** [5]: pretraining by reconstructing hidden fields.
- **MLP** [2.5]: a plain stack of layers with no attention.
- **Overfitting** [1.4]: memorizing training rows instead of learning general patterns.
- **Padding** [2.6]: empty slots added when an account has fewer than N prior transactions.
- **Periodic embedding** [2.2]: numeric value expanded into sines and cosines at learned frequencies.
- **Position embedding** [2.6]: a learned vector per transaction slot added to its tokens.
- **PR-AUC** [1.7]: this project's name for average precision; the selection metric.
- **Precision / recall** [1.6]: of the alerts, how many are fraud / of the fraud, how much is alerted.
- **Pretraining** [5]: training the encoder on a proxy task before the fraud task.
- **Residual connection** [2.3]: adding a block's input back to its output.
- **Seed** [1.8]: the number that fixes the randomness of a run; every sweep used 42.
- **Sharding** [6.2]: splitting the file list across data-loader workers.
- **Softmax** [2.3]: turns a list of numbers into positive weights that sum to 1.
- **Standard error** [1.8]: typical wobble of an estimate due to finite data.
- **Stride** [3]: use every k-th transaction as a training target (fraud never skipped); 1 here.
- **SupCon** [5]: supervised contrastive pretraining; same-label examples pulled together.
- **Time delta** [2.6]: log(1 + seconds before the target), one per transaction in the window.
- **Token** [2.2]: one feature's vector.
- **Transformer block** [2.3]: attention plus feed-forward plus normalization and residuals.
- **UNK / MISSING** [3]: reserved category indices 0 and 1 in the frozen vocabularies.
- **Validation set** [1.4]: files 4950 to 4954, used for early stopping and model selection.
- **Weight** [1.1]: an adjustable number inside the model; about 10^5 of them.
- **Weight decay** [2.9]: gentle pull of weights toward zero to limit overfitting.
- **Window** [2.6]: the target transaction plus its N prior transactions.
- **XGBoost** [1.10]: gradient-boosted trees; the production baseline.
