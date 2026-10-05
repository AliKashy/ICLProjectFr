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
