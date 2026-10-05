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
