# Data Split, Evaluation Design, and Pre-Scoring Audit

## Purpose

This document reviews the current in-context-learning fraud experiment and provides a checker that a local LLM or reviewer can use before final scoring.

The goal is not merely to confirm that the notebook runs. The goal is to verify that:

- the temporal split is valid,
- the target population is frozen correctly,
- historical demonstrations do not leak target information,
- same-account histories are point-in-time correct,
- feature sets are legitimate at score time,
- the 100-example and 200-example conditions are constructed as intended,
- prevalence-sensitive metrics are interpreted correctly,
- the protected test month remains a true holdout.

---

# 1. Experimental Structure

This is not a conventional supervised train/test workflow because the LLM weights are not updated. Historical labeled transactions are used as in-context demonstrations, strictly prior same-account history is used as context, and a later month is used as a protected forward-time evaluation.

```text
Historical period / development
        |
        | prompt design
        | feature design
        | demonstration design
        | retrieval design
        v
---------------- TEMPORAL BOUNDARY ----------------
        |
        v
Protected future evaluation period
```

The protected evaluation month must not be used iteratively to redesign the primary prespecified experiment.

---

# 2. Temporal Split

## Current intended split

```text
Historical source:
October 2024 through May 2025

Protected forward-time evaluation:
June 2025
```

This is methodologically preferable to a random 80/20 split for fraud detection. A random split can allow later customer behavior to influence earlier transactions, near-duplicate account patterns across train and test, temporal leakage, and unrealistic deployment estimates.

**Verdict: KEEP THIS TEMPORAL DESIGN.**

Do not replace it with a random train/test split.

---

# 3. Role of the EDA Notebook

The small number of deterministically selected Parquet files in the EDA notebook should be treated only as bounded exploratory evidence. They are appropriate for examining schemas, data types, date ranges, label status, ID quality, source overlap, and representative field values.

They should **not** define the final target or demonstration populations.

### Checker

Verify that no final target or demonstration population is accidentally derived only from the small EDA file subset.

Expected result:

```text
PASS: EDA file sampling is isolated from final target/demo creation.
```

---

# 4. June Target Population

The final evaluation population should be defined with a half-open interval:

```python
transaction_dttm >= "2025-06-01"
transaction_dttm <  "2025-07-01"
```

Eligibility should require:

```text
transaction ID is non-null
account ID is non-null
transaction timestamp is non-null
fraud label is binary {0,1}
```

The observed June population in the workbook is approximately:

```text
Eligible June transactions:     218,996,174
Fraud transactions:                 325,350
Nonfraud transactions:          218,670,824
Distinct transaction IDs:       218,996,174
Duplicate transaction-ID excess:          0
Distinct accounts:                ~5.35M
```

Interpretation:

```text
frd_tag = 1 -> fraud
frd_tag = 0 -> nonfraud
```

### Checker

Confirm directly from code/output:

- June boundaries are `[2025-06-01, 2025-07-01)`.
- `frd_tag` is restricted to 0 or 1.
- transaction ID/account/timestamp are required.
- June target IDs are unique.
- no null target labels exist in the frozen evaluation population.

---

# 5. Frozen Evaluation Cohort

The current design intentionally freezes:

```text
1,000 fraud
9,000 nonfraud
10,000 total
seed = 2026
```

The exact same 10,000 transaction IDs must be used for every experiment.

This creates paired comparisons rather than four unrelated experiments.

### Checker

Confirm:

```text
exactly 10,000 frozen targets
exactly 1,000 fraud
exactly 9,000 nonfraud
all target IDs unique
seed recorded
same frozen artifact loaded by every configuration
target artifact is never silently regenerated after scoring begins
```

A checksum or fingerprint of the frozen target artifact is strongly recommended.

---

# 6. Deterministic Target Selection

The current target sampling strategy uses a deterministic transaction-ID hash with a fixed seed and then selects exact counts within each class.

Conceptually:

```python
sampling_key = hash(transaction_id, seed)
sort within fraud/nonfraud
take exact requested N
freeze IDs
```

This is appropriate for an expensive model-evaluation cohort.

### Checker

Verify:

- the hash includes the fixed seed,
- target selection is performed independently within class,
- a deterministic tie breaker exists after the hash,
- class counts are exact,
- selected IDs are frozen before model scoring.

---

# 7. Artificial Test-Set Prevalence

The natural June fraud prevalence is approximately:

```text
325,350 / 218,996,174 ~= 0.149%
```

The frozen evaluation cohort is:

```text
1,000 / 10,000 = 10% fraud
```

Therefore the test cohort is deliberately case-control enriched. That is reasonable because a natural 10,000-row sample would contain only around 15 fraud cases on average, which is inadequate for robust model comparison.

### Metrics that remain useful on this enriched cohort

Generally appropriate for ranking discrimination:

- ROC AUC
- KS
- ROC curves
- paired score comparisons across identical targets

### Metrics that are prevalence-sensitive

Do **not** interpret these naively as production values:

- raw precision,
- average precision / PR-AUC,
- alert yield,
- fraud percentage in top X%,
- probability calibration,
- expected review productivity.

These require either sample/population weighting or a second naturally distributed evaluation cohort.

### Checker

The final report should explicitly state that the evaluation cohort is fraud-enriched.

Any prevalence-sensitive metric should be:

```text
WEIGHTED
or
LABELED AS CASE-CONTROL / ENRICHED-SAMPLE ONLY
```

---

# 8. Demonstration Population

Historical demonstrations should be restricted to transactions occurring before the protected evaluation month.

Current intended rule:

```text
transaction timestamp < June 1, 2025
```

Eligibility should require:

```text
transaction ID present
account ID present
timestamp present
binary fraud label
```

In addition, demonstration examples should not leak the final evaluation cohort.

Current strong restriction:

```text
demonstration account NOT IN any frozen June target account
demonstration transaction ID NOT IN frozen target IDs
```

This prevents labeled demonstration accounts from overlapping the target-account population.

### Checker

Confirm:

```python
set(demo_transaction_ids).isdisjoint(set(target_transaction_ids))
set(demo_account_ids).isdisjoint(set(target_account_ids))
```

Expected:

```text
True
True
```

---

# 9. Section 8 Scalability Problem

The conceptual demonstration-selection rule is valid. The current implementation is computationally fragile because it globally groups the full historical source by transaction ID in order to prove uniqueness before selecting only a tiny number of examples.

Problematic pattern:

```python
historical_population.groupBy("fi_transaction_id").count()
```

This can create an enormous Spark shuffle and overwhelm a small Spark allocation.

## Better methodologically equivalent strategy

```text
1. filter to pre-June eligible historical rows
2. exclude target accounts using a left-anti join
3. exclude target transaction IDs
4. separate fraud and nonfraud
5. deterministically hash-rank within class
6. over-fetch a bounded candidate pool
7. verify transaction-ID uniqueness only for those candidate IDs
8. retain the first 100 valid fraud and first 100 valid nonfraud examples
9. freeze the result
```

For target-account exclusion, prefer a Spark DataFrame and left-anti join instead of collecting a large Python list and using `.isin(...)`.

---

# 10. Demonstration Pool Size

The final frozen pool is intended to contain:

```text
100 fraud
100 nonfraud
200 unique demonstrations
```

This is useful because the 100-example condition can be nested inside the 200-example condition.

---

# 11. Critical Check: 100-Example Subset

The intended conditions are:

```text
100-example configuration:
50 fraud + 50 nonfraud

200-example configuration:
100 fraud + 100 nonfraud
```

and:

```text
the 100-example set is a strict subset of the 200-example set
```

### Checker

For each configuration calculate:

```python
examples_100["demonstration_label"].value_counts()
examples_200["demonstration_label"].value_counts()
```

Required output:

```text
100 examples:
0 -> 50
1 -> 50

200 examples:
0 -> 100
1 -> 100
```

Also verify:

```python
set(examples_100_ids).issubset(set(examples_200_ids))
```

must be `True`.

This is a blocking validation before scoring.

---

# 12. Demonstration Ordering

Even if the same examples are used, ordering can affect an LLM.

Avoid accidentally presenting all fraud demonstrations followed by all nonfraud demonstrations unless that is intentionally part of the experiment.

Prefer one frozen deterministic order, for example deterministic interleaving or deterministic label-blind shuffling, and use the identical rule across comparable configurations.

### Checker

Report:

```text
demonstration ordering rule
whether order is label-blocked
whether 100 and 200 conditions use the same ordering convention
```

---

# 13. Same-Account Historical Context

Same-account history should be allowed for targets because this is information that would legitimately exist at score time.

For each focal transaction:

```text
history.account_id == focal.account_id
history.transaction_time < focal.transaction_time
```

The strict `<` matters. The focal transaction itself must never enter its own history.

### Checker

For every frozen history row verify:

```python
history_account_id == focal_account_id
history_timestamp < focal_timestamp
history_transaction_id != focal_transaction_id
```

No violations are acceptable.

---

# 14. May Evaluation Rows and Earlier June Rows

Prior rows from the evaluation source may legitimately serve as history for later June targets, provided they occurred strictly before the focal transaction.

Their fraud labels should not be included in the prompt unless there is a separately justified design.

### Checker

Ensure history serialization never contains:

```text
historical frd_tag
future target label
future transaction information
```

unless explicitly authorized as a separate experimental condition.

---

# 15. Label Availability / Label Maturity

Knowing that:

```text
frd_tag = 1 means fraud
frd_tag = 0 means nonfraud
```

does not prove that the label would have been known at the time a historical transaction became a demonstration.

Example:

```text
Transaction date: May 30
Final fraud label: 1
Fraud confirmed: June 20
```

If scoring is conceptually performed before June 20, that label was not yet available.

### Search for fields such as

- fraud confirmation date,
- claim date,
- chargeback date,
- investigation completion date,
- disposition date,
- label date,
- outcome date,
- maturity date.

### Checker

The reviewer should answer:

```text
Is frd_tag contemporaneously available for all demonstration rows?
If not, what maturity lag must be imposed before a historical row can become a labeled demonstration?
```

If this cannot be established, document it explicitly as a limitation.

---

# 16. Feature Sets

The workbook compares two feature contracts:

```text
compact/natural set: 37 features
broad set: 214 features
```

The comparison is useful because it provides a small factorial design measuring feature breadth, demonstration count, and their interaction.

A feature being physically present in a Parquet file does **not** prove it is valid at transaction score time.

### Checker

Every feature used by the LLM should be categorized as:

```text
AVAILABLE_AT_SCORE_TIME
DERIVED_ONLY_FROM_PRIOR_INFORMATION
REQUIRES_REVIEW
PROHIBITED / LEAKAGE
```

No unresolved feature should remain in a primary production-style experiment.

---

# 17. Known / Decision Proxy Leakage

Explicitly verify that neither feature set contains:

- target fraud label,
- post-event disposition,
- investigation result,
- downstream decision,
- target-derived proxy,
- any field computed using future information.

### Checker

The local LLM should output:

```text
Potential leakage feature
Reason
Evidence
Recommended disposition
```

Do not rely on column names alone; descriptions and lineage matter.

---

# 18. Frozen Artifacts

Before scoring, the following should be immutable/frozen:

```text
target IDs
demonstration IDs
100-example subset
200-example pool
history mapping
feature-contract version
required feature rows
prompt version
model configuration
seed
target batching rule
```

Each artifact should ideally have:

```text
path
row count
unique key count
creation timestamp
seed/config
checksum/fingerprint
```

If a frozen artifact already exists, code should load it instead of silently regenerating it. Regeneration should require an explicit version change.

---

# 19. Target Batching

The target artifact may be stored in class-blocked order because it was constructed as fraud rows plus nonfraud rows. That storage order must not silently determine API request batches.

If requests contain five targets each, avoid all-fraud batches followed by all-nonfraud batches unless intentional.

### Preferred approach

Create a deterministic, label-blind target order using target ID + seed, then batch that frozen order.

`frd_tag` must not be used to order requests.

### Checker

Verify:

- target batch ordering does not use `frd_tag`,
- each target appears exactly once,
- all 10,000 targets are assigned,
- no duplicate aliases,
- batch size is correct except possibly the final batch,
- the batch map is reproducible.

---

# 20. Prompt-Safe Target Aliases

June target labels must never be exposed to the model.

Use aliases unrelated to fraud status.

The prompt should request one probability per alias.

### Checker

Search constructed prompts for:

```text
frd_tag
fraud label attached to target
ground-truth target outcome
class-based alias names
```

No target-label leakage is acceptable.

---

# 21. Holdout Discipline

The June test remains a valid holdout only if it is not used repeatedly to redesign the primary experiment.

Correct:

```text
Oct-May:
prompt design
feature design
demo selection design
retrieval design
hyperparameters

LOCK DESIGN

June:
final evaluation
```

Incorrect:

```text
run June
change prompt based on June result
run June
change feature set
run June
choose best
call June "test"
```

Once June results influence design decisions, later runs are exploratory/development analyses.

Maintain an experiment registry with experiment ID, prespecified/exploratory status, prompt version, feature version, demonstration policy, retrieval policy, model, seed, target cohort, and result.

---

# 22. Recommended Metric Hierarchy

For the enriched 10,000-row cohort:

### Primary discrimination metrics

```text
ROC AUC
KS
```

### Useful additional comparisons

```text
score distributions by class
pairwise score differences
bootstrap confidence intervals
API/scoring coverage
parse success rate
latency
input tokens
output tokens
```

### Use with prevalence correction / caution

```text
PR-AUC
precision
top-X% fraud rate
alert yield
calibration
fraud capture under operational review capacity
```

---

# 23. Pre-Scoring Blocking Checklist

Do not start full scoring unless every blocking item passes.

## Target population

- [ ] June boundaries correct
- [ ] exactly 10,000 targets
- [ ] exactly 1,000 fraud
- [ ] exactly 9,000 nonfraud
- [ ] unique transaction IDs
- [ ] frozen artifact exists
- [ ] all configurations use identical target IDs

## Demonstrations

- [ ] exactly 200 unique frozen demonstrations
- [ ] 100 fraud + 100 nonfraud
- [ ] no target transaction overlap
- [ ] no target account overlap
- [ ] 100-example condition is exactly 50 + 50
- [ ] 100-example set is nested inside 200
- [ ] demonstration order deterministic
- [ ] labels only exposed for demonstration focal transactions

## Histories

- [ ] same account
- [ ] strictly prior timestamp
- [ ] focal transaction excluded from its own history
- [ ] history length <= configured limit
- [ ] no target label exposed
- [ ] no future transaction included

## Features

- [ ] compact feature set validated
- [ ] broad feature set validated
- [ ] score-time availability checked
- [ ] no target label/proxy leakage
- [ ] no future-derived fields
- [ ] feature order frozen

## Scoring

- [ ] target batch order label-blind
- [ ] batch map deterministic
- [ ] prompt version frozen
- [ ] model name frozen
- [ ] temperature/top-p frozen
- [ ] retry logic does not alter prediction semantics
- [ ] output parser validated
- [ ] response aliases match requested targets

## Evaluation

- [ ] scoring coverage reported
- [ ] invalid/failed predictions reported
- [ ] ROC AUC calculated on valid predictions
- [ ] KS calculated consistently
- [ ] enriched prevalence disclosed
- [ ] prevalence-sensitive metrics weighted or caveated
- [ ] June results not used to redesign the prespecified primary test

---

# 24. Automated Assertions Worth Adding

Adapt field names to the workbook:

```python
assert len(frozen_targets) == 10_000
assert frozen_targets["fi_transaction_id"].is_unique
assert (frozen_targets["frd_tag"] == 1).sum() == 1_000
assert (frozen_targets["frd_tag"] == 0).sum() == 9_000

assert len(fixed_examples) == 200
assert fixed_examples["demonstration_transaction_id"].is_unique
assert (fixed_examples["demonstration_label"] == 1).sum() == 100
assert (fixed_examples["demonstration_label"] == 0).sum() == 100

assert len(examples_100) == 100
assert (examples_100["demonstration_label"] == 1).sum() == 50
assert (examples_100["demonstration_label"] == 0).sum() == 50

assert set(examples_100["demonstration_transaction_id"]).issubset(
    set(fixed_examples["demonstration_transaction_id"])
)

assert set(fixed_examples["demonstration_transaction_id"]).isdisjoint(
    set(frozen_targets["fi_transaction_id"])
)

assert set(demo_account_ids).isdisjoint(set(target_account_ids))

assert (
    validated_history["history_transaction_dttm"]
    < validated_history["focal_transaction_dttm"]
).all()

assert (
    validated_history["account_reference_xid"]
    == validated_history["focal_account_reference_xid"]
).all()
```

---

# 25. Local LLM Review Prompt

Copy everything below and give it to the local LLM after it has access to the notebooks.

## REVIEW PROMPT

You are acting as an independent senior data scientist and model-risk reviewer.

Audit the complete fraud in-context-learning workbook from top to bottom. Do not assume that code is correct merely because it runs. Challenge the design.

The primary experiment is intended to:

- use historical data through May 2025 for development/demonstrations,
- use June 2025 as a protected forward-time test,
- freeze 10,000 June targets consisting of 1,000 fraud and 9,000 nonfraud,
- use the exact same target IDs across every experimental configuration,
- create a fixed 200-example demonstration pool containing exactly 100 fraud and 100 nonfraud,
- create a nested 100-example condition containing exactly 50 fraud and 50 nonfraud,
- exclude all June target accounts and target transactions from labeled demonstration focal examples,
- allow only strictly earlier same-account history for each focal transaction,
- never expose June target labels to the LLM,
- compare 37-feature and 214-feature contracts,
- compare 100-example and 200-example ICL conditions,
- use a fixed seed of 2026,
- freeze all experimental artifacts before full scoring,
- evaluate the final June cohort without using June results to redesign the primary experiment.

Perform the following audit.

### A. Reconstruct the actual data flow

Give an exact diagram/table showing source population, date range, eligibility filters, sampling method, frozen artifact, role in experiment, labels available to code, and labels exposed to the LLM. Verify from executable code, not notebook prose alone.

### B. Verify the temporal split

Determine whether any future information can enter demonstrations, target histories, features, prompt text, retrieval, or evaluation. Identify every possible temporal leakage route.

### C. Verify the frozen target cohort

Prove from code/artifacts:

```text
10,000 total
1,000 fraud
9,000 nonfraud
unique IDs
June only
same IDs for every configuration
```

Determine whether deterministic hash ordering is implemented correctly and whether ties have deterministic resolution.

### D. Verify demonstrations

Prove:

```text
200 unique demonstration focal transactions
100 fraud
100 nonfraud
no target transaction overlap
no target-account overlap
pre-June only
```

Then inspect exactly how the 100-example condition is produced. Prove it contains 50 fraud and 50 nonfraud and is nested inside the 200-example pool.

### E. Review demonstration ordering

Determine whether examples are fraud-first, nonfraud-first, interleaved, or deterministically shuffled. Assess whether order differs across configurations or could confound comparisons.

### F. Verify same-account history

For every focal target and demonstration, verify:

```text
same account
history timestamp < focal timestamp
focal transaction absent from history
history length <= configured maximum
no duplicate history keys
no future row
```

Check how training-source and evaluation-source copies are reconciled.

### G. Audit label maturity

Determine whether `frd_tag` was genuinely available at the point historical transactions become demonstrations. Search schema/metadata for claim, confirmation, chargeback, disposition, investigation, outcome, label, or maturity dates. If no maturity evidence exists, state that explicitly and explain the risk.

### H. Audit all feature contracts for score-time leakage

For every feature in both the compact and broad contracts, classify it as:

```text
SAFE_AT_SCORE_TIME
SAFE_IF_DERIVED_ONLY_FROM_PRIOR_DATA
REQUIRES_REVIEW
LIKELY_LEAKAGE
```

Use feature descriptions, lineage, field names, and code. Pay special attention to unresolved fields.

### I. Verify request batching

Inspect the code after the scoring gate. Determine exactly how the 10,000 targets are ordered and split into requests. Verify `frd_tag` does not influence target order or batching. Confirm every target exactly once, no duplicate aliases, no missing targets, correct batch sizes, deterministic ordering, and label-blind aliases.

### J. Inspect prompt construction

Show precisely what information the LLM receives for demonstrations, demonstration history, targets, target history, feature names/descriptions, and labels. Prove target labels are absent. Check whether feature count or history length can exceed the model context budget.

### K. Validate enriched-prevalence interpretation

The natural June fraud prevalence is roughly 0.149%, while the frozen test cohort is 10% fraud. Explain which metrics are directly interpretable and which require weighting. At minimum assess ROC AUC, KS, PR-AUC, precision, recall, top-decile fraud rate, and calibration.

### L. Check experiment isolation

Determine whether any June scoring results have already been used to modify prompts, feature sets, demonstration selection, retrieval, history length, or model settings. If so, classify subsequent June runs as exploratory rather than pristine holdout evaluation.

### M. Check Section 8 Spark scalability

Identify whether demonstration-pool code performs a global full-population transaction-ID `groupBy` merely to select a very small final pool. If yes, recommend the smallest methodologically equivalent scalable redesign. Prefer filter -> left-anti join target accounts -> deterministic bounded candidate retrieval -> bounded uniqueness verification -> freeze exact final examples. Do not recommend collecting the full historical population to pandas or globally sorting the entire dataset using random numbers.

### N. Produce a final decision table

Return:

| Area | PASS / WARN / FAIL | Evidence | Why it matters | Exact fix |
|---|---|---|---|---|

Include at least temporal split, target cohort, target determinism, demo pool, 100-example subset, demo ordering, target/demo overlap, same-account history, label maturity, feature leakage, batch ordering, prompt leakage, context size, metric validity, artifact freezing, holdout discipline, and Section 8 scalability.

### O. Final verdict

End with exactly one of:

```text
READY FOR FULL SCORING
READY AFTER MINOR FIXES
NOT READY FOR FULL SCORING
```

Then list only the blocking fixes required before full scoring. Prefer targeted corrections to the existing experiment rather than rewriting the entire project.

---

# 26. Current Overall Assessment

Based on the reviewed code and outputs:

```text
Temporal forward-time architecture:        STRONG
Frozen common target cohort:               STRONG
Target/account leakage prevention:         STRONG
Strict prior same-account history concept: STRONG
Factorial 37/214 x 100/200 design:          STRONG
Section 8 Spark execution strategy:         NEEDS FIX
100-example balance:                        VERIFY IN CODE/ASSERT
Target request ordering:                    MUST VERIFY
Broad feature score-time validity:          MUST VERIFY
Label maturity:                             MUST VERIFY
Prevalence-sensitive metric handling:       MUST ADDRESS
```

The core split should not be replaced with a random train/test split. Remaining work should focus on validating and hardening the experiment rather than redesigning the temporal architecture.
