# Notebook 03 — Review + Next Actions

## Current status

The overall experiment structure is sound: forward-time evaluation, frozen targets, fixed balanced demonstrations, strictly prior same-account history, fixed feature contracts, strict structured-output parsing, and common-valid comparisons.

Do **not** redesign the experiment right now. Fix the implementation bottleneck in the demonstration-pool step, verify target batching, then continue.

---

## 1. Demonstration-pool step: current problem

The current implementation tries to prove transaction-ID uniqueness across the **entire eligible historical source** using a global groupBy/count before selecting only a tiny final demonstration set.

That forces a very large distributed shuffle and is the likely reason the Spark session becomes unstable or stops responding.

### Keep the experimental meaning exactly the same

Final demonstration pool must still be:

- pre-cutoff historical transactions only
- binary label only
- non-null transaction ID
- non-null account ID
- non-null timestamp
- no overlap with frozen target accounts
- no overlap with frozen target transaction IDs
- deterministic with seed 2026
- exactly 100 positive-label demonstrations
- exactly 100 negative-label demonstrations
- unique final transaction IDs

### Change only the computational strategy

Use:

1. Filter eligibility first.
2. Exclude frozen target accounts with a Spark **broadcast left-anti join** instead of a large Python `.isin(...)` list.
3. Exclude frozen target transaction IDs the same way.
4. Compute a deterministic hash key from `TXN_ID + SEED` separately within each label class.
5. Retrieve a bounded candidate pool substantially larger than the final requirement (for example a few thousand candidates per class, not the full source).
6. Check source-wide uniqueness only for those bounded candidate IDs.
7. Keep only candidates whose transaction ID occurs exactly once in the historical source.
8. Select the first 100 valid positive-label and first 100 valid negative-label candidates by the deterministic hash order.
9. Freeze the resulting 200-row demonstration pool.
10. Validate exact counts, label balance, target-account disjointness, target-ID disjointness, and transaction-ID uniqueness.

Do **not** globally sort or globally group every historical row just to obtain the final 200 demonstrations.

---

## 2. Copy-paste Copilot instruction for the demonstration-pool step

```text
Change ONLY the demonstration-pool creation section. Remove the full historical groupBy(TXN_ID).count() shuffle. Preserve every current eligibility/exclusion rule, but first anti-join frozen target accounts/IDs, deterministically hash-rank eligible rows by label with seed 2026, over-fetch a bounded candidate pool, then verify global uniqueness only for those candidate IDs and freeze exactly 100 positive + 100 negative demonstrations. Keep downstream artifact columns/validation compatible and do not change any later experiment logic.
```

If the current code has a large Python `.isin(target_accounts)` or `.isin(target_ids)`, replace it with small Spark DataFrames plus broadcast left-anti joins.

---

## 3. Demonstration artifact simplification

If every target uses the exact same fixed 200 demonstrations, a target-to-demonstration table with `N_targets × 200` rows is redundant.

Preferred frozen artifact:

```text
demonstration_transaction_id
demonstration_label
sampling_key
within_class_rank
prompt_order
sampling_seed
```

A single 200-row demonstration pool is enough. The downstream code can apply that same frozen pool to every target.

Do not make this simplification if it would require a large rewrite right now; fixing the shuffle bottleneck is the priority.

---

## 4. 100-example and 200-example conditions

Required behavior:

```text
100-example condition = 50 positive + 50 negative
200-example condition = 100 positive + 100 negative
```

The 100-example condition must be a nested subset of the 200-example condition.

Preserve the deterministic selection ranking from the original hash step. Do not select the 50-example subsets by later lexicographic transaction-ID ordering.

Recommended ordering inside the prompt:

```text
positive rank 1
negative rank 1
positive rank 2
negative rank 2
...
```

This avoids putting one entire label class first and the other entire class last.

---

## 5. Section 13: what still needs verification

The visible scoring-settings gate is fine. It validates that model name, worker count, and targets-per-request are populated and that numeric settings are positive integers.

The next thing that must be inspected is the code that creates the actual target ordering and batches.

Find the block containing variables/functions similar to:

```text
request_alias
batch_id
REQUEST_BATCHES
target_batch
request_records
```

### Requirement

Before batching, frozen targets should be placed into a deterministic **label-blind** order, preferably:

```text
hash(TXN_ID, seed=2026)
```

Then split that ordered list into fixed request batches.

Do **not** batch targets in the physical order of a frozen artifact if that artifact was created by concatenating positive rows and negative rows.

### Copilot audit instruction

```text
Audit ONLY the target-batching code after the scoring-settings gate. Confirm that target order is deterministic and label-blind before splitting into request batches. If it currently preserves class-concatenation or dataframe order, replace only the ordering step with a deterministic hash of TXN_ID using seed 2026, then create the same-sized batches. Do not change target IDs, labels, model settings, prompt logic, or batch size.
```

---

## 6. Label semantics vs label availability

`LABEL_COL = 1` means fraud and `LABEL_COL = 0` means nonfraud. That defines the target correctly.

A separate temporal-validity question remains:

> When was the historical label actually known?

For a historical transaction used as a labeled demonstration, transaction time being before the evaluation cutoff is not by itself sufficient if the fraud/nonfraud outcome was only confirmed later.

Before final interpretation, search metadata/source fields for something equivalent to:

```text
label_available_time
claim_time
fraud_confirmation_time
disposition_time
chargeback_time
investigation_close_time
```

If no label-availability field exists, document the limitation and determine whether the source label is defined as point-in-time available or retrospectively matured.

Do **not** change the experiment based on a guess.

---

## 7. Checks that must pass before model scoring begins

### Frozen targets

- exact requested class counts
- unique transaction IDs
- binary labels
- all timestamps within target month
- fixed seed recorded
- deterministic target order recorded separately from labels

### Frozen demonstrations

- exactly 200 unique focal transactions
- exactly 100 positive + 100 negative
- no frozen target transaction overlap
- no frozen target account overlap
- deterministic seed/hash ranking retained
- 100-example set is exactly 50 + 50 and nested in the 200 set

### History

- same account as focal transaction
- history timestamp strictly earlier than focal timestamp unless a documented same-time ordering rule exists
- focal transaction never appears in its own history
- at most configured history length
- no target account appears in demonstration history

### Features

- target label excluded from model input
- private linkage fields excluded from model input
- decision/target proxy fields excluded
- natural feature set is a subset of broad feature set
- score-time availability of the primary feature set must be defensible

### Prompt/API

- build one prompt for each configuration before launching full scoring
- record prompt bytes for every configuration
- confirm requests fit gateway/model context limits
- run a tiny execution-only pilot before high concurrency

---

## 8. Do not make these changes yet

Do not add similarity retrieval, nearest neighbors, graph retrieval, new prompt families, tabular foundation models, or new feature engineering until this frozen baseline executes correctly.

Those are later experiment axes. The current notebook should first produce a defensible controlled benchmark.

---

## Immediate sequence

```text
1. Fix demonstration-pool scalability only.
2. Freeze/validate the 200 demonstrations.
3. Audit Section 13 target ordering and batching.
4. Verify label-availability semantics.
5. Preflight one prompt per configuration for size.
6. Run a tiny scoring pilot.
7. Freeze experiment contract.
8. Run the prespecified configurations.
```
