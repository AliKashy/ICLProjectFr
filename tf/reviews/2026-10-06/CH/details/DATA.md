# October 6 data pipeline review

**Status qualification:** This detailed report tests the reviewed GitHub snapshot. My later update says the bank trainers already use the corrected phase-owned builders; that makes the import finding a repository sync gap. My h0 config is created, results are pending, and the corrected knob rerun table is provisional. See the [current status](../README.md).

> Public first-person project copy prepared by CH on 2026-10-06. I keep my original findings and guides unchanged. CH performed the checks reported here. Source paths and commands are illustrative private-project aliases; code, evidence programs and the candidate patch remain in the private review record. Exact approved scientific details are retained.

Pinned source: `October-6-reviewed-source`, compared with `previous-reviewed-source`. Source was read only. No bank data, credentials, S3 access, GPU training, source modifications, commits, or pushes.

## Verdict

The new phase-owned file-sharding implementation is correct for a single training process, but **neither Phase 4 training entrypoint uses it**. That leaves the old omission/duplication defect active under the shipped four-worker training configuration. Changing only the scorer imports does not repair training; scorers use `shuffle=False`, which already avoided this particular defect. This must be fixed before describing the planned reruns as post-sharding-fix experiments.

The normal local data path is otherwise supported by useful positive evidence: target/label/amount alignment, same-account prior-row windows, padding, history lengths 0/2/16, time-delta on/off, and deterministic unshuffled full evaluation coverage passed synthetic checks. Fifteen existing preprocessing/sequence tests passed. Neither those tests nor this CPU-only audit establish that bank schemas, label maturity, account containment, artifact lineage, or GPU runs meet their required contracts.

## Findings

### OCT6-DATA-01 — High / P1: training still imports the unfixed sequence builder

**Evidence:** `my_work/phase4a/scripts/train_sequential.py:27` and `my_work/phase4b/scripts/train_sequential.py:27` import `fraud.data.sequence_builder.SequenceBuilder`. Their `_make_loader` constructs this class at line 36, and training passes `shuffle=True` at lines 88–90. Defaults use four workers (`my_work/phase4a/configs/sequential.yaml:62`, `my_work/phase4b/configs/sequential_hier.yaml:63`). The inherited method at `fraud/data/sequence_builder.py:136–143` shuffles with each worker's independently seeded NumPy state before worker slicing.

The new implementations at `my_work/phase4a/sequence_builder.py:191–202` and `my_work/phase4b/sequence_builder.py:191–202` correctly create a common fixed permutation. Only the scorer imports changed (`my_work/phase4a/scripts/score_sequential.py:28`, `my_work/phase4b/scripts/score_sequential.py:28`); scorer construction sets `shuffle=False` at line 63.

**Actual-file reproduction:** importing both trainer modules and calling their actual `_make_loader` with synthetic local Parquet, four workers, and DataLoader generator seed 42 emitted ten rows from ten one-row shards, but just eight distinct targets. Targets 7 and 9 were absent; 3 and 6 appeared twice. Both phases produce the same result. This directly exercises the real imported dataset, rather than a copied algorithm or a test-only patched module.

**Positive control:** the new phase-owned builders, directly instantiated, passed 24 controls: A/B × workers 0/2/4 × seeds 42/2026 × two epochs; all 17 targets across ten unequal-size shards appeared exactly once. Original/A/B unshuffled evaluation with zero/four workers also gave exact coverage.

**Impact:** nominal ten-file/100-file epochs still have uncontrolled account/transaction exposure in actual training. An equal output row count cannot detect this. Existing AP measurements are not thereby fabricated, and this does not prove evaluation rows were omitted; it means the claimed corrected training cohort and fair rerun conditions are not established.

**Recommended change:** use the corresponding phase-owned builder in both trainers and add regression tests through each real `_make_loader`, checking source/emitted ID multisets across several worker counts and unequal shard sizes. The existing `my_work/tests/test_sequence_builder.py:27` still imports the original class, uses one file/direct iteration, and its exact-once test at lines 155–158 only checks total count. For unchanged Phase 3/5 paths, the streaming loader has the same defect (`fraud/data/parquet_loader.py:123–130`); use a phase-owned corrected loader or supported single-worker-process workaround before interpreting those experiments.

**Minimal candidate prepared, not applied:** `phase4_training_builder_imports.patch` changes only those two training imports. `git apply --check` passes on the pinned tree. The harness temporarily substitutes the imported class symbol (using `patch.object`, then restores it) and exercises each actual `_make_loader` body with four workers over 17 targets in ten unequal shards. A/B × two seeds × two epochs gives eight passing exact-ID coverage controls. This is a clearly labeled import-substitution simulation, not a claim that the repository is already fixed or a full retraining run. No timestamp, label, rank, or artifact behavior is changed by that patch.

### OCT6-DATA-02 — Medium / P2, schema-dependent: parsed timestamps are not used for ordering

**Evidence:** both new `sequence_builder.py` copies parse timestamps at lines 220–221 but sort the original raw column at lines 232–238. Negative deltas are silently clamped at line 270. Equivalent inherited code remains in `fraud/data/sequence_builder.py:160–179,213` and is what training currently uses.

**Reproduction:** for one account with `12/31/2023 23:00` (ID 0) and `01/01/2024 00:00` (ID 1), both new builders emit targets in order `[1, 0]`; the December target then has the later January transaction as history and reports its elapsed time as zero. In a second case a missing timestamp is accepted and causes `overflow encountered in scalar subtract`, yielding a history log-seconds feature of approximately 43.6683.

**Scope:** native Arrow/pandas timestamp columns or consistently lexically sortable ISO text avoid the non-ISO ordering case. The repository provides no actual bank schema/invalid-timestamp counts, so this is an executable conditional defect, not proof of real-data leakage.

**Recommended change:** parse/validate one canonical timestamp representation before both sorting and delta computation; reject or quarantine missing/invalid times and negative elapsed values. Test timezone policy, invalid times, non-ISO text, and subsecond ordering.

### OCT6-DATA-03 — Medium / P2, data-contract disagreement: equal-time history is allowed despite the updated strict-time plan

**Evidence:** `docs/phase4_sequential_plan.md:103` requires `time < target_time`. Both builders sort by account/time/tie-break at lines 232–238, then take preceding row positions at line 257, with no strict timestamp filter.

**Reproduction:** two transactions from one account with the same timestamp produce a second window containing the first transaction as history; its delta is zero. The configured tie-break is transaction ID, which makes ordering deterministic but does not independently prove availability before the target.

**Scope:** this can be valid if the actual contract is a proven decision/arrival sequence rather than strictly earlier timestamps. It is not automatically evidence of future leakage. The written contract and implementation need one explicit shared definition.

**Recommended change:** either enforce strict earlier time or document and validate an arrival-safe sequence key that can order ties. Include tied-time and delayed-arrival tests.

### OCT6-DATA-04 — Medium / P2 only if distributed training is used: no rank partitioning

The new `_get_worker_files` methods (`my_work/phase4a/sequence_builder.py:191–202` and the matching 4b lines) consult DataLoader worker identity but never distributed rank/world size. Both trainers construct Lightning with `accelerator` but no explicit `devices=1` (`train_sequential.py:125–137` in each phase). A rank simulation shows rank 0 worker 0 and rank 1 worker 0 choose the identical list `[8,5,7,9,3]` from ten shards with two workers. This is static/control-flow corroboration, not a multi-GPU Trainer test. Fixing local worker partitioning alone does not support multiple distributed processes.

For the currently expected single-GPU execution this does not trigger. Before scaling, either explicitly constrain execution to one device or implement composite rank/worker partitioning, equal-step policy, and correctly aggregated validation metrics.

## Inherited contracts still requiring evidence at the bank

These are narrower than claiming all existing data are bad.

| Contract | Current code behavior and reproduced evidence | Required project evidence/action |
|---|---|---|
| Labels are mature, binary, and correctly typed | Both new builders at line 219 map every nonmatching value to zero; `[1, null, "1", 2]` becomes `[1,0,0,0]`. The inherited/streaming loaders do the same. | Assert accepted raw label values/type and maturity cutoffs before training/evaluation. Unlabeled rows must not silently become confirmed nonfraud in supervised datasets. |
| Numeric inputs are finite and coercion flags are meaningful | `fraud/data/preprocessing.py:188–191` retains infinity. New builders set flags from raw `isna()` at lines 214–217. `inf` reaches a continuous tensor; malformed numeric text is imputed but receives missing flag 0. | Validate/coerce nonfinite inputs consistently, count parse failures, assert finite tensors and appropriate flags. |
| Every account is wholly contained in a file | Explicit assumption in `docs/phase4_design_decisions.md`, carried forward by per-file iteration at new builder lines 300–304. Two same-account transactions in separate files have real-slot counts `[1,1]`; in one file `[1,2]`. | Verify distinct source-file count per account over the relevant split. A future temporal holdout needs an explicit admissible historical-context policy. |
| Split boundaries match the generalization question | Configs select file-number ranges; changing eval end to 8010 expands the intended range but does not establish chronological or account separation. | Save ID/account overlap counts, date ranges, maturity cutoffs, and as-of feature availability. Account overlap may be intentional for future transactions of known customers; choose the intended question explicitly. |
| Artifacts match every checkpoint | Artifact fitting defaults to training files 1–5; no apparent validation fitting in that config. Manifest has feature names/version and source basenames, no ordered schema/content checksum (`build_preprocessing_artifacts.py:80–84`; `preprocessing.py:194–202`). Feature order comes from the CSV. | Verify My versus Ya's artifact content, archive hashes/ordered schema/resolved config with checkpoints, reject mismatches at score time. Equal dimensions alone do not prove semantic equivalence. |
| Evaluation is the natural full target population | Both changed scorers still accept arbitrary `eval_stride` (`score_sequential.py:60`); stride logic keeps all positive targets and subsamples negatives (`sequence_builder.py:252–256`). Current configs use 1, which is correct. | Reject evaluation stride values other than 1. Otherwise prevalence and AP population change. |
| Optional cyclical datetime configuration works | Unchanged `fraud/data/parquet_loader.py:132–137` accesses missing flags before deriving fields; supported cyclical predictors trigger `KeyError('hour_sin')`. | Fix before using cyclical datetime features in streaming/Phase 5. Current default calendar categories and periodic numerical embeddings do not trigger this distinct option. |

The shared deterministic file seed remains fixed across epochs and is not wired to `training.seed`. This is an explicitly documented trade-off, not a missing/duplicate sample error: within-worker window shuffling still changes sample order. A future shared epoch seed can add diversity without reintroducing independent pre-partition shuffles.

## Positive evidence and limits

- All normal fixture windows matched an independent expected same-account transaction list, including left padding and final target slot, h0/h2/h16, and time-delta enabled/disabled. Label and raw amount metadata stayed aligned after input/window shuffling.
- Historical `frd_tag`, account IDs, and transaction IDs are not in the configured predictor list. The model gets six continuous features, 17 categorical features, six missing flags, and optionally one extra continuous time delta; no historical label channel is constructed.
- Frozen preprocessing is applied, not fitted, inside the loaders. Categorical missing/OOV reservations and finite-value median standardization are internally coherent on valid fixtures; existing serialization/encoding tests pass.
- `history_len` and time-delta changes propagate from current training/scoring config to dataset construction. Expanded evaluation range 8000–8010 is present in both base configs.
- New phase-owned copies are byte-identical to each other. Compared with the original, executable differences are fixed-seed sharding and its constructor field, plus cosmetic variable/comment changes. They intentionally inherit the other data contracts.
- The sequence builder loads an entire file into host memory and materializes window buffers. It is not bounded solely by mini-batch size. At 30 tokens/transaction, h20 raw buffer payload is roughly 280 MiB per worker before Python objects, the full source file, arrays, and prefetch. Four workers multiply that cost. No live host-RAM evidence was available.

## Checks and reproducibility

From the pinned repo:

```bash
PYTHONPATH=. <audit-python> <private-evidence>/data/verify_data.py
PYTHONPATH=. <audit-python> -m pytest tests/test_preprocessing.py my_work/tests/test_sequence_builder.py -q -p no:cacheprovider
```

- `verify_data.py`: exit 0; detailed `results.json`; empty `stderr.txt`. Assertions verify both good controls and reproduced defects.
- Existing pytest: **15 passed**, 99 NumPy timedelta deprecation warnings, 1.58 seconds. Tests exercise the original sequence class, not either new phase-owned class.
- Runtime: Python 3.12 audit environment, torch 2.14.1+cpu, pandas 2.3.3, NumPy 2.5.3. These are not assertions about bank dependency versions.
- `results.json`, `pytest.stdout`, `pytest.stderr`, and the harness are retained in the private evidence record. No external data were read.

## Examined changed files

Full data-path review:

- `my_work/phase4a/sequence_builder.py`
- `my_work/phase4b/sequence_builder.py`
- `my_work/phase4a/scripts/score_sequential.py`
- `my_work/phase4b/scripts/score_sequential.py`
- `my_work/phase4a/configs/sequential.yaml`
- `my_work/phase4b/configs/sequential_hier.yaml`
- `my_work/phase4a/configs/ft_decisions.yaml`
- `my_work/phase4a/configs/ft_decisions_flat_long.yaml`
- `my_work/phase4a/configs/ft_decisions_capacity.yaml`
- `my_work/phase4a/configs/ft_decisions_timedelta.yaml`
- `my_work/phase4b/configs/ft_decisions_hier.yaml`
- `my_work/phase4b/configs/ft_decisions_hier_h16.yaml`

Changed files reviewed for relevant scope, not claiming another agent's full ownership:

- `my_work/phase4a/scripts/run_ablations.py` and `my_work/phase4b/scripts/run_ablations.py`: train/score command construction and override forwarding.
- `my_work/phase4a/sequential.py` and `my_work/phase4b/sequential_hier.py`: data interface diff only; model/attention verification assigned separately.
- `docs/phase4_sequential_plan.md`: current correctness claims, sampling/ties, evaluation/data contract and next-step context.
- `Update Overview/Changes Made - 2026-10-06 11-02 AM ET.md`: full update claims, reconstructed-chat versus source distinction.

Supporting unchanged files read: both Phase 4 train scripts; all of `fraud/data/{parquet_loader,sequence_builder,preprocessing,datetime_features,feature_list}.py`; `fraud/scripts/build_preprocessing_artifacts.py`; `fraud/configs/{predictors.csv,preprocessing.yaml}`; `docs/phase4_design_decisions.md`; `my_work/tests/test_sequence_builder.py`; prior audit `CH/reviews/data_pipeline.md`, its check harness/results. Phase 5 loader/import/config call sites were traced by search and both pretrain loader scripts read, establishing inherited loader use; no claim of complete Phase 5 algorithm review in this report.

## Old finding disposition

| Earlier finding | Current disposition |
|---|---|
| DATA-01 worker file overlap/omissions | **Implemented fix passes direct tests, but still outstanding in real Phase 4 training due to old imports; also remains in streaming Phase 3/5.** |
| DATA-02 cyclical streaming `KeyError` | Unchanged; reproduced, optional configuration only. |
| DATA-03 raw-string timestamp ordering | Inherited into both new copies; reproduced, schema-dependent. |
| DATA-04 distributed rank partitioning | Unchanged in new copies and trainers; conditional on multi-process execution. |
| Account containment, tied-time semantics, label quality, artifact lineage | Still unverified bank-data contracts; tied-time code still contradicts literal updated strict-time plan. |
