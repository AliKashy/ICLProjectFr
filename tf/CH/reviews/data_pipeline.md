# TF: my data-pipeline audit record

> Public first-person adaptation of documents originally prepared by CH on 2026-10-05. I retain the findings and their original review scope; CH performed the reported independent checks. Source paths are illustrative references to my private project, with `my_work/` representing my work and Ya identifying my collaborator. Source code, private evidence files and bank data are not included in this document set. Later changes require a separate review.

Review date: 2026-10-05. Scope: every file under `fraud/data/`, both preprocessing/sequence test files, artifact-building and loader call sites, data-related notebook cells, and the written Phase 3/4 data contract. Production code was not changed. No bank data, S3 artifacts, or live training process was accessed.

The main confirmed problem is shared infrastructure rather than my architecture: with the configured four training workers, each worker shuffles the entire shard list independently before selecting its share. Some files are processed multiple times in an epoch while others are omitted. The ordinary validation/scoring path has shuffling disabled and did not show this defect in the synthetic control. My sequence code correctly carries each target's label, transaction ID and raw transaction amount through sorting and window shuffling.

## Confirmed findings and reachability

### DATA-01 — High priority: independent worker shuffles duplicate and omit training shards

- Evidence: `fraud/data/sequence_builder.py:136-143` and `fraud/data/parquet_loader.py:123-130`.
- Trigger: `num_workers > 1` with `shuffle=True` or `shuffle_files=True`. Both are used in current Phase 3, Phase 4a, Phase 4b and Phase 5 training defaults. The training scripts call `pl.seed_everything(..., workers=True)`; worker RNGs are deliberately distinct, which does not fix this partitioning error.
- Mechanism: worker 0 selects every fourth file from its own random permutation; worker 1 selects every fourth file from a different permutation, and so on. The slices are disjoint only if they all use the same permutation.
- Reproduction: a real PyTorch `DataLoader(num_workers=4)` over ten one-row Parquet files, generator seed 42, emitted ten rows for both dataset classes but only eight unique shards. Shards 3 and 6 appeared twice; shards 7 and 9 were absent. The unshuffled controls emitted every shard exactly once. A row-count-only assertion therefore misses the defect.
- Impact: an epoch does not mean one pass through all configured data. Class balance, exposure to accounts, early stopping and reproducibility are affected. This compromises how tightly the nominal 10-file/100-file experiments are controlled; it does not prove the recorded AP values are fabricated or that evaluation rows were omitted. Files omitted in one epoch can appear in subsequent epochs.
- Suggested fix: partition the stable file list by worker first, then shuffle each worker's own slice; alternatively compute one shared deterministic epoch permutation before partitioning. Preserve a separate window/row shuffle. For distributed training, include the process rank in partitioning as described below.
- Required regression: compare the multiset of emitted unique transaction IDs against the source IDs for multiple workers/seeds and unequal shard sizes. Verify no omissions/duplicates across an entire epoch, including uneven file counts.
- Temporary operational workaround: `training.num_workers=0` avoids this defect while retaining the existing row/window shuffle. This is a suggestion, not a config change made by this audit.

### DATA-02 — Medium priority, optional feature: cyclical datetime streaming raises `KeyError`

- Evidence: `fraud/data/parquet_loader.py:132-137` computes missing flags before deriving datetime features.
- Trigger: a predictor list containing continuous derived fields such as `hour_sin`/`hour_cos`, `datetime.encoding=cyclical`, and the default `add_missing_flag=true`.
- Reproduction: the matching frozen artifacts and predictor schema were built successfully, then `_encode_batch` failed with `KeyError('hour_sin')`. The sequence builder successfully encoded the same input because it derives datetime columns first.
- Current reachability: shipped predictor CSV uses categorical month/day/hour/day-of-week, so this does not invalidate the current linear-versus-periodic embedding sweep. Periodic numerical embeddings are a separate concept from cyclical calendar feature derivation.
- Suggested fix: derive the configured datetime columns before accessing their flags; decide explicitly whether invalid original timestamps should produce a missing flag for each derived field. Add a loader-level test with a predictor CSV matching the cyclical config.

### DATA-03 — Medium priority, schema-dependent: sorting raw string timestamps can admit future history

- Evidence: `fraud/data/sequence_builder.py:160-162` parses timestamps for time deltas, but `_account_groups` at `:171-184` sorts the original unparsed column. Negative deltas are then clamped at `:213`.
- Reproduction: one account with `12/31/2023 23:00` followed chronologically by `01/01/2024 00:00` was ordered January first, December second. The December target used the January transaction as history and its negative elapsed time silently became zero.
- Current reachability: genuine Arrow/pandas timestamp columns and consistently sortable ISO strings avoid this specific ordering bug. The live bank schema is unavailable, so the reproduction is not evidence that the present bank runs leaked future transactions.
- Suggested fix: parse one canonical UTC timestamp column, validate it, and use it for both ordering and deltas. Reject impossible negative deltas rather than masking them. Test unsorted input, non-ISO strings, timezones, missing timestamps and equal timestamps.

### DATA-04 — Medium priority on multiple GPUs: iterable datasets do not partition by distributed rank

- Evidence: both `_get_worker_files` methods consult only `get_worker_info()`; neither reads distributed rank/world size. Training scripts construct `pl.Trainer` without an explicit device count (`my_work/phase4a/scripts/train_sequential.py:128`, corresponding Phase 3/4b/5 call sites).
- Trigger: execution with more than one distributed trainer process, such as multiple visible GPUs selected by Lightning's automatic devices behavior. A single-GPU run is unaffected.
- Impact: separate ranks independently iterate the same global shard set; worker partitioning within a rank cannot prevent duplication across ranks. Epoch/global-batch interpretation becomes wrong, and random file sizes can create unequal per-rank step counts.
- Evidence level: static control-flow finding; no distributed trainer run was attempted. Dataset code has no rank-dependent input that could partition the data. Lightning cannot repair an `IterableDataset` by applying an ordinary distributed sampler.
- Suggested fix: either explicitly constrain the supported path to one device, or shard by a composite rank/worker identity with a deliberate equal-step policy and globally aggregated evaluation metrics. Do not silently assume that adding GPUs automatically partitions the dataset.

## Correct behavior verified or supported by the code

1. The predictor CSV contains six continuous features and 17 categorical features, including four calendar categories. With six missing flags, a transaction contains 29 original/flag features. The sequence builder adds one continuous time-delta channel, yielding seven continuous and 23 categorical channels per slot. History 5 means six transaction slots, including the target.
2. The frozen artifact pipeline uses training-fitted means/stds/medians and categorical vocabularies during training, validation and scoring. It does not refit on an evaluation batch. Unknown categories map to 0; missing categories to 1; learned categories begin at 2.
3. The baseline's `ContinuousImputer.fit` is called on `train_df` and then applied to validation/test (`fraud/scripts/train_baseline.py:117-126`). Artifact-build defaults fit files 1–5, a subset of the normal 1–10/1–100 training ranges. These nominal choices avoid direct validation fitting; actual stored artifact provenance was not available to verify.
4. The predictor CSV does not include `frd_tag`, account ID or transaction ID as model features. Historical fraud labels are not passed into the feature tensors. This follows `docs/phase4_design_decisions.md:12` and avoids the obvious use of unavailable historical outcomes.
5. Within a valid, complete account file with correctly ordered timestamps, windows use preceding row positions and keep the target in the final slot. Left padding, boolean masks, and target `[B,1]` shape agree with the intended model interface. `history_len=0` is supported by the windowing logic.
6. Stride-based training subsampling always keeps positive targets and uses all transactions, including skipped negative targets, as potential history. This is intentional. Defaults use stride 1, so no deliberate negative downsampling occurs in those runs.
7. Labels, IDs and raw amounts are gathered using the same `target_row` and then moved together through the shuffle/batching tuples (`sequence_builder.py:163-168,217-238`). A ten-target reproduction with shuffled input rows, account sorting and window shuffling verified exact alignment. Standardization returns arrays rather than overwriting raw amounts; the emitted amount is the raw target transaction amount.
8. The single-transaction loader applies the same sampled row index to continuous, categorical and label arrays, preserving their alignment during within-block shuffle.
9. File partitioning works for ordinary unshuffled single-process evaluation with multiple DataLoader workers. The synthetic four-worker control returned each of ten unique target IDs once. This is narrower than saying every possible config is safe.
10. Bidirectional attention within a target-plus-prior-history window is not intrinsically future leakage: target features and the historical transactions are observed inputs. The data window, feature availability and label maturity establish causality, not the attention-mode name alone.

## Data contracts that must be validated at the bank

These are limitations or unverified assumptions, not claims that the current real data violates them.

| Contract | Evidence and consequence | Needed validation |
| --- | --- | --- |
| Every account is fully contained in one file | Explicit design assumption at `docs/phase4_design_decisions.md:22`; the builder resets history for each file at `sequence_builder.py:245-249`. Cross-file accounts would have incomplete history. | Count distinct source files per account over the relevant raw datasets. Check each split, and clarify how a later temporal holdout receives permissible earlier history. |
| Splits represent the intended generalization question | Configs choose shard number ranges, not timestamp cutoffs. Different file numbers alone prove neither chronological separation nor account separation. | Record ID overlap, account overlap, min/max date and label maturity by split. Establish a genuinely later untouched evaluation period before final reporting. |
| Equal timestamps have an availability-safe order | The design names a transaction-ID tie-break, while `docs/phase4_sequential_plan.md:99` says to exclude transactions at or after target time. Code allows earlier tie-break IDs with equal timestamps. | Decide whether ordering means strict event time or `(event time, arrival sequence)`. A lexically smaller arbitrary ID does not prove the other transaction was available first. Test the chosen contract. |
| Valid timestamps and account IDs | Invalid times become `NaT` and then the minimum signed integer; missing account values are not rejected. Numeric subtraction on `NaT` can overflow or produce meaningless deltas. | Report and reject/quarantine invalid timestamps and absent account keys; never assume coercion makes them causally safe. |
| Labels are mature binary values of the configured type | Both loaders compute `raw == positive_label`; every other value becomes negative. Reproduction `[1, None, "1", 2]` became `[1,0,0,0]` with `positive_label=1`. | Assert accepted label values, null rate and maturity policy before supervised training/evaluation. A string-only positive column must have matching explicit configuration or normalization. Unlabeled examples must not silently become confirmed genuine transactions. |
| Feature values were available at decision time | Predictor descriptions say balances/limits are pre-transaction and external score is available, but the repository cannot verify raw lineage. | Confirm snapshot/availability timestamps for external scores, verification results and balance fields. Avoid post-decision updates. |
| Frozen artifacts match a checkpoint exactly | Manifests store versions and feature names, but not ordered schema/content hashes. Artifact builder stores only source basenames (`build_preprocessing_artifacts.py:83`) and overwrites a version directory if reused. | Save resolved feature order, artifact hash, full input URIs/versions, code revision and resolved config alongside each checkpoint; assert them on scoring. Verify my artifacts (ownership label: My) match Ya’s intended preprocessing before claiming identical reproduction. |
| Evaluation always uses stride 1 | Written design requires it, but scorer constructors accept arbitrary `eval_stride`. Stride greater than 1 keeps every fraud while skipping negatives, changing both the scored population and AP prevalence. | Reject `eval_stride != 1` in scoring/validation entrypoints. Current shipped settings are 1. |
| Continuous inputs are finite and consistently typed | Numeric coercion imputes bad strings, but loader missing flags use raw `isna()` and therefore omit coercion failures. Infinity is not converted to missing; fitting or transformation can yield nonfinite tensors. | Measure parsing failures/inf rates, define a consistent nonfinite policy, and assert finite model inputs. |
| Categorical representation is consistent across files | Vocab fitting/encoding stringify raw values. A category seen as integer `1` during fit differs from float `1.0` later. | Freeze Arrow schemas or canonicalize category representations and report unseen-category rates. This is not observed in the available synthetic numeric schema. |

The current temporal evaluation helper groups already-produced predictions by timestamp. Such a report is useful for drift analysis, but it does not itself retrain on an earlier window or create an untouched chronological holdout. That experimental distinction matters even when the plotting or bucketing code runs successfully.

## Memory and throughput limits

- `ParquetStreamingDataset` genuinely reads bounded Arrow record batches, then batches/shuffles within each block.
- `SequenceBuilder` reads an entire file into a DataFrame and encoded arrays, then holds a window shuffle buffer per worker. It is bounded by file size plus buffer size, not by the GPU mini-batch alone. That is acceptable under the account-per-file contract only if individual files fit host memory.
- For the current seven continuous/23 categorical channels, the raw window payload is approximately `T * (7*4 + 23*8 + 1) = 213*T` bytes, before labels, Python object overhead and transient copies. At 65,536 windows, that is about 80 MiB per worker for h5 and 226 MiB per worker for h16, plus each worker's full source file and DataLoader prefetch. Four workers multiply that buffer footprint. These are host-memory estimates, separate from GPU attention memory.
- The artifact builder bounds columns per batch but concatenates all selected rows for those columns (`build_preprocessing_artifacts.py:34-41,105`). `features_per_batch` is not a bound on the row count. Expanding preprocessing fitting from five files to thousands needs a deliberate online-statistics/quantile/vocabulary plan.
- The unbounded categorical vocabulary is appropriate only while feature cardinalities remain manageable. An expanded merchant/account identifier vocabulary would require a considered encoding and rare/OOV policy.

## Test and file coverage

| File(s) | Review coverage |
| --- | --- |
| `fraud/data/__init__.py` | Empty initializer. |
| `fraud/data/datetime_features.py` | Config validation, categorical/cyclical calendar derivation, missing-time behavior and periodic field periods. |
| `fraud/data/feature_list.py` | CSV schema/type/version handling, ordered feature lists and source-column resolution. Duplicate feature names, empty schema, unsupported derived sources and timestamp/config mismatches lack explicit rejection; current CSV is coherent. |
| `fraud/data/parquet_loader.py` | Every method: lazy S3 setup, worker sharding, raw columns, missing flags, encoding, label mapping, row shuffle and batches. |
| `fraud/data/preprocessing.py` | Entire imputer, continuous/categorical fitting, application, manifest and local/S3 save/load. |
| `fraud/data/sequence_builder.py` | Every method: metadata, worker sharding, encoding, per-account sorting, history selection, stride, padding, deltas, shuffling and iteration. |
| `fraud/data/s3_utils.py` | Credential loading, file naming and scheme stripping. It explicitly disables TLS certificate verification (`client_kwargs.verify=False`); use the organization's configured CA trust rather than disabling verification where supported. No credentials were read by this audit. |
| `tests/test_preprocessing.py` | All six tests reviewed: predictor parsing/versioning, categorical and cyclical derivation, imputation, encoding/save/load round-trip. They do not exercise a streaming loader with cyclical predictors, invalid labels or nonfinite raw data. |
| `my_work/tests/test_sequence_builder.py` | All nine tests reviewed: shapes, count, target position, positive deltas, padding, account isolation, first-transaction history and fraud-preserving stride. All use one local file, direct dataset iteration and nonmissing strictly increasing timestamps; they cannot detect worker sharding, rank partitioning, non-ISO/tied/missing times or cross-file account issues. |
| Notebook data paths | Reviewed code cells in `notebooks/eda.ipynb`, `notebooks/smoke_test.ipynb`, `notebooks/sequence_smoke.ipynb`. Sequence smoke manually reconstructs a sampled account's history and checks deltas, useful but limited to one file/direct iteration. No notebook was executed against S3. |
| Supporting call sites/contracts | Reviewed `build_preprocessing_artifacts.py`, train/evaluation dataset construction in baseline/streaming/sequence/Phase 5 entrypoints, default predictor and data configs, and Phase 3/4 design docs. This supports the reachability distinctions above. |

The existing test suite is useful but its “every transaction once” test asserts row count only. Ten rows can still contain eight distinct targets, as DATA-01 demonstrates. Add unique-ID/multiset checks, worker-count parity, no-shuffle/shuffle parity, raw-amount alignment and explicit malformed-data contracts before scaling experiments.

## Local reproduction artifacts

- Synthetic executable checks (private source reference)
- Captured results (private source reference)

CH executed the synthetic checks successfully. Its original private audit retains the exact commands and outputs. The public copy preserves the findings; it does not include the executable reproduction scripts.

Environment: Python 3.12, CPU PyTorch 2.14.1; dependency versions are the audit environment, not proof of the bank runtime versions. The script uses temporary local synthetic Parquet files and assertions for the observed behavior, including the reproduced defects. It makes no external calls and does not train a model. Root audit results cover the repository's existing suite separately.
