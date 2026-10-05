# TF: my corrections and experiment plan

> Public first-person adaptation of documents originally prepared by CH on 2026-10-05. I retain the findings and their original review scope; CH performed the reported independent checks. Source paths are illustrative references to my private project, with `my_work/` representing my work and Ya identifying my collaborator. Source code, private evidence files and bank data are not included in this document set. Later changes require a separate review.

I use this backlog to plan corrections and experiments from CH’s original 2026-10-05 audit. It is not a record of fixes already applied. I will preserve the original audit and historical run artifacts separately from corrected experiments. The action lists below describe my proposed work and its acceptance criteria; publishing this plan does not implement the fixes.

## 1. Preserve the experiments before rerunning anything

For each existing h5 finder, h10/h16, h5@100 and hierarchical run, I need to recover the actual bank-side:

- Command and fully resolved YAML, source commit/hash, Python/package versions, GPU and precision.
- Training/validation/evaluation file lists, target counts, positive counts, unique transaction-ID digest, date range, and account coverage.
- Frozen feature order, category mapping/artifact hashes, target encoding, history, stride, mask and time-delta settings.
- Selected checkpoint, selected epoch, AP and AUROC at that epoch, full metric CSV and score log.
- Distinct prediction objects and the checkpoint each belongs to. Archive available S3 versions before overwriting shared paths.

**Acceptance:** a result row identifies exactly one checkpoint and one scored cohort. Resolve whether historical eval APs used 8000 alone or 8000–8010. The documented eleven-file reference has 2,466,461 rows and 3,818 positives; counts alone are insufficient without ID identity.

No source audit can substitute for this evidence. Preserve completed runs as exploratory observations if their exact training exposure cannot be recovered.

## 2. Restore reproducible installation and the original reference pipeline

Files: `pyproject.toml`, `requirements.txt`, `fraud/streaming_model.py`, active vendored checkpoint helper.

- Replace the invalid build backend with `setuptools.build_meta` after confirming the intended packaging approach.
- Declare eager runtime imports including `requests` and `pytorch-tabnet`; include intended `my_work` packages and required config assets in a built installation.
- Reconcile `InferredConfig` fields with the actual intended fork rather than mixing a recovered custom caller and stock upstream implementation silently.
- Establish explicit checkpoint serialization/loading for trusted local legacy checkpoints and record supported package versions.
- Preserve baseline run metrics outside destructive model-directory cleanup.

**Acceptance:** in a clean environment, install the project and run a tiny synthetic single-transaction baseline and streaming train/save/load/score round trip. Model predictions before and after loading must match within stated numerical tolerance. Confirm run metrics survive model saving. Verify my installed package outside the repository working directory.

Do not assume upstream v1.1.0 is byte-identical to the bank's modified library. Keep version reconciliation explicit.

## 3. Fix shared input coverage before measuring model improvements

Files: `fraud/data/sequence_builder.py`, `fraud/data/parquet_loader.py`; relevant training configs.

- Slice a stable file manifest by worker, then shuffle that worker's files, or share one deterministic epoch permutation.
- Keep the intended row/window shuffle. If multiple trainer processes are supported, include rank in the partition; otherwise explicitly limit training to one device.
- Validate canonical timestamps before sorting, accepted label values, finite features, unique transaction keys, account keys, and evaluation stride 1.
- Preserve each target's raw metadata and ensure negatives omitted as training targets remain available as history when intended.

**Acceptance tests:** with uneven synthetic shard sizes and several seeds, `num_workers=0/1/4` each emit the expected multiset of target IDs exactly once at stride 1. Row count alone is not sufficient. Test flags, sorted/unsorted input, ties under an explicit availability policy, timestamp failures, account separation, label/amount alignment, and cyclical datetime encoding if that option remains supported.

**Experiment consequence:** rerun the key baseline and best candidates on verified exposure. There is no need to immediately repeat every old knob combination before the pipeline is reliable.

## 4. Correct pretraining before scaling it

Files: `my_work/phase5_pretrain/supcon.py`, `ssl_pretrain.py`, `ssl_finetune.py`, their train/score scripts.

- SupCon: make pair-label expansion agree with feature flattening. For current sample-major flattening, repeat each label V times before building pair equality.
- SSL: retain numeric field identity when hiding values; avoid giving all masked numeric columns the same undifferentiated token plus shared decoder.
- Require explicit scratch mode. Missing/incomplete pretrained states should fail a requested pretrained run. Assert all expected encoder tensors loaded.
- Forward checkpoint hyperparameters consistently, including nondefault class weights. Preserve strict learned-weight checks.
- Define the empty-mask contract, deterministic frozen-probe behavior, accumulation settings and validation augmentation policy.

**Acceptance tests:** compare SupCon to an independent formula, with correct positives/negatives and joint permutation invariance. Demonstrate learning different masked values for different numeric fields. Verify legitimate SSL and SupCon encoder transfers tensor-for-tensor; reject incomplete transfers. Exercise weighted checkpoint reload. Check finite meaningful gradients, not only scalar loss shape.

The existing audit reproduction scripts intentionally assert current bugs. Replace those expectations with regression requirements in the project tests when implementing fixes; do not mistake a reproduction's exit code 0 for correctness.

## 5. Make each experiment and its score use one immutable identity

Files: both Phase 4 `scripts/run_ablations.py`, training/scoring scripts, configs, shared metric logger; Phase 5 equivalents.

- Resolve common and axis overrides once and reuse the same saved config for training and scoring.
- Save unique run-level checkpoint, metrics and prediction destinations; record actual output URIs in result rows.
- Explicitly select the intended validation/evaluation ranges.
- Load architecture and artifact identity from the run/checkpoint contract; do not silently reinterpret the same-shaped tensors with a different feature mapping.
- Report AP and AUROC from the selected epoch; label independent maxima separately. Save sufficiently precise machine-readable metrics rather than parsing rounded console logs.
- Record both train and score return codes and exit nonzero for required failures. Do not rank stale result files after failed reruns.
- Log the actual epoch training loss after its reduction, not the last minibatch loss.
- For EMA, save the exact averaged weights whose validation result selected the checkpoint, persist EMA state, and update on optimizer steps.
- If retaining “causal” as a tested option, remove the multilayer CLS bypass and add a two-layer dependency-invariance test. Retain a clear distinction between mask semantics and target-data availability.

**Acceptance:** a small runner integration test verifies resolved train/score config identity, unique output paths, failure propagation and checkpoint-score agreement. Test EMA separately because it is off in current runs.

## 6. Repair scoring comparisons and uncertainty

Files: `my_work/scoring/ensemble.py`, `leaderboard.py`, `calibrate_and_report.py`, `temporal_eval.py`.

- Give tied scores average ranks; metric results should not change when rows are permuted together with labels/IDs.
- Require unique and matching target IDs, consistent labels and deliberate join cardinality. Reject unintended exclusions/duplicates and overlapping calibration/test targets.
- Label per-seed results and intervals honestly. Do not attach a last-seed interval to a multi-seed mean.
- Compare paired model differences on the same cases; account for overlapping account histories and time dependencies in uncertainty estimates.
- Save fitted calibrators and ensemble weights with their fitting-cohort identity. Distinguish raw and calibrated ranking because isotonic ties can change AP/AUROC.
- Validate alert-budget fractions and report all temporal buckets, including counts and undefined metrics. A temporal breakdown is monitoring; it is not an unseen holdout by itself.

**Acceptance:** synthetic tied-score, duplicate-ID, missing-ID, label-conflict, overlap and multi-seed examples either give mathematically correct results or an explicit error. Compare metric calculations with sklearn on the exact same arrays.

## 7. Run a small, controlled scientific confirmation

My confirmation experiments need the bank environment. **CH did not execute them in this audit.**

1. Lock train, selection-validation, calibration and untouched later test cohorts. Establish point-in-time feature availability and label maturity. Keep natural fraud prevalence in validation/test.
2. Use identical predictor/artifact versions and verified training exposure. Compare the same custom flat model at h0 and the promising history lengths; include time-delta on/off. Compare hierarchical candidates with documented capacity and compute.
3. Repeat selected contenders over several seeds, rather than declaring significance from one sweep. Report per-seed metrics, paired uncertainty and practical effect sizes.
4. Compare with the existing internal XGBoost and an external-score-only control on the same targets where available. The project already includes an external fraud score as an input; quantify the incremental value of my model and history.
5. Compare corrected SSL, corrected SupCon and scratch using the same single-transaction encoder/head and data first. If useful, integrate an explicitly compatible pretrained transaction encoder into Architecture B or a deliberately designed Architecture A transfer. Align feature schemas, widths and depths; no current automatic transfer exists.
6. Select using development/validation data. Evaluate the frozen final choice once on the untouched later cohort. Report AP/AUROC, fraud prevalence, alert-budget precision/recall, calibration, amount/cost metrics where appropriate, latency, throughput and peak memory.

Use “numerically close in this one-seed sweep” until uncertainty supports stronger claims. Do not call models statistically tied, saturated, or production-superior based on the available tables.

## How CH produced the audit evidence

CH used a separate local CPU environment. Exact dependency records remain in the private audit; this was not the bank’s runtime.

CH’s original private audit retains the exact synthetic-check commands and outputs. The public copy preserves the findings; it does not include the executable reproduction scripts.

Those checks used local synthetic examples only. They collected evidence for the original source snapshot and can intentionally stop passing after a correct fix. Ordinary regression tests should assert the corrected contract instead. Installing this snapshot with `pip install -e .` currently fails for the backend reason already documented.
