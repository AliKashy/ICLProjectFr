# TF: findings from my CH audit

> Public first-person adaptation of documents originally prepared by CH on 2026-10-05. I retain the findings and their original review scope; CH performed the reported independent checks. Source paths are illustrative references to my private project, with `my_work/` representing my work and Ya identifying my collaborator. Source code, private evidence files and bank data are not included in this document set. Later changes require a separate review.

**My work broadly follows Ya’s intended direction, and the history architectures are real implementations. It is not yet correct enough to sign off all experiments or claim a final winner.** The review found reproducible data-loading and pretraining bugs, mismatches between documented experiments and executable configs, and problems reproducing the original pipeline from this recovered checkout.

The distinction matters: working code, passing shape tests, accurate photo transcription, and a scientifically valid experiment are four different claims. This audit provides evidence for specific code behavior; it cannot certify the unseen bank runs.

## Scope and evidence

- Source: the original private TF snapshot reviewed on 2026-10-05. Exact source identity remains in the private audit.
- Date: 2026-10-05 UTC. The standing document's results are a **2026-10-04 23:09 pod-time snapshot**, not live status.
- Inventory: 225 tracked files, including 146 Python files, 14 YAML configs, three notebooks, and four existing test modules.
- Detailed review covered the project plans; application data, training, scoring and callback paths; all my Phase 4a/4b/5 files and grids; scoring tools; tests; and reconstruction provenance. Active FT-Transformer/library integration was examined. Unused vendored model families received dependency/static coverage, not exhaustive behavioral validation.
- All 146 Python files parsed, critical Ruff checks reported zero issues, all YAML configs parsed, notebook code cells parsed, and all three shell scripts passed `bash -n`.
- **All 28 existing tests passed** on CPU with a fresh dependency environment. Separate targeted synthetic checks reproduced the findings below and tested several positive invariants.
- Bank data, actual preprocessing objects, original run CSVs, model checkpoints, prediction parquet files, pod logs and H200 execution were unavailable. No bank credentials or S3 data were accessed. Original source photos were not re-inspected in this audit.

The exact source hashes and raw test outputs remain in the private audit. The public detailed reviews below preserve the scope, observed values and qualifications. The reconstruction metadata itself says source recovery does not prove runtime correctness; several older docs/notebooks still have fidelity review pending.

## Highest-priority findings

### 1. Training workers repeat some input files and miss others

**Confirmed; affects current multiworker training defaults; shared with Ya’s pipeline.**

`fraud/data/sequence_builder.py:136` and `fraud/data/parquet_loader.py:123` shuffle the complete file list separately in each worker and then take worker-specific slices. Those slices do not form one disjoint partition because the permutations differ.

A real four-worker DataLoader over ten synthetic parquet shards emitted ten rows, but only eight unique source shards: two were duplicated and two omitted. Both dataset classes showed the defect; unshuffled evaluation controls covered all ten exactly once.

This changes accounts/examples seen during an epoch and weakens “same 10/100 files” comparisons. It does not prove the reported evaluation APs are fabricated or that evaluation itself omitted rows. Some omitted training files may appear in later epochs.

**Fix first:** partition the stable file list before shuffling each worker's portion, or use one shared epoch permutation. Include distributed rank if supporting multiple GPUs. Verify exact unique-ID coverage, not just row count. A temporary single-worker setting (`training.num_workers=0`) avoids this specific bug at the cost of throughput.

Evidence: [data review, DATA-01](reviews/data_pipeline.md), captured reproduction (private source reference).

### 2. The recovered original pipeline is not cleanly reproducible

**Confirmed startup/integration failures in this snapshot; not proof the bank environment failed.**

| Problem | Evidence | Consequence / correction |
|---|---|---|
| Invalid build backend | `pyproject.toml:3`; actual editable-install failure | `setuptools.backends._legacy:_Backend` cannot import. Use the supported setuptools build backend and verify a clean install. |
| Undeclared mandatory dependencies | `src/pytorch_tabular/utils/data_utils.py:8`; `models/tabnet/tabnet_model.py:10` | Requirements-only baseline/streaming imports fail. `requests` and `pytorch-tabnet` are required by eager imports. Installing them in the audit environment allowed imports. |
| Original streaming builder expects a different library API | `fraud/streaming_model.py:34`; vendored `config/config.py:206` | `InferredConfig` rejects `output_cardinality`. Align the recovered builder with the intended fork/API before reproducing Phase 3. |
| Legacy checkpoint loader conflicts with modern Torch defaults | `src/pytorch_tabular/utils/python_utils.py:84`; `tabular_model.py:1494` | Default weights-only loading rejects the saved OmegaConf checkpoint metadata. A tiny synthetic baseline trained and then failed restoring its best checkpoint. Define an explicit trusted-checkpoint loading policy or safe serialization format. |
| My packages omitted from distribution discovery | `pyproject.toml:31` | Package discovery includes only `pytorch_tabular*` and `fraud*`. Source-checkout scripts can work while a built distribution omits my work. Include intended packages/config data and test the installed artifact. |

The core vendored files were imported/pinned from upstream according to Git history; some were not recovered from bank-source photographs. The streaming API mismatch may be a reconstruction/fork integration problem. Preserve that distinction before assigning blame or rejecting historical results.

Evidence: install failure (private source reference), requirements-only imports (private source reference), [original-project review](reviews/original_project.md).

### 3. SupCon trains against incorrect view/label pairs

**Confirmed mathematical bug in my Phase 5 path.**

`my_work/phase5_pretrain/supcon.py:37` flattens views in sample-major order, while `:40` repeats labels as though the views were ordered differently. For labels `[0,1]`, flattened features belong to `[0,0,1,1]`, but the pair mask treats them as `[0,1,0,1]`.

In a controlled example, correctly separated fraud/nonfraud views received loss **14.2857**, while deliberately cross-paired views received approximately **0.000001**. A correct reference loss reverses those results. The current loss is finite and backpropagates, so the existing tests pass.

**Fix:** match labels to flattened views, e.g. sample-major labels expanded with `repeat_interleave(V)`. Verify against a known reference and class-pair examples. Rerun any affected SupCon pretraining; changing the sampler does not correct the loss.

Evidence: [Phase 5 review, P5-01](reviews/phase5.md), synthetic values (private source reference). Existing photo provenance supports that this code was faithfully transcribed; the audit did not independently reopen those photos.

### 4. Masked SSL cannot distinguish simultaneously hidden numeric fields

**Confirmed representational defect in my Phase 5 path.**

`my_work/phase5_pretrain/ssl_pretrain.py:65` replaces whole field tokens with one shared mask token. There is no separate field identity after replacement, and `:77` uses one shared numeric decoder. Two hidden numeric fields therefore receive identical inputs and context and, with dropout off, identical predictions.

The synthetic check masked fields whose correct values were +2 and −2; the model necessarily predicted **0.5007388 for both**. It still returned a loss and gradients.

**Fix:** retain learned field identity while masking the value, or deliberately design field-specific decoders. Test distinct masked-column reconstruction. Rerun the affected SSL pretraining after the objective is repaired.

Evidence: [Phase 5 review, P5-02](reviews/phase5.md), synthetic values (private source reference).

### 5. Scoring defaults select one file while the reported comparison claims eleven

**Confirmed configuration/documentation mismatch; historical run impact unverified.**

Both `my_work/phase4a/configs/sequential.yaml:83` and `my_work/phase4b/configs/sequential_hier.yaml:90` evaluate 8000–8000. No current Phase 4 sweep widens that range. The ready h5@100 scoring command in `my_work/docs/my_current_standing.md:142` also omits the override, yet the document describes an eleven-file evaluation set. Ya’s documented Phase 3 comparator uses 8000–8010.

APs on different transaction populations are not interchangeable. The bank commands may have used local overrides absent here; no raw score logs are available to resolve this.

**Fix:** recover the exact historical config/command and cohort counts/IDs. For the documented eleven-file comparison, explicitly use `eval.file_start=8000 eval.file_end=8010`, and verify identical target IDs and labels across models. Do not assume two APs of 0.0940 refer to the same baseline: the original smoke and streaming documents report that rounded value on different evaluation sizes.

Evidence: [Phase 4 review, P4-01](reviews/phase4.md), [evaluation review, E1](reviews/evaluation_status.md).

### 6. Ablations overwrite earlier prediction sets

**Confirmed output-path behavior in the current runners; no external write performed during audit.**

Both Phase 4 scorers write `<eval.predictions_base>/sequential_preds.parquet` (`scripts/score_sequential.py:125`). The runners give each run a new checkpoint folder but reuse the prediction prefix (`scripts/run_ablations.py:134`). Consequently, successive runs within a phase replace earlier predictions. Flat h5/h10/h16 and the h5@100 command also share the flat destination.

Scalar result rows and checkpoints may survive; the default prediction object cannot identify all those runs. This affects later paired comparisons, uncertainty estimates, calibration and ensembles.

**Fix:** preserve existing outputs, assign a distinct immutable prediction URI to every run, and store that URI plus checkpoint/config/data identities in the result row.

Evidence: [Phase 4 review, P4-02](reviews/phase4.md).

## Other confirmed issues and conditional failures

These matter, but should not all be described as defects affecting the published default runs.

| Finding | Scope / observed consequence | Detailed evidence |
|---|---|---|
| Causal mask has a CLS backchannel | With two or more layers, earlier transaction states see later ones through CLS. The causal ablation is not strictly causal. This does **not** establish post-target or label leakage in current target-only scoring; bidirectional remains legitimate. | [P4-03](reviews/phase4.md); inherited original flat mask |
| EMA validates averaged weights but saves raw weights | Actual Lightning probe validated weight 0.5 and saved weight 1.0. Current configs have EMA **off**. Accumulation also changes its per-step decay behavior. | [P4-04](reviews/phase4.md) |
| Weighted checkpoints fail in some scorers | Nondefault `pos_weight` registers a loss buffer omitted when rebuilding the original sequential/streaming and Phase 5 scoring models. Default weight 1 avoids this. My Phase 4 factories forward the setting. | [Original review](reviews/original_project.md), [P5-03](reviews/phase5.md) |
| Pretraining can silently fall back to random initialization | Missing checkpoint path warns and continues; empty/incomplete `encoder.*` states are accepted with `strict=False`. A run can be labeled pretrained without loading the full encoder. | [P5-05](reviews/phase5.md) |
| Training loss CSV is not an epoch curve | Synthetic batches with losses 1 and 9 produce CSV `train_loss=9` even though epoch mean is 5. AP checkpoint selection is unaffected by this logging defect. | [Independent verification](reviews/independent_verification.md) |
| Summary AP and AUROC can come from different epochs | h10's AP-selected epoch has AUROC 0.946059, while its displayed maximum AUROC 0.948879 is another epoch. Label maxima separately. | [P4-06](reviews/phase4.md), [E10](reviews/evaluation_status.md) |
| Ensemble tie ranking fabricates ordering | Equal scores with true AP/AUROC 0.5/0.5 became 1.0/1.0 after double-argsort ranking in a synthetic row order. Use average ranks for ties. | [E3](reviews/evaluation_status.md) |
| Comparison tools do not enforce cohort identity | Inner joins can silently drop targets or duplicate them; label conflicts are ignored. Leaderboard does not compare ID sets. | [E4](reviews/evaluation_status.md) |
| Multi-seed confidence interval describes only the last seed | Leaderboard displays mean metrics but bootstraps only its last prediction file. Account/window dependence is also ignored. | [E5](reviews/evaluation_status.md) |
| Runner failures and override precedence are unsafe | Required subprocess failures need not make the overall shell run fail. Train and score apply conflicting common/axis keys in opposite orders; current canonical grids avoid that collision. | [P4-05](reviews/phase4.md), [E11](reviews/evaluation_status.md) |
| Optional cyclical datetime streaming fails | Matching derived continuous predictors plus missing flags raise `KeyError('hour_sin')`. Current categorical datetime defaults avoid it. | [DATA-02](reviews/data_pipeline.md) |
| Non-sortable string timestamps can put future rows in history | Synthetic non-ISO strings reproduce it; actual typed timestamp columns avoid this particular failure. Bank-schema impact unknown. | [DATA-03](reviews/data_pipeline.md) |
| Multiple GPU ranks are not partitioned or globally evaluated | Static gap; documented single-GPU use avoids it. Loss synchronization alone does not produce global AP. | [DATA-04](reviews/data_pipeline.md), [independent review](reviews/independent_verification.md) |
| Baseline model save deletes its metrics CSV | `train_baseline.py` writes inside the model-save directory; vendored save code clears files there. Preserve metrics outside that directory. | [Original review](reviews/original_project.md) |
| Inherited FT feature importance uses the opposite attention direction | The optional importance calculation reads feature-to-CLS rather than CLS-to-feature attention. This affects interpretation, not the forward prediction. | [Original review](reviews/original_project.md) |

Detailed reports also cover empty-mask SSL batches, active dropout in frozen probes, unchecked enum/config values, ignored accumulation settings, saved-artifact/schema fingerprints, finite input/label validation, equal-time ordering, cross-file account assumptions, host/GPU memory scaling, stale commands, insecure certificate-verification configuration, and small documentation arithmetic errors. These are not omitted just because they are outside the top six.

## What is right about my work

- The flat and hierarchical models are implemented and their shapes and training interfaces are coherent.
- Historical fraud labels and account/transaction IDs are not ordinary predictor inputs. This follows Ya’s no-historical-label design.
- The shared preprocessing uses frozen training statistics and categorical vocabularies. Its nominal fit range is within the training range.
- For valid complete account files and correctly typed timestamps, sequence construction uses preceding transactions and scores the target in the final slot. Stride keeps fraud targets and skipped negative targets can still contribute context.
- Synthetic probes confirm IDs, labels and **raw** transaction amounts stay aligned through sorting/window shuffling.
- Altering masked padding values does not change target predictions in the tested flat/hierarchical, causal/bidirectional, linear/periodic cases.
- The single-process metric callback pools the full validation set and resets between epochs. This is the right approach for rare fraud.
- Valid SSL/SupCon encoder states transfer exactly into the matching fine-tuner. The problem is the objectives and integration/verification contracts, not a universal failure to transfer weights.
- Reducing physical batch size while accumulating gradients is a reasonable response to long-history GPU memory pressure.

## Which conclusions are supportable now

My reported Phase 4 validation AP improves from 0.1365 at h5 to 0.139942 at h10, with h16 at 0.138256. The h5 model trained on 100 files reports 0.141585. These are promising recorded observations, subject to the provenance and loader caveats above.

The available evidence does **not** establish that h10 and h16 are statistically equivalent, that only embeddings matter, that flat architecture is saturated, that Architecture B won, or that pretraining improved fraud detection. One seed and a repeatedly consulted development evaluation set cannot establish those claims. A temporal breakdown of the same scores does not create an untouched temporal holdout.

The Phase 5 defaults also compare different validation ranges, and pretraining is not connected to either Phase 4 history model. Compare corrected SSL/SupCon to a matched scratch single-transaction control first, then implement explicit compatible transfer into the chosen history architecture if that is the goal.

AP around 0.14 is **not 14% accuracy or recall**. It is a ranking summary. Report precision and recall at the bank's alert budget, calibration, and cost/value metrics on a fixed natural-prevalence cohort alongside AP/AUROC.

## Recommended order of work

1. Preserve the existing bank-side configs, logs, checkpoints and predictions, each with a run identity. Resolve one-file versus eleven-file evaluation before further comparisons.
2. Repair the recovered environment/library integration and shared loader partitioning. Add end-to-end synthetic checks before spending more GPU time.
3. Repair SupCon and masked SSL objectives. Keep their previous results separate from corrected runs.
4. Compare matched custom-model h0/h5/h10/h16 and hierarchical candidates, with the same data/artifacts and several seeds. Include time-delta on/off to identify what adds signal.
5. Define an untouched later-in-time, mature-label test cohort. Verify target ID overlap, account distribution, permissible history and feature availability. Compare with the existing bank XGBoost on those same targets.
6. Report paired uncertainty using account/time structure, per-run resources, operational metrics and final-test results. Fit calibration/ensembles only on designated development/calibration data.

My [implementation backlog](NEXT_STEPS.md) turns this sequence into concrete proposed tasks. CH applied no source fixes in this audit.
