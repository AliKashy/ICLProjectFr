# TF: my Phase 4a / 4b audit record

> Public first-person adaptation of documents originally prepared by CH on 2026-10-05. I retain the findings and their original review scope; CH performed the reported independent checks. Source paths are illustrative references to my private project, with `my_work/` representing my work and Ya identifying my collaborator. Source code, private evidence files and bank data are not included in this document set. Later changes require a separate review.

Audit date: 2026-10-05. This review examines the checked-in snapshot, not the bank pod. Source was not changed. No training on bank data or S3 access was performed.

## What my code is doing

Each example contains a current transaction and the same account's last N transactions. Frozen preprocessing converts its numeric and categorical fields into numbers. The model learns a small vector, called a token, for each field. The output is a two-class fraud/non-fraud score. My history code imports Ya’s `fraud.data.sequence_builder.SequenceBuilder`, so correctness of histories and preprocessing remains dependent on that shared implementation; see the separate data review.

**Architecture A / Phase 4a:** every field of every transaction enters one transformer. With F fields and N history transactions, the attention sequence is `1 + (N+1)*F` tokens. A special CLS token collects information for the fraud decision. A transaction-position vector tells the network which transaction a field belongs to. There is no actual SEP token in this implementation, despite the SEP name in documentation; positions and a fixed token count provide transaction boundaries. This is a naming discrepancy, not evidence that the model cannot work.

**Architecture B / Phase 4b:** first, one shared transformer compresses each transaction's fields into a single vector; second, another transformer combines those transaction vectors. It is implemented, not merely proposed. The second stage has `1 + (N+1)` tokens. Rough attention cost changes from `O(T²F²)` to `O(TF² + T²)` per example, excluding dimensions/layer counts. This gives longer histories a much more affordable path, although first-stage memory and computation still grow with transaction count. Its transaction encoder starts from random weights; there is no loading of an existing single-transaction checkpoint in Phase 4b.

`history_len=5` means 5 previous transactions plus the current one. `history_len=16` means 17 total. “100-file” is the number of training parquet files, not a history of 100 and not 100 evaluation runs.

The linear embedding is the same per-field linear tokenizer as Ya’s base. The periodic option adds learned sine/cosine features. Both use the same overall supervised objective. AdamW versus RAdam changes the optimizer. Gradient accumulation allows the long flat sequences to fit in memory while approximately preserving a 4096-example optimizer batch.

## Findings that affect use of the checked-in code

### P4-01 — Checked-in evaluation commands use one file, while the reported benchmark is eleven files

**Priority:** high for benchmark comparability. **Confidence:** confirmed configuration discrepancy; effect on past pod runs unverified.

- `my_work/phase4a/configs/sequential.yaml:81-87` sets evaluation to 8000–8000.
- `my_work/phase4b/configs/sequential_hier.yaml:88-94` does the same.
- None of the three Phase 4 sweep specs overrides the evaluation range.
- `my_work/docs/my_roadmap.md:24` describes 8000–8010, and `my_work/docs/my_current_standing.md:79,154` describes the already-inspected eleven-file set.
- The ready-to-run 100-file h5 scoring command at `my_work/docs/my_current_standing.md:144-149` does not override the evaluation range.

Running those checked-in instructions evaluates only file 8000. Its AP is not directly comparable with a baseline evaluated over 8000–8010, especially with rare fraud. This does **not** prove the pasted results were computed on the wrong range: a pod-local YAML or CLI override may differ, and the raw logs/config manifests are absent.

**Action:** establish and record the intended fixed evaluation range; compare models on exactly the same transaction IDs/labels and preprocessing artifact version. For the documented eleven-file benchmark, explicitly set `eval.file_start=8000 eval.file_end=8010`. Recover original score logs and check `n`, positives, IDs, and range before interpreting historical AP differences. The committed configs were verified by `CH/checks/phase4_checks.py`.

### P4-02 — Every run in a phase overwrites the same prediction object

**Priority:** high for traceability and subsequent comparisons. **Confidence:** confirmed static path analysis; no S3 write attempted.

Both `my_work/phase4a/scripts/score_sequential.py:125-128` and `my_work/phase4b/scripts/score_sequential.py:125-128` write `<eval.predictions_base>/sequential_preds.parquet`. Within a phase all sweep runs inherit the same `predictions_base`. In both `scripts/run_ablations.py:134-139`, changing `eval.checkpoint` and `output.checkpoint_dir` does not change the prediction destination.

Consequently, the eighth Phase 4a finder score replaces the first seven prediction sets. The flat long-history sweep also reuses the flat destination. Running the documented 100-file score again replaces that same object. Checkpoints and scalar result rows survive, but earlier run-level predictions are unavailable at their expected path unless S3 object versioning or copies exist. This undermines reproducible paired comparisons, calibration, ensembles, and audit trails.

**Action:** use a unique, immutable prediction prefix for each run and include that URI in `results.csv`. Record checkpoint hash, resolved config, input file manifest, preprocessing version, row count, positives, and transaction-ID fingerprint alongside predictions. Fail rather than silently replacing a different run's output. This finding concerns the present runner, not a hypothetical new feature.

### P4-03 — The “causal” mask has a global-CLS bypass across layers

**Priority:** medium; invalidates strict causal-ablation semantics and future per-transaction prediction use. **Confidence:** reproduced on CPU.

At `my_work/phase4a/sequential.py:60-68` and `my_work/phase4b/sequential_hier.py:88-95`, `allowed = (tj < 0) | (ti < 0) | (tj <= ti)` lets CLS read every transaction and lets every transaction read CLS. After one layer, CLS contains later-transaction information. After two layers, an earlier transaction can read that information through CLS. The flat implementation inherits this behavior from `fraud/sequential_model.py`; it was not newly introduced by my tokenizer changes. The hierarchical implementation repeats it.

The synthetic reproduction changes only the final transaction's numeric fields and inspects an earlier transaction's hidden state. The maximum absolute changes were:

| Model | One layer | Two layers |
|---|---:|---:|
| Flat | 0.0 | 0.0351890 |
| Hierarchical sequence stage | 0.0 | 0.1675981 |

The numbers are seeded diagnostic differences, not performance measurements. Existing `test_sequential_causal_mask` only checks output shape and uses one layer, so it cannot detect this.

**Scope:** this is **not** evidence that current target-level fraud scores use labels or transactions after the scored target. For the existing single-target classifier, every transaction in the input should already be observable. Bidirectional attention remains a legitimate default. The problem is the advertised causal dependency restriction and the interpretation of “causal versus bidirectional” as a clean ablation.

**Action:** prevent ordinary transaction queries from reading global CLS while allowing CLS to aggregate all observed transactions, or adopt an end-of-sequence readout with a suitable causal mask. Handle left-padded query rows without creating all-masked softmax rows. Add a two-or-more-layer dependency test. A future predict-every-transaction model also needs time features that do not depend on a later target.

### P4-04 — Enabling EMA validates one model but saves another

**Priority:** medium; must fix before an EMA experiment. **Confidence:** reproduced with Lightning 2.6.6.

The duplicated callbacks at `my_work/phase4a/callbacks.py:29-42` and `my_work/phase4b/callbacks.py:29-42` swap averaged weights in for validation, then restore raw weights in `on_validation_end`. The training scripts register EMA before ModelCheckpoint at `scripts/train_sequential.py:112-122`. The callback documentation at line 12 claims this order makes the checkpoint capture EMA weights, but the restoration happens before checkpoint saving.

A one-step scalar Lightning training run reproduced validation at weight **0.5** while `best.ckpt` contained weight **1.0**. Thus a recorded validation AP can describe weights different from those later loaded by scoring. Callback state also has no checkpoint serialization for its shadow weights.

An additional accumulation detail: EMA updates in `on_train_batch_end`, even on microbatches with no optimizer step. At accumulation=16 the raw weights are repeatedly folded into the average sixteen times per optimizer interval, changing the intended decay behavior compared with accumulation=1.

**Scope:** all committed Phase 4 configs set `ema: false`. This finding does not retroactively invalidate the recorded AdamW/RAdam sweep on its own.

**Action:** explicitly serialize an EMA model/state or keep the averaged model active through checkpoint capture, restore raw training parameters at a correctly ordered hook, and update EMA only after optimizer steps. Verify the checkpoint's reloaded validation scores match the score used for selection. The reproduction is in the audit check script; the existing callback test only checks swap/restore arithmetic, not checkpoint identity.

### P4-05 — Training and scoring resolve overlapping ablation overrides in opposite orders

**Priority:** medium, latent in the runner. **Confidence:** reproduced OmegaConf behavior.

Both copies of `scripts/run_ablations.py` use `common + overrides` for training at line 114 but `*overrides, *common` for scoring at line 136. With common `sequence.attention_mask=bidirectional` and an axis override `sequence.attention_mask=causal`, training uses causal while scoring uses bidirectional. Mask choice does not affect parameter shapes, so the checkpoint loads without detecting the mismatch. Other overlaps can fail with shape errors or silently change time features, evaluation stride, or data settings.

**Scope:** the current three canonical specs do not overlap common/axis keys in this way; no evidence that this defect affected the pasted runs.

**Action:** resolve one full run config once and reuse it for train and score, with only explicit evaluation destination/checkpoint additions. Save the resolved YAML. Score from checkpoint architecture hyperparameters and validate the preprocessing/schema contract instead of rebuilding solely from a mutable external YAML (`scripts/score_sequential.py:89-93`).

### P4-06 — The reported AUROC maximum need not belong to the AP-selected checkpoint

**Priority:** low for code correctness, medium for reporting interpretation. **Confidence:** confirmed.

`scripts/run_ablations.py:125-127` separately takes maximum `valid_ap` and maximum `valid_auroc` across epochs. Early stopping/checkpointing select `valid_ap`. The current-standing document itself provides examples: h10 best AP is epoch 6 with AUROC 0.946059, but the summary AUROC 0.948879 comes from epoch 9 (`my_work/docs/my_current_standing.md:49-60`). For h16, AP selects epoch 8 whereas the summary maximum AUROC comes from epoch 9.

The column names say “best,” so the numbers are not intrinsically false; they should not be read as one model's pair of scores.

**Action:** include selected epoch, AP and AUROC at that epoch, checkpoint path, and optionally a separately labeled maximum-AUROC column. Keep unrounded metric values in machine-readable outputs; scoring currently logs/parses only four decimal places (`scripts/score_sequential.py:136`, `scripts/run_ablations.py:33,142-145`).

## Scientific conclusions that are not established yet

- **The direction of the work is aligned with Ya’s plan.** History, transaction positions, frozen preprocessing, target-only labels, left padding, early stopping on full-stream AP, and configurable masks are present. The hierarchical model implements his deferred Architecture B. This is a useful and coherent extension, not an obviously wrong model family.
- **“Embedding is the only lever that matters,” “mask/optimizer are within noise,” “statistically tied,” and “Architecture A is saturated” are stronger than the evidence.** The posted grid uses one seed, one small training slice, repeatedly reused validation data, and no uncertainty estimates. Marginal means describe this grid; they do not isolate general significance or prove a ceiling. Repeat candidate configurations over several seeds and compute paired uncertainty on a locked holdout, resampling accounts rather than treating overlapping windows as independent. Compare the same frozen cases and record enough positives for useful confidence intervals.
- **The plan's decisive h0 and time-delta ablations are absent from the current Phase 4 grids.** `docs/phase4_sequential_plan.md:98-101` calls for history 0 versus 5 and time-delta on/off. Those establish whether additional history contributes beyond model/training changes and whether timing explains the gain. Existing single-transaction historical results are informative but less controlled than a matched current-pipeline h0 baseline. Audit artifacts may exist only on the pod; none are in this checkout.
- **The periodic option is a particular implementation, not a verdict on numerical embeddings generally.** At `numerical_embeddings.py:19-25`, frequencies are per feature but the projection is shared. For zero-valued inputs all continuous features produce the same token, with no per-field periodic bias. This differs from an explicitly feature-aware tokenizer and may matter for many imputed/zero features. Review the cited paper's exact embedding variant, scaling and frequency initialization before interpreting its result as “periodic embeddings do not help.” This is a design limitation/research hypothesis, not a reproduced training failure.
- **Accumulation is a sensible memory solution but not guaranteed exact run equivalence.** The “mathematically equivalent” claim in `my_work/docs/my_phase4a_guide.md:21` needs qualifications: matching example groups, correctly weighted partial batches, optimizer step count, dropout/RNG and numerical differences. With `pos_weight != 1`, PyTorch weighted cross-entropy normalizes separately within each microbatch, so a simple mean of microbatch losses differs from a full-batch weighted mean if class composition varies. Published runs have `pos_weight=1` and EMA off, so the sharpest variants of these issues are not enabled.
- **Architecture comparisons change both structure and capacity.** A has four flat blocks; B has two per-transaction blocks plus four sequence blocks. This is a legitimate practical comparison, but a mechanism claim needs parameter count, throughput, peak memory and roughly matched compute/parameter controls. Report predictive and operational tradeoffs together.
- **The current B guide predicts memory viability more strongly than the evidence.** Its two-stage design reduces attention cost substantially; it does not guarantee any particular batch/history fits a specific GPU. The current-standing document has no completed B metrics. Profile first, then state observed memory rather than promise h16/4096 success.

## Small implementation and documentation issues

- `my_work/phase4b/factory.py:14` eagerly evaluates `cfg.model.num_attn_blocks` as the default argument even when `num_seq_blocks` exists. Removing the redundant flat-model key from a valid hierarchical YAML causes a missing-key error. Current YAML includes both keys, so this does not break the supplied run.
- `model.type` is logged but does not route architectures. Phase 4a's factory always builds flat, Phase 4b's always builds hierarchical. The comment in the flat base YAML implying the type can switch to `FTTransformerHier` is misleading. Use the correct phase's script/config pair.
- Invalid `continuous_embedding` names silently take the linear branch; any `attention_mask` other than exactly `causal` silently takes bidirectional. Validate enums to avoid misleading experiment labels.
- The Optuna configuration searches history 10/20 at the base 4096 batch despite known flat-model OOM at long histories. Failed processes are pruned and their captured stdout/stderr is discarded (`my_work/phase4a/scripts/tune.py:99-102`), hiding whether failures were memory, config, or data errors. Use a memory-aware batch/accumulation policy and keep trial logs. The tuner is not evidence that these larger histories were tested successfully.
- `run_ablations._run` says “tee” but buffers the entire subprocess until completion (`scripts/run_ablations.py:68-75`). Its run log is unavailable for live inspection during a long training run and subprocess output is retained in RAM. Streaming to a run-specific log would make monitoring reliable.
- `my_work/docs/my_roadmap.md` references `my_work/run_remaining.ps1`; that file is absent. Phase 4 docs also retain “Phase 6” labels and a nonexistent `fraud/scripts/tune.py` invocation. These look like stale relocation/transcription documentation, not proof that the main algorithms failed.

## Verification and coverage

Read in full: every Python source in `my_work/phase4a/` and `my_work/phase4b/`, all seven YAML configs, `docs/phase4_design_decisions.md`, `docs/phase4_sequential_plan.md`, the Phase 4a/4b guides, roadmap and current-standing document. Compared the Phase 4a model with `fraud/sequential_model.py`; compared duplicated tokenizer, embeddings, optimizer, callbacks and runners across both phases. Read the shared callback implementation and relevant sequence-builder batch logic. Read existing model tests, including all their actual assertions.

Executed an additional synthetic CPU check using Python 3.12, PyTorch 2.14.1+cpu and Lightning 2.6.6:

CH’s original private audit retains the exact synthetic-check commands and outputs. The public copy preserves the findings; it does not include the executable reproduction scripts.

Artifacts:

- `CH/checks/phase4_checks.py`: standalone reproduction and positive padding checks.
- `CH/checks/phase4_checks.stdout.json`: numeric observations and resolved eval ranges.
- `CH/checks/phase4_checks.stderr.txt`: runtime warnings/trainer messages.
- `CH/checks/pytest.txt`: the root audit's full existing-suite run; root reported 28 passing tests.

The four padding tests change both continuous values and categorical IDs exclusively in padded slots. Target-logit deltas are exactly zero for flat/hierarchical and causal/bidirectional in eval mode. This supports the intended padding behavior for ordinary finite inputs.

The existing model tests exercise dimensions, one backward pass, a one-layer causal forward, periodic forward, the hierarchical forward/embed, built-in optimizer construction, and basic EMA arithmetic. They do not establish full training quality, strict multilayer causality, checkpoint/metric identity, train-score configuration identity, S3 artifact preservation, a matched benchmark cohort, H200 mixed-precision behavior, optional external optimizers, or statistically significant lift. The new reproductions close only the explicitly stated gaps.

No bank datasets, trained checkpoints, raw ablation CSVs or pod logs were available here. The absence of these items could reflect an incomplete screenshot-transferred checkout; it is not evidence that the original work never ran. Exact reported numbers and on-pod status therefore remain externally reported rather than independently reproduced.
