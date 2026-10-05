# TF: my review record for Ya’s design and the shared implementation

> Public first-person adaptation of documents originally prepared by CH on 2026-10-05. I retain the findings and their original review scope; CH performed the reported independent checks. Source paths are illustrative references to my private project, with `my_work/` representing my work and Ya identifying my collaborator. Source code, private evidence files and bank data are not included in this document set. Later changes require a separate review.

Audit date: 2026-10-05. Read-only review of application code; only `CH/` audit artifacts were added. No production training or bank S3 requests were performed. This report describes the checked-out reconstructed repository, not an independently recovered copy of the bank workstation.

## What the project actually does

I use **Ya’s** as the public ownership label for the original configuration and documentation, and `my_work/` for my separate contributions. These aliases reflect my project context, not an independent inference about authorship.

The task is binary ranking of transactions: higher `score` should mean more likely fraud (`frd_tag=1`). The selected version-1 data is **23 base features**: six numeric fields, 13 raw categorical fields, and four calendar features derived from transaction time. Six missingness indicators usually increase the token count to 29. Account and transaction identifiers locate/group records; they are not intended predictor tokens. Historical fraud labels are intentionally excluded.

There are three actual supervised model paths:

1. **In-memory baseline:** `train_baseline.py` reads selected files into DataFrames; fits the numeric imputer on training data; lets the vendored PyTorch Tabular package fit category encoders/scalers; trains one-transaction FT-Transformer; evaluates AP/AUROC over complete splits. `score_eval.py` reuses the saved imputer and model datamodule.
2. **Streaming one-transaction baseline:** a separate artifact-builder fits fixed statistics/vocabularies; `ParquetStreamingDataset` encodes file batches with these artifacts; `build_streaming_model` manually supplies feature dimensions to the vendored FT-Transformer; Lightning callbacks compute full-validation AP/AUROC. `score_streaming.py` scores batches and writes per-file predictions.
3. **Original flat sequential model:** `SequenceBuilder` supplies previous transactions and the target, missing-history mask, and optionally a time-to-target feature. `fraud/sequential_model.py` tokenizes each numeric/categorical feature, adds a learned transaction position, prepends a summary/CLS token, and runs a PyTorch TransformerEncoder. Its final summary vector produces two logits. This is an FT-style extension; it is not implemented in the placeholder `src/pytorch_tabular/models/ft_transformer_seq/` directory.

Current detailed phase numbering is **1: predictors/preprocessing; 2: baseline smoke; 3: streaming scale-up; 4: history-aware modeling**. `README.md` and `fraud_detection_plan_overall.md` retain earlier numbering and an obsolete “coming soon” placeholder. This explains some apparent contradictions; a placeholder is not proof that sequential code was never written.

The initial public benchmark is **UCI Bank Marketing**, whose target is term-deposit subscription, not fraud. It motivated selection of FT-Transformer but is not direct fraud evidence or a controlled reproduction of a research paper. The relevant source citation normally associated with FT-Transformer is *Revisiting Deep Learning Models for Tabular Data*; this repository primarily adopts a PyTorch Tabular implementation and extends it.

## Confirmed issues in the checked-out code

### O1 — Streaming model creation fails because the imported config API does not match the reconstructed builder (high priority)

- `fraud/streaming_model.py:30-36` constructs `InferredConfig(output_cardinality=[2], ...)`.
- `src/pytorch_tabular/config/config.py:206-230` has no `output_cardinality` field.
- Direct synthetic call `build_inferred_config([3], 1, 1)` raises `TypeError: InferredConfig.__init__() got an unexpected keyword argument 'output_cardinality'`.
- Both original streaming training and scoring call this builder, so installing dependencies alone does not make those paths run in this checkout.
- Proposed correction: reconcile the expected fork API with the pinned source. With current v1.1.0, `output_dim=2` is the binary classification contract; remove the unsupported argument only after checking checkpoint/config compatibility and run a synthetic train/save/load/score integration test.
- **Attribution limitation:** `git log` records replacement/pinning of `src/pytorch_tabular` to upstream v1.1.0 in the private source history. The streaming builder's reconstruction record says its 102 source lines were visually recovered. This is evidence of a current repository integration mismatch, not evidence that Ya’s reported bank runs failed or that I caused it.

### O2 — Baseline best-checkpoint reload fails on allowed modern PyTorch versions (high priority)

- `src/pytorch_tabular/utils/python_utils.py:74-84` calls `torch.load` without specifying `weights_only`; modern PyTorch defaults to weights-only loading.
- `src/pytorch_tabular/tabular_model.py:1494` uses this helper after fitting. Lightning checkpoints contain OmegaConf configuration objects, which are not accepted by the default weights-only unpickler.
- In the audit environment (Torch 2.14.1 CPU, Lightning 2.6.6), a 32-row, one-epoch synthetic FT-Transformer fit completed optimization and then raised `UnpicklingError` for unsupported `omegaconf.dictconfig.DictConfig` while loading its best checkpoint.
- A minimal synthetic checkpoint reproduction is in `CH/checks/original_integration_repros.py`.
- Proposed correction: use an explicit serialization contract. For the project's trusted, locally produced legacy checkpoints, explicitly load that format with the intended non-weights-only mode; prefer a tensors-plus-plain-metadata format for a new stable contract. Pin and record package versions. Do not blanket-change loading of arbitrary untrusted checkpoints.
- The custom original streaming/sequential scorers already explicitly specify `weights_only=False`; the failure is in the vendored baseline path. Again, the current imported upstream source need not match the bank fork.

### O3 — Saving the baseline removes its training metrics CSV (medium priority)

- `fraud/scripts/train_baseline.py:208-211` puts `training_metrics.csv` in `output.checkpoint_dir`.
- Line 233 calls `tabular_model.save_model` with that same directory.
- `src/pytorch_tabular/tabular_model.py:1532-1535` deletes every existing file in the destination before saving.
- The actual method, executed on a temporary directory with serialization stopped immediately after cleanup, deleted the synthetic training CSV. The documented post-training progress file therefore disappears on a successful save.
- Proposed correction: separate model serialization from run metrics, or change model saving to replace only its known model artifacts. Preserve a run manifest and full training history. The existing deletion also fails if the directory contains subdirectories.

### O4 — The original “causal” attention mask has a CLS route around its transaction restriction (medium priority; does not establish future-data leakage in current last-target classification)

- `fraud/sequential_model.py:104-106` allows every query to attend CLS and CLS to attend every transaction.
- After block 1, CLS contains later-transaction information. At block 2, earlier transactions can obtain it by attending CLS.
- Synthetic check changing only the latest transaction's token vectors changed the earliest transaction representation by **0.0248523** after two blocks, versus exactly **0** after one block.
- Current prediction is for the last target in an already-past-only window, so this alone does **not** leak transactions later than that target. It does invalidate the promised block-causal property and makes the causal/bidirectional ablation imprecise; it is unsuitable for the proposed future prediction-of-every-transaction extension.
- Proposed correction: prevent transaction tokens from reading a globally aggregating CLS token, or structure the summary token at the terminal transaction with a causally correct layout. Test earlier hidden representations for invariance when later inputs are changed.

### O5 — Non-default positive class weighting breaks the original scoring checkpoint load (medium priority)

- Original sequential training accepts `training.pos_weight`; `fraud/sequential_model.py:89-92` registers `loss_fn.weight` for non-default weighting.
- `fraud/scripts/score_sequential.py:89-101` reconstructs the model with default weight 1 and then strictly loads all state.
- Loading a `pos_weight=5` model state into the scorer's default constructor raises an unexpected `loss_fn.weight` key error. This is directly reproduced in `original_project_repros.py`.
- The same construction problem exists statically in streaming scoring: `score_streaming.py:70-79` omits `pos_weight` passed by training; the base class registers custom loss modules and weight buffers (`base_model.py:99,199-207`). The streaming API blocker O1 prevents an unmodified end-to-end confirmation of this second path.
- Proposed correction: reconstruct architecture/loss from checkpoint metadata, or use an intentionally defined inference-only state contract. Preserve strict checking of actual learned weights; do not solve this with unconditional `strict=False`.
- Default checked-in weight 1 is unaffected.

### O6 — Default FT attention-based feature importance reads the wrong attention direction (interpretability issue)

- Attention arrays are `[sample, head, query, key]` (`common/layers/transformers.py:61`).
- `ft_transformer.py:91` uses `attn_weights[:, :, :, -1]`, selecting **each feature query's attention to CLS**, instead of the final CLS query's attention to features.
- With synthetic attention, current output is `[0.1, 0.3]`; the CLS-to-feature values are `[0.55, 0.4]`.
- Model logits are unaffected, but interpreting this value as which features the summary token used is wrong. Even corrected attention weights are not causal explanations of model decisions.
- Proposed correction: correct the axis and label the explanation's limitations; verify independent predictive sensitivity/permutation or attribution analyses before using it for bank decisions.

### O7 — Scoring/checkpoint configuration does not bind the feature vocabulary and artifact identity (reproducibility gap)

- Original scorers load features/artifacts from current YAML then load only checkpoint tensor state: `score_streaming.py:53-79`, `score_sequential.py:49-53,89-101`.
- Sequential hyperparameters save dimensions, not feature names/category meanings or artifact checksums. Streaming config saves column names/dimensions but does not validate the scoring artifact/version against the checkpoint.
- An equal-cardinality category mapping or same-sized reordered feature list can load successfully while assigning learned weights to the wrong meanings. Merely matching tensor shapes cannot detect this.
- Proposed correction: save a resolved run manifest with feature order, category mappings/artifact hashes, target convention, datetime settings, history/mask/time-delta options, git revision, library versions, and actual train/validation/evaluation file lists. Scoring should fail on incompatible identity unless an explicit migration is performed.
- This is a confirmed missing guard, not a claim that existing bank artifacts actually differ.

### O8 — Original sequential scoring is not memory-bounded across the evaluation range (scalability gap)

- `fraud/scripts/score_sequential.py:105-130` accumulates all scores, labels, and every ID/amount for the full range, concatenates them, and constructs one Arrow table.
- This accumulation occurs even with `eval.compute_metrics=false`; widening from one file to hundreds of millions of records grows host RAM with the total evaluation size. CPU copies during concatenation/table construction increase peak demand.
- `StreamingBinaryMetrics` also intentionally retains all validation scores/targets (`fraud/callbacks.py:67-74`); it streams features but is not constant-memory full-range metric computation.
- Proposed correction: write incremental predictions with a Parquet writer or per-file output; move exact full-range AP/AUROC calculation to a deliberately provisioned offline job, or use documented approximations when appropriate. Preserve IDs and one-to-one coverage checks. This is a scale limit, not a demonstrated OOM on my documented 100-file jobs.

## Architecture/documentation mismatches and lesser issues

- **No SEP tokens in original sequential model:** despite its name/docstrings and design plan, `sequential_model.py:114-123` directly concatenates per-transaction feature blocks and CLS. Boundaries remain recoverable through feature identity and transaction-position embeddings, so absence is not automatically a modeling defect; document it accurately or test SEP as a deliberate architecture change.
- **Different transformer blocks across paths:** the original sequential model uses standard PyTorch pre-norm GELU TransformerEncoder (`sequential_model.py:75-86`); vendored FT uses post-norm AddNorm and default GEGLU (`common/layers/transformers.py:30-31,108-128`). Attention head dimensions also differ: vendored blocks default to *embed dimension per head* (`:103`), whereas standard MHA divides embedding width across heads. A gain over the one-transaction model changes more than history alone unless a matched history-0 control is used.
- `FTTransformerConfig.transformer_head_dim` is advertised but not forwarded by `FTTransformerBackbone._build_network` (`ft_transformer.py:37-46`), so changing that config field does not change the active FT attention head size.
- `share_embedding=False` is advertised but the FT builder always passes `share_embedding_strategy` (`ft_transformer.py:60`); default `fraction` enables SharedEmbeddings. This affects exact replication/parameter assumptions; the training and scoring default still agree.
- Ordinary documented `transformer_activation="ReLU"` raises `KeyError` because `TransformerEncoderBlock` catches `AttributeError` instead of missing-dictionary `KeyError` (`transformers.py:108-119`); the fallback also accesses undefined `self.hparams`. Default GEGLU works. A direct construction reproduces the KeyError.
- `eval_stride` is configurable even though the plan says it must equal 1. Builders deliberately retain every positive while subsampling negatives for larger stride; an evaluation override greater than 1 therefore changes prevalence and can make AP incomparable. Enforce full coverage in evaluation entrypoints. The data-pipeline audit covers the sampling details.
- Multi-GPU support is not established: callbacks collect rank-local metrics with no all-rank aggregation; the existing file datasets shard DataLoader workers, not necessarily distributed ranks. Configure one device explicitly until a complete distributed contract is implemented and tested.
- `show_progress.py`'s idle timeout does not apply while initially waiting for a nonexistent file (`:167-169`), and row-count-based follow mode misses edits/restarts that do not increase row count. This is a convenience limitation, not a fraud-model issue.
- Root audit covers the invalid editable-install backend, undeclared eager import dependencies, and exclusion of `my_work` from packaging.

## What is already sensible

- Numeric imputation fitted on training data and reuse of saved transformations is the right direction.
- Explicit fixed vocabulary/UNK/MISSING conventions in the streaming path are preferable to fitting category encoders on evaluation files.
- Excluding historical fraud labels avoids an obvious operational availability problem.
- The original per-feature tokenization expresses feature identity without requiring arbitrary within-transaction positional ordering.
- Bidirectional attention over only legitimately available past/target observations is a defensible choice for predicting the final target.
- Full-split AP/AUROC is appropriate for very rare fraud. Averaging batch AP would be wrong. Approximately 99.85% accuracy can be obtained by calling every transaction legitimate, so accuracy is not the success criterion.
- Original sequential masking passed a synthetic padding-invariance check: changing masked history values changed logits by exactly 0.
- Checkpoint selection by validation AP, evaluation on separate files, and a history-length-zero ablation are conceptually sound if the actual split/feature-availability contracts hold.

## How much of the claimed progress is established

Original docs report a baseline test AP **0.0940**, streaming test AP **0.0940** over a broader range, and original history-5 best validation AP **0.1293**. These are documentation claims, not recreated results from this environment. They use different train/validation/evaluation sizes in several comparisons; rounded proximity does not establish numerical preprocessing parity or a causal architecture advantage.

Calling the original single-transaction model “confirmed feature-limited” or declaring FT the production winner is too strong for the shown evidence. The UCI comparisons use different test samples and training settings; there are no reported uncertainty intervals or matched tuning budgets. More epochs plateauing can reflect optimization, data quality, label delay, drift, sampling, or model family limitations. A matched comparison with the existing internal XGBoost model and a current external-score-only control would be particularly useful because `external_1_scr` is already an upstream fraud score.

Before concluding history helps, use identical held-out targets, the same feature/artifact version, matched training exposure/model family, several seeds, and a paired confidence interval clustered by account. Report AP with prevalence, recall/precision at the bank's alert budget, false-positive rate, amount-weighted fraud caught, calibration, and operational latency. Temporal holdout and point-in-time feature availability matter more than a small validation AP difference.

There is no evidence here sufficient to certify full production readiness or to say all tests reflect the bank's actual data. The repository has enough implementation to support Phases 1–4 conceptually; its current recovered baseline/streaming integration has concrete blockers that existing unit tests miss.

## Verification and coverage

CH executed these checks in its local audit environment, using CPU and synthetic data only:

- `PYTHONPATH=src:. ... CH/checks/original_project_repros.py`: reproduced O3, O4, O5 (sequential), O6 and passed padding invariance.
- `PYTHONPATH=src:. ... CH/checks/original_integration_repros.py`: reproduced O1, O2's checkpoint-loading contract, and the ordinary-activation error.
- A 32-row, one-epoch in-memory FT baseline experiment completed optimization and then reproduced O2 at best-checkpoint reload. No claim of trained fraud-model quality follows from synthetic data.

Read in full: README; all seven documents in `docs/`; all `fraud/` model/callback/logging files; all original Python training, artifact-building, scoring, progress scripts and shell helpers; all original YAML/CSV configs; FT config/backbone/model; transformer and embedding layers; relevant BaseModel loss/metrics/optimizer/training code. Reviewed relevant datamodule preprocessing/dimension inference/DataLoader code, TabularModel fit/save/load/checkpoint orchestration, common head/layer wiring and utility imports/loading code. Read reconstruction status/handoff/provenance records and relevant git history to distinguish source recovery from runtime validation.

The generic vendored model families (AutoInt, DANet, GANDALF, GATE, MDN, NODE, stacking, TabNet, TabTransformer, SSL DAE) were inventoried and their eager-import integration considered; their unrelated algorithms were **not exhaustively behaviorally audited**. Separate agents cover `fraud/data/`, original notebooks/tests, all `my_work/` phases/tests/scoring, and ablations. No original screenshot pixels were re-audited in this task. `DISPLAY_RECONSTRUCTED` docs/notebooks and historically missing/upstream-filled core files remain fidelity limitations independent of these code findings.
