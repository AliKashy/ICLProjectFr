# TF: my Phase 5 audit record — SSL and supervised contrastive pretraining

> Public first-person adaptation of documents originally prepared by CH on 2026-10-05. I retain the findings and their original review scope; CH performed the reported independent checks. Source paths are illustrative references to my private project, with `my_work/` representing my work and Ya identifying my collaborator. Source code, private evidence files and bank data are not included in this document set. Later changes require a separate review.

Audit date: 2026-10-05. Source reviewed without modification. All 17 files in `my_work/phase5_pretrain/`, the Phase 5 guide, roadmap/current-standing context, relevant shared data/metric interfaces, and relevant existing model/SupCon tests were read. No bank data, S3 credentials, trained bank checkpoints, or production runs were accessed. Findings concern this repository snapshot, not independently verified performance of code running on the bank pod.

## What this code actually does

- `encoder.py` and `tokenizer.py` turn one transaction's numeric and categorical values into feature vectors, add a CLS token, and run a transformer. The CLS output is a summary of one transaction; there is no transaction history in this encoder.
- SSL hides feature tokens, predicts numeric values with squared error and categorical values with cross-entropy, then discards the reconstruction heads.
- SupCon makes two corrupted copies of each labeled transaction, runs the encoder and a projection head, and is intended to pull together views with the same fraud label while separating the other class. It uses fraud labels, so it is supervised pretraining, not unlabeled SSL.
- Both recipes transfer only `encoder.*` weights into the same `SSLFinetuneClassifier`, which trains a two-class linear head and normally the whole encoder. Validation AP selects the classifier checkpoint. The pretraining checkpoints themselves are selected on reconstruction/contrastive validation loss.
- The two scoring scripts apply frozen preprocessing, use `eval()` and `no_grad()`, compute fraud probability with `softmax(...)[..., 1]`, and save transaction/account IDs, binary labels and scores. Embedding export uses the SSL encoder's CLS vectors.
- These are currently single-transaction classifiers. Neither Phase 4 training script loads these encoder checkpoints. The work is a reasonable basis for a pretraining experiment, but it does not yet initialize the history model chosen in Phase 4.

## Priority findings

### P5-01 — High: SupCon attaches the wrong labels to the flattened views

**Evidence:** `my_work/phase5_pretrain/supcon.py:37-42`; the views are created in `[B, 2, d]` layout at `supcon.py:100-103`.

`reshape(B * V, -1)` produces sample-major rows, e.g. `[transaction0/view0, transaction0/view1, transaction1/view0, transaction1/view1]`. Repeating the original `[B,B]` class mask with `.repeat(V,V)` assumes view-major rows instead. With transaction labels `[0,1]`, actual flattened labels are `[0,0,1,1]`, while the mask treats them as `[0,1,0,1]`. True partners can be treated as negatives and fraud/non-fraud pairs as positives. Finite loss and successful backpropagation do not detect this.

**Reproduced:** On two synthetic transactions, with both views correctly identical within each class and classes orthogonal, current loss is **14.2857151**. Deliberately mixing the views across the two labels makes current loss **0.000001192**. A reference objective with correctly expanded labels reverses these results: **0.000000954** and **14.2857151**, respectively. Gradients remain finite despite the wrong objective.

**Correction:** Keep sample-major flattening and expand labels with `labels.reshape(-1).repeat_interleave(V)` before forming the equality mask, or explicitly flatten features view-major to match the existing mask. Test known pair membership and compare against an independent SupCon reference; additionally verify invariance under jointly permuting transactions and labels. Rerun SupCon pretraining after correction; a balanced sampler alone cannot fix this objective.

**Origin:** This is not evidence of an OCR/photo transcription mistake. `reconstruction/runs/20261004T054800Z-phase5-supcon.json` explicitly records preservation of the photographed `repeat`/`reshape` source. The independent audit record and this file have matching SHA-256 content identities, retained in the private provenance record. This review relies on those provenance records; it did not reopen the photos.

### P5-02 — High: masking erases numeric field identity, forcing identical predictions for masked fields

**Evidence:** `ssl_pretrain.py:50-53` defines one shared mask token and one shared numeric head; `ssl_pretrain.py:65-78` replaces complete feature tokens and predicts masked numeric values. `encoder.py:43-64` contains no positional/feature-identity addition after replacement.

The normal tokenizer has field-specific information in its weights/biases, but masking replaces that information with the same vector for every masked field. A transformer without positional/column identities is permutation equivariant: two identical masked inputs with the same context get identical outputs when dropout is disabled. The shared numeric head therefore cannot give different reconstructions for two simultaneously masked numeric fields in a transaction. Dropout adds noise during training, not knowledge of which numeric field is being reconstructed. Separate categorical heads distinguish categorical outputs; that does not repair the shared numeric head.

**Reproduced:** Masking the first two fields of one row with targets `+2` and `-2`, while leaving a third field visible, produces predictions `[0.5007387996, 0.5007387996]`, exact difference **0.0**. The model computes a loss and backpropagates, but cannot represent the desired two different predictions in this situation.

**Correction:** Retain an independent learned feature/column identity when replacing the feature value, e.g. `column_embedding + mask_embedding`, with that identity also used consistently for visible tokens. A separate decoder head per numeric field is another design option, but should be evaluated deliberately. Test that masked fields can make different predictions from identical masking contexts on a controlled learning task.

**Origin:** The SSL source SHA-256 matches the `VISIBLE_SOURCE_VERIFIED` record in `reconstruction/runs/20261004-phase5-ssl-models.json`. The defect is a model-design issue in the available source, not a demonstrated photo-transfer error.

### P5-03 — Medium: scoring a class-weighted fine-tuned model fails to load

**Evidence:** `ssl_finetune.py:50-53` registers `loss_fn.weight` when `pos_weight != 1`. Both trainers pass `finetune.pos_weight` (`scripts/train_ssl_finetune.py:87`, `scripts/train_contrastive.py:88`). Both scorers reconstruct the classifier without that argument and then load strictly (`scripts/score_ssl_finetune.py:60-69`, `scripts/score_contrastive.py:61-70`).

**Reproduced:** A classifier trained/constructed with `pos_weight=10` has a `loss_fn.weight` checkpoint buffer. Loading its state into the default scorer model raises `Unexpected key(s) in state_dict: "loss_fn.weight"`. The supplied default `pos_weight=1` is unaffected; this is a real failure of the advertised weighting knob.

**Correction:** Reconstruct with checkpoint hyperparameters, or at least forward `pos_weight` consistently. Do not hide arbitrary model-weight mismatches with an unrestricted `strict=False`. Exercise a complete train/checkpoint/score round trip with nondefault weighting.

### P5-04 — Medium: supplied comparisons change validation population and model setup alongside pretraining

**Evidence:** `configs/ssl.yaml:23-27` uses training files 1–100 and validation only file 4950. `configs/contrastive.yaml:22-26` uses training files 1–100 but validation files 4950–4954. SSL fine-tuning reuses `cfg.data.val_*` at `scripts/train_ssl_finetune.py:75-78`. The Phase 5 guide claims head-to-head comparison of the recipes at `my_work/docs/my_phase5_guide.md:3`. Both default eval ranges contain only 8000 (`ssl.yaml:90-93`, `contrastive.yaml:84-87`), whereas current-standing comparisons use the 11-file range.

A validation AP on one file cannot be compared directly to AP on five files. Changing the validation set also changes which checkpoint early stopping selects. Separately, Phase 5 uses width 64 and no history; Phase 4 defaults use width 48 and transaction history. The roadmap calls pretraining the fifth lever on the selected model, but no transfer into either Phase 4 model is implemented. Phase 4b also defaults to two transaction-encoder blocks versus Phase 5's four, so a naive state-dict transplant is not equivalent.

**Correction:** First compare from-scratch vs corrected SSL vs corrected SupCon using the exact same classifier architecture, data ranges, preprocessing, validation/checkpoint rules, and evaluation rows. Give scratch its own explicit run/config and output directory. Then integrate a validated initialization into the selected history architecture, preferably a shared transaction encoder for Architecture B with aligned width, depth and feature schema; rerun scratch controls on that architecture. Do not claim a pure pretraining benefit from a Phase 4-vs-Phase 5 comparison that changes these other factors.

**Status:** Experimental-design/configuration gap. It does not establish that any reported bank result is numerically false; no Phase 5 bank results are available here.

### P5-05 — Medium: a missing or incomplete pretrained checkpoint can silently become a mislabeled scratch run

**Evidence:** `scripts/train_ssl_finetune.py:91-98` and `scripts/train_contrastive.py:92-99` warn but train from scratch when the requested checkpoint is absent. `ssl_finetune.py:55-59` filters whatever `encoder.*` keys exist and loads them with `strict=False`; the scripts log only the counts.

**Reproduced:** `load_pretrained({})` returns successfully with 18 missing keys on the tiny test model; all encoder parameters remain random. A partially populated checkpoint can similarly leave some encoder weights random without an exception, provided loaded tensor shapes match. A missing path also trains under the SSL/SupCon output name after only a warning.

**Correction:** Make scratch initialization an explicit option. For a pretrained run, require the checkpoint to exist and require every expected encoder key to load; allow only the intentionally new head to be missing. Record checkpoint hash, initialization recipe, ordered feature list, preprocessing artifact fingerprint, full resolved config and code revision with the result.

### P5-06 — Lower priority: zero masked features yield a non-differentiable loss

**Evidence:** `ssl_pretrain.py:66,72-86` initializes an independent zero and only adds a differentiable term if a mask contains at least one selected field.

**Reproduced:** `mask_ratio=0` returns zero with `requires_grad=False`; `backward()` raises `element 0 of tensors does not require grad and does not have a grad_fn`. A random mask with no selected positions has the same behavior. This is negligible for normal full batches with many fields and 30% masking, but is reachable with small synthetic batches or a zero masking override.

**Correction:** Validate the mask-ratio contract and enforce at least one masked field in a reconstruction batch, or intentionally return a graph-connected zero when skipping an empty-mask batch. Include the boundary behavior in the masking test.

## Methodological and operational recommendations

These are separated from confirmed high-impact defects.

1. **Sampling is an experiment, not a remedy for P5-01.** The guide correctly flags the rarity issue, but its wording that contrastive loss “needs multiple fraud examples per batch” (`my_phase5_guide.md:27`) is too absolute. After the mask is fixed, two augmented views supply a positive partner even for a singleton fraud transaction. At the measured training prevalence 0.1654% and batch 8192, an IID batch averages about 13.55 fraud rows; real block-local batches may be clustered. Balanced sampling can improve minority-class learning and compute efficiency, and should be compared on untouched natural-prevalence validation data. Changing the distribution need not improve every objective automatically.
2. **Profile SupCon memory before a 100-file run.** The default 8192 rows and two views produce a 16384×16384 pair matrix: 268,435,456 elements, 1 GiB for each float32 tensor. `supcon.py:40-47` materializes several such masks/logit/probability tensors, plus two encoder graphs. This is a concrete large-memory design, not a verified OOM on the bank GPU. Use a measured batch budget or a mathematically equivalent chunked objective. Gradient accumulation does not increase the number of simultaneous contrastive partners.
3. **Honor gradient accumulation consistently.** Only `scripts/pretrain_ssl.py:104-109` passes `accumulate_grad_batches`; `pretrain_contrastive.py:102-106`, `train_ssl_finetune.py:110-114` and `train_contrastive.py:111-115` ignore this setting. The SSL config exposes it at line 54. Default 1 is unaffected; a memory-driven override currently affects SSL pretraining but not its fine-tuner.
4. **Specify the frozen-probe behavior.** `ssl_finetune.py:62-64` disables encoder gradients, but does not put the encoder into eval mode. Training the classifier leaves encoder dropout active. The synthetic frozen probe produced a maximum logit difference of 0.2442 on repeat forwards of the same row. This may be intentional dropout regularization, but it is not a deterministic “linear probe on frozen embeddings” as normally understood. For that baseline, hold the encoder in eval mode during head training.
5. **Make unlabeled input an explicit supported mode if needed.** SSL does not use labels in its loss, but its shared dataset still requires the target column: `fraud/data/parquet_loader.py:112-115,149-150`. It currently works on the available labeled train files with labels ignored, not on arbitrary files with no label column. Export also carries labels. Add an optional-target dataset contract before using additional unlabeled transaction archives.
6. **Avoid run mixing in outputs.** Fixed default checkpoint/prediction directories are overwritten across reruns/overrides. Save a run identifier and manifest; use separate directories for model, seed, data range and recipe. Export is per-file and preserves row alignment, but accumulates a whole file in memory; empty parquet files cause `np.concatenate([])` to fail. The scorers also call AUROC without guarding a one-class eval range.
7. **Keep validation augmentation reproducible when comparing close pretraining losses.** SSL generates fresh masks during validation (`ssl_pretrain.py:93-95`); SupCon generates fresh corruption (`supcon.py:115-116`). This is a legitimate estimate of expected augmented loss, but fixed validation seeds/masks or repeated validation estimates give more stable checkpoint selection. All four Phase 5 loaders also use training `shuffle_rows` for validation. For the unaugmented classifier, row order alone does not alter global AP in ordinary single-device evaluation.
8. **Review loss weighting and realistic corruption.** SSL adds one global numeric MSE plus one mean CE per categorical field (`ssl_pretrain.py:74-85`), so the categorical count and cardinalities affect relative emphasis. Missingness flags may be easy reconstruction targets and correlated derived time features may make others easy. SupCon can create implausible feature combinations by copying fields from another transaction. These are choices to validate with downstream AP, not intrinsically invalid losses.
9. **Clear stale command/documentation names.** Several SSL script docstrings and `configs/ssl.yaml:4-5` point to nonexistent `fraud/scripts/*ssl*` locations. The Phase 5 guide has the correct `my_work/phase5_pretrain/scripts/*` commands. `export_embeddings.py:5` references nonexistent `train_gbm_on_embeddings.py`, while the roadmap drops the team's own GBM branch. The encoder docstring's “validated” wording exceeds what shape tests establish.
10. **Inherited loader/metric issues matter here too.** Phase 5 imports the shared `ParquetStreamingDataset` and `StreamingBinaryMetrics`; random worker partitioning, preprocessing artifacts and mixed-precision/global-metric concerns are reviewed in the data/runtime reports. Phase 5 defaults use four workers, so fixes to shared data correctness must precede expensive reruns.

## What passed / what is implemented reasonably

- The linear tokenizer has field-specific weights and biases; categorical features have distinct embedding tables. Both linear and periodic paths preserve the expected continuous-then-categorical layout.
- All three components use the same `TabularEncoder` parameter layout. With matching settings, SSL and SupCon `encoder.*` tensors transfer **exactly**, tensor for tensor. Only the new head's weight/bias are missing; the SupCon projection and SSL decoder heads are intentionally excluded. The new audit probe verified both transfers.
- Fine-tuning normally updates the encoder and head, uses binary integer labels and cross-entropy, and supplies logits/targets to the existing global-AP callback. The frozen setting excludes encoder parameters from the optimizer as well as using `no_grad()`.
- Pretraining scripts read training files separately from validation files. No label is used in the SSL reconstruction objective. SupCon explicitly consumes labels, as it should for a supervised recipe.
- Scoring/export use eval mode and no-grad; sigmoid vs two-logit-softmax is not a problem here. The scripts compute probability for index 1, consistent with the dataset's binary target encoding. They preserve row correspondence when attaching IDs and scores/embeddings.
- Every Phase 5 `torch.load` explicitly sets `weights_only=False`, so the newer PyTorch default does not break these loads. This assumes locally trusted training checkpoints, as in this project.
- Existing tests pass, but relevant Phase 5 assertions predominantly verify shapes, scalar losses and that backward runs (`my_work/tests/test_models.py:96-113`, `my_work/tests/test_supcon.py:31-52`). They do not validate SupCon pair semantics, field identity under masking, or complete transfer-key coverage. Thus passing tests do not contradict the findings.

## Coverage and evidence files

All Phase 5 files reviewed:

| Group | Files | Focus |
|---|---|---|
| Core | `__init__.py`, `encoder.py`, `tokenizer.py`, `numerical_embeddings.py` | Token layout, dimensions, field identity, pooled output, architecture |
| Objectives / classifier | `ssl_pretrain.py`, `supcon.py`, `ssl_finetune.py` | Masking, objective semantics, gradients, transfer, optimizers, train/eval |
| Pretraining scripts | `scripts/pretrain_ssl.py`, `scripts/pretrain_contrastive.py` | Data splits/loaders, config plumbing, seeds, selection/checkpoints |
| Fine-tuning scripts | `scripts/train_ssl_finetune.py`, `scripts/train_contrastive.py` | Initialization, state coverage, shared harness, output identity |
| Scoring / export | `scripts/score_ssl_finetune.py`, `scripts/score_contrastive.py`, `scripts/export_embeddings.py`, `scripts/__init__.py` | State reconstruction, eval semantics, row IDs, metrics and output |
| Configs | `configs/ssl.yaml`, `configs/contrastive.yaml` | All keys, ranges, defaults, paths, comparison consistency |

Audit checks added only under `CH/checks/`:

- `phase5_repro.py`: self-contained, synthetic CPU checks; leaves project source unchanged.
- `phase5_repro_results.json`: structured measurements and SHA-256 hashes of all Phase 5 Python files used.
- `phase5_repro_output.txt`: captured output.
- `pytest.txt` and `pytest.xml`: whole existing suite run by the audit coordinator; **28 passed**.

Private reproduction record:

CH’s original private audit retains the exact synthetic-check commands and outputs. The public copy preserves the findings; it does not include the executable reproduction scripts.

Runtime used: PyTorch 2.14.1+cpu, PyTorch Lightning 2.6.6. No GPU performance, full-data training quality, production checkpoint compatibility, or external-paper replication claim was established by these synthetic checks.
