# TF: my independent training-metric and architecture review record

> Public first-person adaptation of documents originally prepared by CH on 2026-10-05. I retain the findings and their original review scope; CH performed the reported independent checks. Source paths are illustrative references to my private project, with `my_work/` representing my work and Ya identifying my collaborator. Source code, private evidence files and bank data are not included in this document set. Later changes require a separate review.

Scope: shared metrics/callbacks, Phase 3 library blocks, Phase 4 flat/hierarchical models, and Phase 5 freeze/validation contracts. This is an additional narrow pass; it does not duplicate the separate data, Phase 4, Phase 5, and evaluation reviews. No runtime source changed. All experiments use synthetic tensors, CPU, and the audit virtual environment.

## Confirmed logging issue: CSV training loss is the final minibatch, not the epoch mean

- Location: `fraud/callbacks.py:23-40`; consumed by every streaming/sequential/pretraining runner.
- The callback writes during `on_validation_epoch_end`. Lightning has not completed reduction of training epoch metrics at this point. For the model logging pattern `on_step=True, on_epoch=True`, `train_loss` and `train_loss_step` still refer to the final minibatch.
- The first row fixes CSV fieldnames. `train_loss_epoch`, which becomes available later, is never added because extra fields are ignored.
- Reproduction with train minibatch losses 1 and 9: CSV records `train_loss=9`, while the actual training epoch mean exposed at `on_train_epoch_end` is 5. This repeats in the second epoch.
- Impact: the wildly variable training loss values copied into `my_work/docs/my_current_standing.md` do not describe epoch learning or overfitting. Validation AP, validation AUROC, and AP checkpoint selection remain valid for the evaluated samples; this defect does not itself change gradients or selected checkpoints.
- Suggested repair: collect validation metrics after validation, and write a stable schema at the end of the training epoch using explicitly named `train_loss_epoch` and `train_loss_step`. Preserve validation-only use if that workflow is desired. Avoid interpreting old CSV train-loss trends as epoch curves.

Evidence: `CH/checks/independent_verification.py`, `CH/checks/independent_verification_results.json`.

## Checks that passed

- StreamingBinaryMetrics clears accumulated predictions and targets each epoch. In the two-epoch synthetic run, deliberately reversing rankings changes AP from 1.0 to 0.5 and AUROC from 1.0 to 0.0; the previous epoch is not retained.
- The callback computes ranking metrics from all stored validation predictions rather than averaging minibatch AP values. This is the correct single-process approach for rare fraud labels.
- Padding is respected by original flat, my flat, and my hierarchical encoders, with bidirectional and causal options; my linear and periodic tokenizers were both exercised. Changing all padded continuous values to 1000 and padded categorical values to another valid category causes zero output change. Changing a sample's batch companions causes only floating-point noise, below 6e-8.
- Phase 5 `freeze_encoder=True` correctly prevents encoder gradients and updates only head parameters. It is not a hidden fine-tuning path.
- All numerical claims above are local CPU/synthetic results, not evidence of H200 performance or bank-data predictive quality.

## Additional limitations, not blockers for current defaults

1. Frozen Phase 5 probes still leave encoder dropout active in training. A synthetic repeat of identical inputs produces different outputs, as expected for active dropout (maximum logit difference 0.713 in the saved check). This can be a valid regularization choice, but it is not the same experiment as a linear model on fixed exported embeddings. Set the frozen encoder to evaluation mode if fixed-feature linear probing is the intended comparison. Current configs use `freeze_encoder: false`.
2. Metrics are process-local and the datasets do not shard by distributed rank. `sync_dist=True` for loss does not make AP/AUROC global; CSV writes also lack rank protection. The documented single-H200 usage avoids this additional contract gap. If scaling to multiple GPUs, add rank-aware data partitioning, score/target collection across ranks, and rank-zero output writes before trusting global validation claims.
3. bf16 validation computes softmax on the original logits before casting probabilities to float32 (`fraud/callbacks.py:66-67`), so low-precision score ties can remain. Final scoring uses float32 models. Compare on identical samples and report precision if judging extremely small differences. No measured bank-data effect is claimed.

## What the architecture documentation actually supports

`docs/tabular_transformer_models.md` reports UCI Bank Marketing, with term-deposit subscription as its target. It is not a fraud benchmark. The document itself states that the SAINT and FT/MLP families use different split procedures, train sizes, batch sizes, and training settings. Therefore “production recommendation,” “relative ranking is likely to hold,” and “intersample attention hurts” go beyond the experiment shown. It supports an initial engineering choice of FT-Transformer, not evidence that it dominates alternatives for this bank's fraud data.

The Phase 4 implementation follows the later design decisions in material ways: each field becomes a token, transaction position is added, the target is last in the real transaction window, PAD positions are masked, labels are used for targets rather than historical input, and a CLS vector feeds the classifier. Flat A allows feature interactions across the whole history. Hierarchical B first compresses each transaction with a shared transformer, then models relationships between the transaction summaries. Architecture B is implemented. The hierarchical transaction encoder is freshly initialized rather than automatically loaded from Phase 3 weights; the plan's “reuse” describes a design relationship, not an implemented checkpoint transfer.

The names in early plans are not exact diagrams of the code: there is no separate learned SEP token in the current flat path. Transaction boundaries are represented by shared position embeddings. This is a defensible implementation of the later plan, not by itself a defect.

There are meaningful confounds when comparing Phase 3 and 4 as if the only change were history:

- Phase 3 uses vendored PyTorch Tabular blocks with GEGLU and post-residual LayerNorm (`src/pytorch_tabular/models/common/layers/transformers.py:19-31,124-128`). Phase 4 uses PyTorch GELU TransformerEncoderLayer with `norm_first=True`.
- The vendored attention block defaults each head's dimension to the full embedding dimension (`transformers.py:103`), whereas PyTorch's encoder divides the embedding width across heads. With width 32 and eight heads, the former internal attention width is 256, not 32.
- The vendored feed-forward branch consumes the raw attention output `y`, not the normalized residual `x` (`transformers.py:127`). This is a concrete inherited deviation from the usual Transformer block; verify the intended upstream implementation before treating the library path as a faithful reference. Do not silently repair this while comparing existing checkpoints, because that would change the baseline architecture.
- Neither the checked-in UCI document nor this code inspection establishes that any current architecture reproduces a particular paper exactly. Pin the intended paper/repository/version and document tokenizer, normalization, activation, initialization, attention width, training protocol, and head choices before making that claim.

The useful experimental control is the same Phase 4 architecture at `history_len=0`, trained on the same files and scored on the same untouched data, alongside h5/h10/h16. Hierarchical B's one-transaction control has two encoder stages, so describe that additional capacity as part of its architecture. Claiming that observed lift is exclusively caused by history requires these matched comparisons, preferably repeated across seeds and accompanied by uncertainty at the account level.

## Private reproduction record

CH’s original private audit retains the exact synthetic-check commands and outputs. The public copy preserves the findings; it does not include the executable reproduction scripts.

The check produces `CH/checks/independent_verification_results.json`. The CSV example is generated in a temporary directory and summarized there; no production artifacts are read or changed.
