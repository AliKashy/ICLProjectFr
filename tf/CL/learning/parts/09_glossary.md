# Part 9. Glossary

Original chapter author: CL. Public adaptation: CH, in my first-person project voice.
The original author incorporated an independent CH review on 2026-10-05; this adaptation is not a new joint endorsement.

Status note: experiment status is the recorded 2026-10-04 snapshot; audit and proposed-fix statements describe the 2026-10-05 review. They are not live job or deployment status.
Source paths and commands are illustrative aliases, not files or runnable commands supplied by this public guide. `my_work/` means my implementation; `ya_work/` means Ya’s foundation; `s3://<bank-storage>/` replaces the actual storage address.

- **Ablation** [4]: an experiment that changes one design choice at a time to see what it contributes.
- **Accuracy** [1.5]: fraction of correct yes/no calls; useless at 0.165% fraud.
- **AdamW** [2.9]: the optimizer that adjusts weights during training; "W" means it also applies weight decay.
- **AP, average precision** [1.7]: the metric the code computes (scikit-learn average precision); what this
  project calls PR-AUC.
- **Architecture A / flat** [4]: one transformer over all tokens of the target and its history.
- **Architecture B / hierarchical** [4]: a transformer per transaction, then a transformer over transactions.
- **Artifacts, frozen** [3]: preprocessing statistics and vocabularies computed once on training files and
  reused everywhere.
- **Attention** [2.3]: each token forms a learned, input-dependent weighted average over all tokens.
- **AUROC, ROC-AUC** [1.7]: probability that a random fraud scores above a random legitimate transaction.
- **Backpropagation / backward pass** [1.3]: computing how each weight should change to reduce the loss.
- **Batch** [1.3]: rows processed together in one training step (4,096 here).
- **bf16** [2.7]: a 16-bit number format used on the GPU to halve memory.
- **Bidirectional mask** [2.6]: every token may attend to every other; the default here.
- **Bootstrap** [1.8]: estimating uncertainty by resampling the data many times.
- **Calibration** [1.9]: making output probabilities match real frequencies; a strictly increasing
  recalibration does not change ranking, isotonic can create ties.
- **Causal mask** [2.6]: tokens may not attend to later transactions; the option in this code leaks via CLS.
- **Checkpoint** [1.4]: a saved copy of model weights (`best.ckpt`).
- **CLS token** [2.4]: the extra token whose final vector is classified.
- **Class weighting** [1.9]: counting fraud rows more heavily in the loss; `pos_weight`, unused so far.
- **Continuous feature** [1.2]: a number with magnitude, such as amount.
- **Categorical feature** [1.2]: a code from a fixed list, such as transaction type.
- **Cross-entropy** [1.3]: the loss used for yes/no predictions.
- **Downsampling** [1.9]: dropping most legitimate rows during training to raise the fraud share.
- **Dropout** [2.9]: randomly zeroing some activations during training to reduce overfitting.
- **Early stopping** [1.4]: stop training when validation performance stalls (patience 5).
- **EMA** [2.9]: exponential moving average of weights; optional, never enabled.
- **Embedding** [2.2]: turning a value into a learned vector.
- **Epoch** [1.3]: one pass over the training data.
- **Evaluation set** [1.4]: files 8000 and up, meant for the final honest number.
- **Feature** [1.2]: a column the model may use; 23 here.
- **Feature-limited** [2.8]: a plateau when more capacity, epochs or data stop helping; suggests, but does not
  prove, that the inputs are exhausted.
- **FT-Transformer** [2.5]: Feature Tokenizer + Transformer, the model family used.
- **Fraud rate / prevalence** [1.5]: fraction of fraud rows; 0.1654% in training.
- **GELU** [2.1]: the smooth activation used in the feed-forward blocks.
- **Gradient accumulation** [4]: several small batches per weight update so a large batch fits in memory.
- **Head** [2.4]: the final linear layer producing two logits.
- **History length N** [2.6]: number of prior transactions in the window; 5, 10 or 16 tried.
- **Isotonic regression / Platt scaling** [1.9]: two calibration methods.
- **Label** [1.2]: the answer column, `frd_tag`.
- **Leakage** [1.9]: using information not available at decision time.
- **Learning rate** [1.3]: step size of weight updates (3e-4).
- **Linear embedding** [2.2]: value times a learned vector plus a bias; the winning numeric embedding.
- **Logit** [2.4]: a raw score before softmax.
- **Loss** [1.3]: the number training tries to minimize.
- **Mask (padding)** [2.6]: marks empty history slots so attention ignores them.
- **Masked autoencoder** [5]: pretraining by reconstructing hidden fields.
- **MLP** [2.5]: a plain stack of layers with no attention.
- **Overfitting** [1.4]: memorizing training rows instead of learning general patterns.
- **Padding** [2.6]: empty slots added when an account has fewer than N prior transactions.
- **Periodic embedding** [2.2]: numeric value expanded into sines and cosines at learned frequencies.
- **Position embedding** [2.6]: a learned vector per transaction slot added to its tokens.
- **PR-AUC** [1.7]: this project's name for average precision; the selection metric.
- **Precision / recall** [1.6]: of the alerts, how many are fraud / of the fraud, how much is alerted.
- **Pretraining** [5]: training the encoder on a proxy task before the fraud task.
- **Residual connection** [2.3]: adding a block's input back to its output.
- **Seed** [1.8]: the number that fixes the randomness of a run; every sweep used 42.
- **Sharding** [6.2]: splitting the file list across data-loader workers.
- **Softmax** [2.3]: turns a list of numbers into positive weights that sum to 1.
- **Standard error** [1.8]: typical wobble of an estimate due to finite data.
- **Stride** [3]: use every k-th transaction as a training target (fraud never skipped); 1 here.
- **SupCon** [5]: supervised contrastive pretraining; same-label examples pulled together.
- **Time delta** [2.6]: log(1 + seconds before the target), one per transaction in the window.
- **Token** [2.2]: one feature's vector.
- **Transformer block** [2.3]: attention plus feed-forward plus normalization and residuals.
- **UNK / MISSING** [3]: reserved category indices 0 and 1 in the frozen vocabularies.
- **Validation set** [1.4]: files 4950 to 4954, used for early stopping and model selection.
- **Weight** [1.1]: an adjustable number inside the model; about 10^5 of them.
- **Weight decay** [2.9]: gentle pull of weights toward zero to limit overfitting.
- **Window** [2.6]: the target transaction plus its N prior transactions.
- **XGBoost** [1.10]: gradient-boosted trees; the production baseline.
