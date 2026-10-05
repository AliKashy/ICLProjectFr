# TF: my proposed next steps, based on CL’s recommendations

> Public adaptation prepared by CH from CL’s dated 2026-10-05 documents. CL remains the original author; this adaptation does not imply CL approved its wording. I use **TF** for my project, **Ya** for my collaborator, **My** for my storage ownership and **Ya’s** for Ya’s storage ownership. Dataset counts, rates, scores and model details are retained with my approval.
>
> This is a documentation-only copy. Code paths such as `my_work/` and `ya_work/`, storage addresses, and references to private proofs are illustrative aliases or evidence descriptions—not files or commands available in this public folder. Test execution and findings are credited to the assistant that reported them. Status is historical as of the dated review.

Original author: CL. Date: 2026-10-05. I am using the following as my proposed plan, based on [audit findings](AUDIT_FINDINGS.md) (finding ids C-xx), CH's audit,
Ya’s plan (`ya_work/docs/phase4_sequential_plan.md`, `ya_work/docs/phase4_design_decisions.md`), a related bank model repository, and
the literature listed in Section 6. Costs are rough H200 hours; a 10-file, 10-epoch history-5 run is taken as
about one hour, a 100-file run as several hours.

CL records agreement with CH as decision D-004 (the private joint decision record): verified loader coverage (1.1), a reproducible
environment (1.4), an identified evaluation cohort (2.x) and unique prediction artifacts (1.3) are required before
new comparative runs. The other Section 1 fixes gate only the experiments that use the affected path (the Phase 5
fixes gate Phase 5; the EMA fix gates EMA; the causal-mask fix gates causal-mode runs). Measurement planning
(Section 2) and independent verification proceed in parallel. (An earlier version of this paragraph broadened
D-004 to "nothing in Sections 2 to 4 before all of Section 1"; CH rejected that attribution and it is
withdrawn.)

---

## 1. Fix first (no GPU needed; one day of work)

| # | Action | Why | How |
|---|---|---|---|
| 1.1 | Apply the loader sharding fix to `ya_work/fraud/data/sequence_builder.py` and `ya_work/fraud/data/parquet_loader.py` | C-01: every documented multi-worker training run so far saw a random two-thirds of its files per epoch | the private proposal fraud_data_*.patch; verify with CL’s shard_demo_CL.py proof (private) (expect 0 missing per epoch) |
| 1.2 | Pin the evaluation cohort | C-06: one file vs eleven | Set `eval.file_start: 8000`, `eval.file_end: 8010` in every `my_work/` config; have every scorer print and store `n`, `positives`, and a SHA-256 of the sorted `fi_transaction_id` list; put that digest in `results.csv` |
| 1.3 | Give every run its own prediction path and fix score-side override order | C-07, C-08 | the private proposal my_phase4{a,b}_scripts_run_ablations.patch; also make `run_ablations.py` exit non-zero if any run failed |
| 1.4 | Make the package installable | C-09 | the private proposal pyproject.patch; verify an editable install and imports of both project packages and the vendored model library |
| 1.5 | Re-vendor `pytorch_tabular` at 1.1.1 (fixes C-10 only) and fix checkpoint loading separately | C-10: Ya’s Phase 3 scripts cannot start from this checkout. C-11 needs its own fix: CH verified that 1.1.1's helper still calls `torch.load` without `weights_only` | Replace `ya_work/src/pytorch_tabular` with upstream 1.1.1 (keep the stacking addition if wanted); pass `weights_only=False` in the vendored loader or pin torch below 2.6; record both in `pyproject.toml`; acceptance-check constructor and checkpoint reload separately |
| 1.6 | Fix the causal mask in all three models | C-02 | Three `*_sequential*.patch` files; verify with CL’s causal_leak_demo_CL.py proof (private) (expect 0.0 at all depths) |
| 1.7 | Fix SupCon view ordering; fix masked-SSL field identity | C-04, C-05; gate Phase 5 only | the private proposal my_phase5_pretrain_supcon.patch; for SSL add a learned per-column embedding to every token (visible and masked) in `TabularEncoder` and test that two masked numeric fields can get different predictions |
| 1.8 | Fix EMA checkpointing before ever enabling EMA | C-03 | the private proposal my_phase4{a,b}_callbacks.patch; CL reports that v3 updates the shadow at every completed optimizer step, including the final partial accumulation group; independent verification of v3 is pending |
| 1.9 | Recover the pod's run artifacts | C-06, C-19 | Retain result tables, training metrics and run logs in the authorized private experiment archive; answer [open questions](../QUESTIONS.md) |
| 1.10 | Reporting fixes | C-14, C-15 | Log `train_loss_epoch`; report the selected epoch's AP and AUROC together; keep unrounded values in CSV |

Acceptance for Section 1: `pytest` passes; `shard_demo` shows a partition; `causal_leak_demo` shows zero;
`supcon_demo` matches the reference; one 2-file smoke run on the pod completes and its `score.log` prints the positive count of the pinned cohort
(3,818 is expected for files 8000 to 8010 only if that count is confirmed for the same cohort, artifact and
label version). This public document does not include or authorize publication of raw run logs, checkpoints or prediction rows.

---

## 2. Measurement protocol (adopt before the next sweep)

1. **Seeds.** Every configuration that will be compared runs with at least 3 seeds (5 for finalists). Report
   mean and spread. Cost multiplies by 3 to 5; compensate by dropping uninformative axes (optimizer, mask).
2. **Paired comparison.** Compare two models on identical transaction ids. Bootstrap by resampling **accounts**
   (windows of one account are correlated), 1,000 resamples, and report the interval of the AP difference.
   The approximate single-estimate standard error (`sqrt(AP(1-AP)/n_pos)`, Boyd et al. 2013) is only a sanity
   check. Keep per-run predictions (1.3) so this is possible.
3. **Selection.** Select on validation PR-AUC (files 4950 to 4954) as now. Report evaluation PR-AUC on the pinned
   11-file cohort once per finalist, not per sweep row. Treat the 11-file set as a development set: it has been
   inspected repeatedly.
4. **Final holdout.** Define a later, untouched cohort with mature labels (label-maturity rule written down)
   and use it exactly once for the final report. Check that no account's history crosses from training into
   that cohort's scored windows in a way that would not exist in production.
5. **Operational metrics.** Alongside PR-AUC, report precision and recall at the bank's alert budget (0.1%,
   0.5%, 1%), fraud-amount-weighted recall, and Brier score, on the same cohort as the production XGBoost.
6. **Run manifest.** Each run writes a JSON with git commit, resolved config, library versions, train/val/eval
   file lists, artifact version and hash, selected epoch, checkpoint hash, prediction URI, cohort digest.

---

## 3. Experiments, in the order that answers the most for the least GPU

| # | Experiment | Question answered | Design | Cost |
|---|---|---|---|---|
| 3.1 | **History 0 control** on the Phase 4 flat model | Is the 0.125 to 0.14 lift history, or the architecture change from the library FT-Transformer? (C-20) | `sequence.history_len=0`, same files, same knobs, 3 seeds | 3 h |
| 3.2 | **Time delta on/off** at h5 | How much of the lift is velocity? (C-20) | `sequence.time_delta=false`, 3 seeds | 3 h |
| 3.3 | **Rerun h5/h10/h16 flat on the fixed loader**, 3 seeds | Replace the single-seed, bootstrap-sampled numbers | existing `ft_decisions_flat_long.yaml` plus seeds axis | 12 h |
| 3.4 | **Hierarchical B at h5/h10/h16**, 3 seeds, same knobs | A vs B, matched knobs; B's memory headroom | existing `ft_decisions_hier.yaml` plus seeds | 9 h |
| 3.5 | **Longer histories with B**: 32, 64 | Does more history keep helping? Accounts have median 133 transactions | B only; watch stage-1 cost (`B*T` encodings) | 6 h |
| 3.6 | **Capacity sweep on the winner** (decision 6) | Is the history model capacity-limited, unlike the single-transaction one? | embed {48, 64, 96} x blocks {2, 4, 6}, lr scaled down for deeper; or run `tune.py` with a memory-aware batch policy | 10 to 20 h |
| 3.7 | **XGBoost head-to-head and stacking** | The business question | Score the production XGBoost and the transformer on the same pinned cohort; then add the transformer score as one feature to XGBoost using out-of-fold or earlier-period transformer predictions for the XGBoost training rows, and measure the delta at the alert budget on an untouched cohort | CPU + 2 h |
| 3.8 | **Downsampling recipe** (roadmap lever 4) | Does training at ~12% fraud help ranking? | keep all fraud, sample non-fraud to 12%; evaluate at natural prevalence; apply prior-shift correction (Saerens et al. 2002) only for probabilities, not for ranking | 4 h |
| 3.9 | **Per-feature periodic embedding** (PLR) with sigma in {0.01, 0.1, 1} | Was "periodic loses" an artifact of the shared projection and fixed sigma? (C-18) | implement per-feature `nn.Linear` followed by ReLU inside `PeriodicNumericalEmbedding` (that is PLR; without the ReLU it is PL); 3 seeds at the best setting | 4 h |
| 3.10 | **Phase 5 after fixes**: scratch vs masked-SSL vs SupCon, identical encoder and data | Does pretraining help? | matched validation ranges (ssl.yaml uses 4950 only today); balanced sampler as a separate axis; then transfer into B's stage-1 encoder | 15 h |

Stop rules (planning heuristics, not derived statistical tests; the hour counts above are unbenchmarked estimates): if 3.1 shows the h0 model at or above 0.135, the "history" story needs rethinking before anything
else; if 3.3 shows seed spread above 0.005, every earlier single-seed conclusion is withdrawn and 5 seeds
become mandatory.

---

## 4. Modeling improvements worth a try after Section 3

- **Time-gap encoding.** Replace the single log-delta scalar with a learned embedding of bucketed log-delta
  (for example 24 log-spaced buckets) or Time2Vec (Kazemi et al. 2019); velocity is a plausible source of history
  signal (a hypothesis until 3.2 runs) and deserves more than one scalar token.
- **Memory-efficient attention for the flat model.** `nn.TransformerEncoderLayer` already uses scaled-dot-product
  attention; with `norm_first=True` the nested-tensor fast path is disabled (the warning in the tests). Setting
  `enable_nested_tensor=False` silences it; for h16, measure peak memory and throughput before and after the
  causal-mask change at the intended batch, history and precision: the merged mask is a dense `[B*H, L, L]`
  tensor, which PyTorch converts to float internally either way, and whether a fused attention kernel accepts
  an arbitrary mask depends on the backend and must be checked on the GPU, not assumed.
- **Readout.** Compare CLS readout with mean pooling over the target transaction's tokens; on short sequences
  the difference is often small, but it removes the CLS backchannel question entirely.
- **FT-Transformer fidelity.** If a claim of "FT-Transformer" matters to reviewers, switch the feed-forward
  activation to ReGLU and skip only the pre-normalization before the first block's attention (`first_prenormalization=False` in the reference code, which keeps pre-norm everywhere else) as in Gorishniy et al. 2021; otherwise call the models
  "FT-Transformer-style".
- **Target transaction emphasis.** Add a learned "is target" flag token or embedding so the model can separate
  target features from history features without relying on position alone.
- **Account-level features as a cheap baseline.** Compute simple velocity features (count and amount in the last
  1 h / 24 h / 7 d, time since last transaction) and give them to XGBoost; if that closes most of the gap, it
  changes the production story.

---

## 5. Process and hygiene

- Keep the private collaboration bridge as the record of decisions; move agreed items into the private joint plan with owners.
- One dated experiment ledger instead of three documents that disagree (`roadmap`, `current_standing`, guides).
- Fix stale paths in docstrings and config comments (C-22); delete or implement `model.type` routing; restore
  TensorBoard logging or delete the `experiment.*` config block.
- Add tests that would have caught the defects: multi-worker unique-id coverage; two-layer causal invariance;
  checkpoint equals validated weights with EMA on; SupCon loss on a hand-built batch equals the reference;
  masked-SSL distinct predictions for two masked numeric fields; scorer round trip with `pos_weight=5`.
- Vectorize `encode_categorical` (C-26); the per-value Python loop is a likely throughput cost for the
  sequence builder (not profiled; measure before and after).
- I need to discuss C-01, C-02, C-10, C-11, C-12 and C-24 with Ya: they affect Ya’s original implementation and interpretation of the Phase 3/4 results too.

---

## 6. Literature behind the recommendations

- Gorishniy, Rubachev, Khrulkov, Babenko. *Revisiting Deep Learning Models for Tabular Data.* NeurIPS 2021.
  FT-Transformer definition; pre-norm blocks (no pre-norm before the first block's attention), ReGLU, CLS readout; roughly matches GBDT across benchmarks.
- Gorishniy, Rubachev, Babenko. *On Embeddings for Numerical Features in Tabular Deep Learning.* NeurIPS 2022.
  Periodic (PLR: periodic features, per-feature linear layer, ReLU) and piecewise-linear embeddings; frequency scale sigma is a
  tuned hyperparameter. Basis for 3.9.
- Grinsztajn, Oyallon, Varoquaux. *Why do tree-based models still outperform deep learning on typical tabular
  data?* NeurIPS 2022 Datasets and Benchmarks. Trees remain strong on plain tables; motivates 3.7.
- McElfresh et al. *When Do Neural Nets Outperform Boosted Trees on Tabular Data?* NeurIPS 2023 (TabZilla).
  Light hyperparameter tuning often matters more than architecture; motivates 3.6 and the existing `tune.py`.
- Rubachev et al. *TabReD: Analyzing Pitfalls and Filling the Gaps in Tabular Deep Learning Benchmarks.* 2024.
  Time-based splits change rankings; motivates the temporal holdout in Section 2.
- Padhi et al. *Tabular Transformers for Modeling Multivariate Time Series.* ICASSP 2021 (TabBERT). A
  hierarchical field-then-row transformer for transaction sequences; the design Architecture B follows.
- Khosla et al. *Supervised Contrastive Learning.* NeurIPS 2020. The SupCon loss and its reference layout
  (view-major concatenation), which the patch restores.
- Bahri, Jiang, Tay, Metzler. *SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption.*
  ICLR 2022. The corruption augmentation used in `supcon.py`.
- Somepalli et al. *SAINT.* 2021. Intersample attention; lost to FT-Transformer in Ya’s benchmark.
- Boyd, Eng, Page. *Area Under the Precision-Recall Curve: Point Estimates and Confidence Intervals.* ECML PKDD
  2013. Binomial and logit intervals with n = number of positives; basis for the standard-error approximation.
- Saerens, Latinne, Decaestecker. *Adjusting the Outputs of a Classifier to New a Priori Probabilities.*
  Neural Computation 2002. Prior-shift correction after downsampling (3.8).
- Kazemi et al. *Time2Vec.* 2019. Learned periodic time representation (Section 4).
- Chen, Guestrin. *XGBoost.* KDD 2016. The production baseline.
- Loshchilov, Hutter. *Decoupled Weight Decay Regularization.* ICLR 2019. AdamW.

A literature pass by a research agent (part of the verification workflow) was still running when this was
written; any additional references it surfaces will be appended here before the freeze or recorded in
CL’s private errata record after it.
