# TF: audit findings for my project

> Public adaptation prepared by CH from CL’s dated 2026-10-05 documents. CL remains the original author; this adaptation does not imply CL approved its wording. I use **TF** for my project, **Ya** for my collaborator, **My** for my storage ownership and **Ya’s** for Ya’s storage ownership. Dataset counts, rates, scores and model details are retained with my approval.
>
> This is a documentation-only copy. Code paths such as `my_work/` and `ya_work/`, storage addresses, and references to private proofs are illustrative aliases or evidence descriptions—not files or commands available in this public folder. Test execution and findings are credited to the assistant that reported them. Status is historical as of the dated review.

Author: CL. Audit date: 2026-10-05. Environment: CPU container, Python 3.11.15, torch 2.14.1+cpu,
pytorch-lightning 2.6.6, pyarrow, scikit-learn, s3fs installed; no S3 access, no GPU, no bank data.
Everything below was read in full by CL (every file under `ya_work/fraud/`, `my_work/`, `ya_work/docs/`, `ya_work/tests/`,
`ya_work/notebooks/`, `reconstruction/`, the FT-Transformer parts of `ya_work/src/pytorch_tabular/`, and a related bank model
repository), then checked by running the test suites, static analysis, import checks, the sweep drivers in
dry-run mode, and purpose-written proof scripts (`CL/scripts/`). CH audited the same commit
independently (CH’s separate audit); where a finding was found by both, or first by CH, that is stated.

Public companions: [project guide](PROJECT_GUIDE.md), [recommendations](RECOMMENDATIONS.md), and [open questions](../QUESTIONS.md). Private patch, proof and machine-readable finding files are not included here.

---

## 1. What the audit means for my work

The checks found no transcription-specific failure: all 146 Python files parse, every targeted import
succeeds, all 28 offline tests pass, no identifier is undefined, and the modules duplicated across phase
folders match each other line for line (the original photographs were not re-inspected, so this is strong
evidence, not a certification of every line). My work follows Ya’s Phase 4 design in its structural
decisions (Section 3 lists the PARTIAL and DEVIATES rows) and goes beyond it with a working hierarchical model,
a sweep driver, pretraining recipes and a scoring toolkit. The best documented history runs scored above the
documented single-transaction baseline (about 0.125 to about 0.14 validation PR-AUC); attribution of that
lift to history alone is not yet isolated, because the history models also change the encoder and training
details and no history-0 control is documented. Four defects affect how the reported numbers should be read:
a multi-worker file-sharding bug in Ya’s loaders that every documented multi-worker training configuration
used, a "causal" mask that does not meet its contract and so weakened one ablation, an evaluation range that
is one file in every config but eleven in the documents, and a sweep driver that overwrote its own prediction
files. Two further defects would corrupt Phase 5 pretraining when it runs (SupCon label ordering, masked-SSL
field identity). The reported defects concern the recovered project logic rather than a demonstrated transcription error. Proposed patches exist for seven of the items;
they are tested offline in disposable copies and are not applied to the project code (Section 8).

---

## 2. Findings

Severity: **high** = changes or invalidates a reported number or a planned experiment; **medium** = wrong
behavior that has not yet affected a reported number, or a reproducibility blocker; **low** = reporting or
hygiene; **info** = documentation. Reported by: CL, CH, or both independently.

| ID | Sev | Kind | Where | Finding | Evidence | Fix / status | By |
|---|---|---|---|---|---|---|---|
| C-01 | high | bug | `ya_work/fraud/data/sequence_builder.py:136-143`, `ya_work/fraud/data/parquet_loader.py:123-130` | With `shuffle=True` and `num_workers>1`, each worker shuffles the full file list independently before taking `files[id::n]`; the slices are not a partition. About `(3/4)^4 = 32%` of files are skipped per epoch with 4 workers and others read twice; over 10 epochs files were seen 4 to 16 times. Every documented multi-worker training configuration (Ya Phase 3/4, all my sweeps, Phase 5 configs; `sequential_smoke.yaml` uses zero workers) uses this path; validation and scoring (no shuffle) do not. | CL’s shard_demo_CL.py proof (private), CL’s private evidence (proof_runs_CL.txt); CH’s private data-pipeline result record | the private proposal fraud_data_sequence_builder.patch, `fraud_data_parquet_loader.patch`: shared per-epoch seed `worker_info.seed - worker_info.id`, then shard. Proof: 0 missing, exactly once per epoch. | Both |
| C-02 | medium | bug | `ya_work/fraud/sequential_model.py:94-106`; copied in `my_work/phase4a/sequential.py:60-68`, `my_work/phase4b/sequential_hier.py:88-95` | Causal mask allows every token to attend the CLS key; CLS attends everything; from layer 2 the target leaks into history tokens (flat max delta 0.0067 at 4 layers, hier 0.217). The bidir-vs-causal ablation therefore did not compare what it meant to compare; the two attention graphs still differ (CH measured different logits from the same weights), so the small AP gap is unattributed, not explained. No label or post-target leakage for target-only scoring. Blocks the planned "predict every transaction" mode. | CL’s causal_leak_demo_CL.py proof (private); CH `phase4_checks.stdout.json` | Three `*_sequential*.patch` files: `allowed = (ti < 0) \| ((tj >= 0) & (tj <= ti))`. Proof: zero leak at all depths. | Both |
| C-03 | medium (latent) | bug | `my_work/phase4a/callbacks.py:29-42`, `phase4b` copy; `scripts/train_sequential.py:112-123` | EMA weights are swapped in for validation and restored in `on_validation_end`; ModelCheckpoint saves afterwards, so `best.ckpt` holds raw weights while `valid_ap` was computed on EMA weights. Docstring claims the opposite. `ema: false` in all configs, so no reported number affected. Also updates per micro-batch under accumulation. | CL’s ema_demo_CL.py proof (private) (ckpt == raw); CH `phase4_checks` (0.5 vs 1.0) | the private proposal my_phase4{a,b}_callbacks.patch: `on_save_checkpoint` writes shadow into `state_dict`, keeps raw + shadow for resume. Proof: ckpt == EMA. v3: shadow updated at every completed optimizer step (tracked through `trainer.global_step`), including the final partial accumulation group; CH's accumulation harness passes on v3 (CL’s private evidence (CH_harness_on_CL_v3.txt)); awaiting independent verification. | Both |
| C-04 | high (Phase 5) | bug | `my_work/phase5_pretrain/supcon.py:37,40` | Features reshaped batch-major (`reshape(B*V,-1)`), label mask tiled view-major (`repeat(V,V)`): positives are assigned to the wrong rows. Perfectly class-clustered views get loss 7.77 vs reference 1.10; mixed views 1.10 vs 7.77. Tests check only scalar + gradient. No Phase 5 run is documented in the snapshot (a pod-side run cannot be excluded). | CL’s supcon_demo_CL.py proof (private); CH `phase5_repro_results.json` | the private proposal my_phase5_pretrain_supcon.patch: view-major `torch.cat(torch.unbind(features,1),0)`. Proof: equals reference. | Both |
| C-05 | high (Phase 5) | bug | `my_work/phase5_pretrain/ssl_pretrain.py:50-53,65-78` | One shared mask token and one shared numeric decoder; masked tokens carry no field identity, so two masked numeric fields in one row get identical predictions (0.50074 for targets +2 and -2). Categorical heads are per-field and unaffected. No run documented in the snapshot. | CH `phase5_repro.py`, re-run by CL with identical output | Open (bridge W-03): add a learned per-column embedding to all tokens, or per-field numeric heads; add a test. | CH |
| C-06 | high | docs / reproducibility | `my_work/phase4a/configs/sequential.yaml:81-87`, `phase4b/configs/sequential_hier.yaml:88-94`, all three sweep specs, `my_current_standing.md:142-150` | Every config evaluates `part_8000` only (228,043 rows, 290 positives); no sweep overrides it; documents quote 11-file numbers (8000-8010, 3,818 positives) and compare with Ya’s 11-file 0.0940. The repo cannot tell which produced the `eval_ap` column. | Config reading; dry runs | Needs the pod's `score.log` (`n=`, `positives=`). the [evaluation-cohort question](../QUESTIONS.md). Pin `eval.file_end=8010` in configs. | Both |
| C-07 | medium | bug | `my_work/phase4a/scripts/score_sequential.py:125-128` (4b same); `run_ablations.py:134-139` | One fixed object name `sequential_preds.parquet` per `predictions_base`; the driver never overrides it per run. Every sweep run overwrote the previous run's predictions on S3; only the last write survives unless S3 versioning or backups exist (not inspected). Blocks per-run bootstrap, paired comparison, calibration, ensembles. | Code reading | the private proposal my_phase4{a,b}_scripts_run_ablations.patch: `eval.predictions_base=<base>/<run_name>`. | Both |
| C-08 | low (latent) | bug | `run_ablations.py:114` vs `:136` | Train passes `common + overrides`, score passes `overrides + common`; an axis key that overlaps a common key would train one config and score another. No overlap in the three current specs. The driver also swallows non-zero subprocess codes (exit 0 even if a run failed). | `--dry-run`; OmegaConf merge order | Precedence fixed in the same patch as C-07. Exit-status propagation open. | Both |
| C-09 | medium | ops | `pyproject.toml:3,31`; `requirements.txt` | Build backend `setuptools.backends._legacy:_Backend` does not exist (`pip install -e .` fails); `my_work*` not in `packages.find`; `pytorch-tabnet` (imported unconditionally by the vendored library) and `pyyaml` undeclared. | CL’s private evidence (test_lint_install_CL.txt) | the private proposal pyproject.patch: `setuptools.build_meta`, add `my_work*`, add deps. Proof: editable install succeeds. | Both |
| C-10 | medium | ops | `ya_work/src/pytorch_tabular/config/config.py:206-230` vs `ya_work/fraud/streaming_model.py:30-36` | Vendored fork is upstream 1.1.0; `InferredConfig.output_cardinality` first appears in upstream **1.1.1** (verified from both sdists). Ya’s Phase 3 streaming train/score cannot construct the model from this checkout. The library version was pinned during reconstruction. | `pip download pytorch_tabular==1.1.0/1.1.1`; CH `original_integration_repros.py` | Re-vendor upstream 1.1.1 (or 1.2.0) into `ya_work/src/`, or pin the installed library; do not edit Ya’s builder. Open. | CH found; CL traced root cause |
| C-11 | medium | ops | `ya_work/src/pytorch_tabular/utils/python_utils.py:74-84`, `tabular_model.py:1494` | Vendored checkpoint loader calls `torch.load` without `weights_only=False`; with torch >= 2.6 the baseline's best-checkpoint reload fails on the OmegaConf object in the checkpoint. Ya’s own scorers already pass `weights_only=False`. | CH `original_integration_repros.py`, re-run by CL | Open; version-pin torch or patch the vendored helper (re-vendoring 1.1.1 does NOT fix this: CH inspected the 1.1.1 helper and it still calls `torch.load` without `weights_only`). | CH |
| C-12 | medium (latent) | bug | `ya_work/fraud/scripts/score_sequential.py:89-101`, `score_streaming.py:70-79`, Phase 5 `score_*.py` | A model trained with `pos_weight != 1` has a `loss_fn.weight` buffer; these scorers rebuild with the default and load strictly, so the load fails. My Phase 4 factories forward `pos_weight`, so my scorers are fine. All runs used weight 1. | CH `original_project_repros.py`, re-run by CL | Open (bridge W-04): rebuild from checkpoint hyperparameters or forward `pos_weight`. | CH |
| C-13 | medium (latent) | bug | `my_work/phase5_pretrain/scripts/train_ssl_finetune.py:91-98`, `train_contrastive.py:92-99`, `ssl_finetune.py:55-59` | Missing pretrained checkpoint only warns and trains from scratch under the pretrained run's name; `load_pretrained` uses `strict=False` and accepts an empty state. | Code reading; CH probe (`load_pretrained({})` succeeds, 18 missing keys) | Open: require the checkpoint and every `encoder.*` key. | CH |
| C-14 | low | reporting | `ya_work/fraud/callbacks.py:23-40` (`CSVMetricsCallback`) | Writes at validation-epoch end before Lightning reduces training metrics; `train_loss` column is the last micro-batch, not the epoch mean. Explains the erratic `train_loss` values in `my_current_standing.md`. `valid_*` columns are correct. | CH `independent_verification.py`, re-run by CL (CSV 9.0 vs mean 5.0) | Open: log `train_loss_epoch` and write at train-epoch end. | CH |
| C-15 | low | reporting | `run_ablations.py:125-127` | `best_valid_ap` and `best_valid_auroc` are independent maxima over epochs; h10's AP epoch (6) has AUROC 0.946059 while the table shows 0.948879 (epoch 9). | Arithmetic on `my_current_standing.md` rows | Open: report selected-epoch pair. | CH |
| C-16 | low-medium | bug | `my_work/scoring/ensemble.py:37-39,42-51`, `leaderboard.py:80-108`, `calibrate_and_report.py:54-60` | Rank percentile via double argsort breaks ties arbitrarily (AP 0.5 becomes 1.0 on tied scores); inner joins drop or duplicate ids silently and trust the first file's labels; leaderboard bootstraps only the last seed; isotonic calibration can create ties that change AP; alert budgets unvalidated. | CH `evaluation_checks.py`, re-run by CL | Open: average ranks, cohort id checks, per-seed CIs. | CH |
| C-17 | medium | missing-work | `my_work/scoring/temporal_eval.py`; `my_current_standing.md:154` | The "temporal holdout" item is unimplemented as a protocol; `temporal_eval.py` only buckets existing predictions by period. | Code reading | Define an untouched later cohort with mature labels. | Both |
| C-18 | medium | methodology | `my_work/phase4a/numerical_embeddings.py:16-25` | Periodic embedding shares one `nn.Linear(2F, d)` across all seven history-model numeric channels (six raw features plus time delta) and fixes `sigma=1.0`; Gorishniy et al. 2022's PLR uses per-feature linear layers followed by ReLU and tunes the frequency scale; this variant has neither the per-feature layers nor the ReLU, so it is a shared-projection PL variant (a legitimate alternative to evaluate, not a coding error). Zero inputs give identical tokens for every feature. The "linear beats periodic" conclusion is about this variant. | Code reading vs paper | Try true PLR (per-feature linear + ReLU) with sigma in {0.01, 0.1, 1}. | Both |
| C-19 | medium | methodology | `my_current_standing.md:30,79`; `my_phase4a_guide.md:23` | Single seed (42) everywhere; "statistically tied", "only lever that matters", "Architecture A is saturated" exceed the evidence. CL’s illustrative marginal SE calculation is about 0.004 (Boyd et al. 2013 binomial form); h10 vs h16 (0.0017) and h5 vs h10 (0.0034) are below that approximation, which is not a project confidence interval or a paired significance test. A paired, account-blocked bootstrap requires retained predictions, affected by C-07. | Arithmetic | 3 to 5 seeds; paired bootstrap; keep per-run predictions. | Both |
| C-20 | medium | missing-work | `ya_work/docs/phase4_sequential_plan.md` 9.3, 9.4; `ya_work/docs/phase4_design_decisions.md` decision 6 | Not documented in the audited snapshot (no config, result row or log): history 0 vs 5 control, time-delta on/off, capacity sweep (`tune.py` exists; no trials recorded), leakage check on the trained runs. Pod-side runs cannot be excluded from the repository alone. Six of the eight finder runs went to optimizer and mask axes that were not in the plan. | Config/doc reading | Run h0 and time-delta ablations first (cheap, decisive). | Both |
| C-21 | low | design-deviation | `my_work/phase4b/sequential_hier.py` | Plan's hierarchical (b) says "reuse the validated single-transaction encoder"; stage 1 is a fresh 2-block encoder trained from scratch. Defensible, but B's comparison with A also changes capacity (2+4 blocks vs 4). | Code reading | State as "matched-knob, not matched-capacity"; optionally initialize stage 1 from a Phase 5 encoder. | Both |
| C-22 | info | docs | `my_work/docs/*.md`, file headers, `README.md`, `run_all.sh:2`, config comments | Marginal means slightly wrong (linear 0.132579 not 0.132604; bidir 0.1289745 not 0.128987); sampling formula gives 1,051,145 not 1,051,813; referenced files missing (`run_remaining.ps1`, `count_train_fraud.py`, `ya_work/fraud/scripts/tune.py`, `ya_work/fraud/scripts/pretrain_ssl.py`); "Phase 6" labels in headers; roadmap says h10/h16 "to run" after they ran; `run_all.sh` claims "every run" but holds 13; README is Ya’s stale Phase 1-2 text pointing to an obsolete external documentation location; `model.type` is logged but does not route; `experiment.*` config is dead in my trainer (`logger=False`). | Reading | Documentation pass. | Both |
| C-23 | low (conditional) | bug | `sequence_builder.py:171-184`; `parquet_loader.py:132-137`; both `_get_worker_files` | Sorting uses the raw `order_col`, so non-ISO string timestamps sort lexicographically (EDA shows the real column is `datetime64[ns]`, so not reachable as observed); cyclical datetime encoding with missing flags raises `KeyError` in the streaming loader (unused option); no distributed-rank sharding (single GPU documented). | CH `data_pipeline_checks.py`, re-run by CL | Sort on the parsed timestamp; derive datetime before flags; constrain to one device. | CH |
| C-24 | low | bug | `ya_work/fraud/scripts/train_baseline.py:208-233`; vendored `tabular_model.py:1532-1535` | `save_model` clears the directory that holds `training_metrics.csv`. Ya’s baseline path only. | CH `original_project_repros.py`, re-run by CL | Write metrics outside the model directory. | CH |
| C-25 | info | upstream | `ya_work/src/pytorch_tabular/models/ft_transformer/ft_transformer.py:91`; `common/layers/transformers.py:103,127` | Upstream quirks: feature importance reads feature-to-CLS attention instead of CLS-to-feature; attention head dim equals the full embed dim; FFN consumes the raw attention output. Affect the library single-transaction path only. | Reading; CH probe | None for this project; note when comparing Phase 3 vs Phase 4 architectures. | Both |
| C-26 | info | perf | `ya_work/fraud/data/preprocessing.py:165-175` | `encode_categorical` loops in Python per value: about 14 million iterations per 840k-row file at 17 categoricals. Correct but slow; makes `num_workers=0` an expensive workaround for C-01. | Reading | Vectorize with `pd.Series.map` on a dict. | CL |
| C-27 | question | ops | `my_work/*/configs/*.yaml` (`artifacts.base` under `My`) versus Ya’s storage | Whether my artifacts v1 are a byte copy of Ya’s or refit; refit statistics would break comparability with Ya’s numbers. | Config reading | the [preprocessing-artifact question](../QUESTIONS.md). | CL |
| C-28 | info | hygiene | `ya_work/fraud/data/s3_utils.py:34`; `ya_work/notebooks/eda.ipynb` | TLS verification disabled on the S3 client. | Reading | Use the organization's CA bundle. | CH |

---

## 3. How my work aligns with Ya’s Phase 4 plan

Spec: `ya_work/docs/phase4_design_decisions.md` (decisions 1 to 11) and `ya_work/docs/phase4_sequential_plan.md` (sections 2 to
10). Verdicts: COMPLIANT / DEVIATES / PARTIAL / NOT DONE.

| # | Decision | Verdict | Evidence |
|---|---|---|---|
| 1 | Flat token sequence first; hierarchical deferred | COMPLIANT, and B delivered | `my_work/phase4a/sequential.py`; `my_work/phase4b/sequential_hier.py` |
| 2 | No historical `frd_tag` as input | COMPLIANT | `predictors.csv` has no label; builder reads labels only for the target |
| 3 | Target + N strictly prior, same account; N=5, stride 1; never skip fraud | COMPLIANT | `sequence_builder.py:187-220`; 9 tests in `my_work/tests/test_sequence_builder.py`; notebook check on real data |
| 4 | Time-delta features | COMPLIANT (log1p seconds); ablation NOT DOCUMENTED | `sequence_builder.py:213`; no `time_delta=false` run |
| 5 | Left-pad with mask | COMPLIANT | `sequence_builder.py:200-215`; padding-invariance probes by both audits |
| 6 | Capacity 48-64 / 4 blocks, then capacity sweep | PARTIAL | 48/4 used; no sweep documented (`tune.py` exists, no trials recorded) |
| 7 | Attention configurable, default bidirectional | COMPLIANT in config; causal implementation DEVIATES | C-02 |
| 8 | Transaction-level position only | COMPLIANT | `sequential.py:43,78-80` |
| 9 | Window shuffle buffer spanning many accounts; large batch | COMPLIANT | `buffer_windows: 65536`, `batch_size: 4096` |
| 10 | `train_stride` separate; `eval_stride` always 1 | COMPLIANT by default, not enforced | configs; scorer accepts any `eval_stride` |
| 11 | bf16, batch 4096, accumulation when needed | COMPLIANT | configs; h10 1024x4, h16 256x16 |
| 9.1 | Smoke on a few files | DONE (Ya) | `ya_work/docs/phase4_sequential_plan.md` results |
| 9.2 | Compare vs single-txn ceiling | DONE | 0.1293 / 0.1365 vs 0.1247 |
| 9.3 | Ablate h0 vs h5, time-delta on/off, bidir vs causal | PARTIAL: mask done but uninformative (C-02); h0 and time-delta NOT DOCUMENTED | sweep specs |
| 9.4 | Leakage check | PARTIAL: causality tested offline and in the notebook; no documented check on the trained runs | tests; `sequence_smoke.ipynb` |
| Deferred: hierarchical reusing the validated encoder | DONE as fresh-init B | C-21 |
| Deferred: predict every transaction at once | NOT STARTED; requires C-02 fix | |
| Deferred: longer histories | DONE for flat (10, 16); B pending | |

My additions beyond the plan: periodic numerical embedding, optimizer axis, EMA option, Optuna tuner,
Phase 5 pretraining, scoring toolkit, 100-file confirmation run. Assessment: all reasonable; the optimizer and
mask axes consumed six of eight finder runs and showed gaps too small to resolve at one seed (the mask axis was
additionally compromised by C-02), while the plan's h0 and time-delta ablations, which would be decisive, are
not documented in the snapshot.

---

## 4. Transcription fidelity

- Byte-compilation, import of all 61 project modules, ruff/pyflakes (F, E9, PLE): no undefined names, no
  syntax errors; only unused imports (`Optional` x2, `pytest`, three in `scripts/sync_pytorch_tabular.py`) and one
  unused local (`pos_label`, `score_streaming.py:91`).
- Duplicated modules: `phase4a` vs `phase4b` copies of `tokenizer.py`, `numerical_embeddings.py`, `optim.py`,
  `callbacks.py` differ only in import paths and one docstring word; `phase5_pretrain` copies of the tokenizer and
  embeddings differ only in docstrings. The Phase 5 copies were pixel-verified, so this cross-validates the
  Phase 4 copies that were not.
- My `train_sequential.py` / `score_sequential.py` vs Ya’s: every difference reads as an intended feature
  (factory, EMA hook, dropped TensorBoard logger, log-line changes).
- Reconstruction records: `check_pid.sh` lines 1 to 3 never supplied (generated header); five docs and three
  notebooks are "normalized" (content-faithful, not byte-exact); the SupCon line with the bug appears in the
  pixel-verified record, so C-04 is an original bug.
- Scope: these checks establish internal consistency (parsing, imports, tests, agreement between duplicated
  copies) and support individual recovered lines. Neither audit re-inspected the original photographs, so they
  do not certify every line, every runtime path, or a clean install.

---

## 5. What was checked and found correct

Causal window construction and left padding; padding invariance of all models under both masks and both
embeddings; no identifier or label inputs; frozen preprocessing with reserved UNK/MISSING indices; full-stream
validation metrics reset per epoch; early stopping and checkpoint on `valid_ap`; sweep specs expand to
exactly 8, 2 and 3 runs with correct train-side precedence; score regex matches the scorer's output; my
isolation rule (no commits to `ya_work/fraud/`); gradient-accumulation configs keep the effective batch at 4,096;
SSL and SupCon encoder states transfer exactly into the fine-tuner; the scorers use eval mode and softmax
index 1 consistently with the label encoding.

---

## 6. Reported numbers versus what the repository would produce

| Documented | Repo state | Note |
|---|---|---|
| Val files 4950-4954 | base configs 4950-4950; sweeps override to 4954 | consistent for sweeps; standalone commands use one val file |
| Eval 8000-8010 | all configs 8000-8000 | C-06 |
| h10 batch 1024x4, h16 256x16 | `ft_decisions_flat_long.yaml` | consistent |
| Winner knobs linear + bidir + adamw | common overrides in long and hier specs | consistent |
| "Architecture A OOMs past h10" | h16 flat ran with accumulation | OOM at full batch, not at all |
| Fraud rate train 0.1654% from `count_train_fraud.py` | file absent | unverifiable here |
| `ablations/*/results.csv`, logs, checkpoints | absent from repo; `ablations/` not in `.gitignore` | public questions about saved run artifacts |

---

## 7. Pending additions

An adversarial-verification workflow (8 readers, 6 finders, 3 verifiers per finding, completeness critic) was
still running when this document was written. Any finding it adds that is not already above will be appended
as Section 8 before CL’s original audit folder is frozen; if CL’s original audit folder is frozen first, additions go to CL’s private errata record.

---

## 8. Revisions after CH's independent review (2026-10-05, CH’s first independent review)

CH applied the v1 patches to disposable copies and found: (a) the causal-mask fix left left-padded query rows
with zero allowed keys, giving NaN logits under the eval fast path with two or more layers; (b) the EMA callback
overwrote a restored shadow in `on_fit_start`, so exact resume failed; (c) the runner read the prediction root from
the unmerged base config and did not separate grids; (d) `requests` was still undeclared; (e) the apply-all helper
had a syntax error. All five are CONFIRMED. CL reported that the v2 patches addressed those five cases: the causal mask is merged per
sample with the padding mask and the diagonal kept open (no fully masked rows); EMA keeps a restored shadow; the runner resolves the effective config and namespaces by grid and run;
`requests` and package data are declared; the helper is rewritten around `patch -p1`. CH's own harnesses
(the private bridge evidence archive) were rerun on v2 and pass; outputs are in
the private bridge evidence archive. CH also corrected CL’s wording: runs that are
absent from the snapshot are "not documented", not "never run" (applied above), and the binomial AP interval
(Boyd et al. 2013) is a marginal approximation, not a paired test (C-19).

### 8.1 Proposed patches v3 (2026-10-05, after CH's second review, CH’s second independent review)

CH's verdict on v2: 7 patches CONFIRMED within scope, 4 PARTIAL, with two new counterexamples. (a) EMA: the
`(batch_idx + 1) % accum` guard misses the optimizer step Lightning takes on an epoch's final, partial accumulation
group (3 batches with accumulation 2 ended at shadow 0.10 instead of 0.19; 3 batches with accumulation 4 at 0
instead of 0.075). (b) Runner: namespacing by `basename(output_root)` collides for two experiments whose roots
share a basename, and for a rerun of the same grid. Both were confirmed. CL reports that v3 addresses them: the EMA callback updates the
shadow whenever `trainer.global_step` advances (every completed optimizer step, including the final partial
group), falling back to the batch rule only without a Lightning trainer, and resets its step marker on resume;
the runner writes predictions to `<effective predictions_base>/<full normalized output_root>/<run_name>/<attempt>`,
where `attempt` is a UTC timestamp plus a random suffix (or `spec.attempt_tag` for a deliberate resume), records
`attempt` and `predictions_path` in `results.csv`, and passes `experiment.run_name` to the scorer so the scoring
config matches the training config in every field. CL reports that CH’s harness, rerun unchanged by CL on v3, passes every
case (CL’s private evidence (CH_harness_on_CL_v3.txt)); CL’s v3_regressions_CL.py proof (private) fails on the unpatched
tree and passes on v3 (CL’s private evidence (v3_regressions_CL.txt)). CL reports that all 11 v3 patches apply to a fresh source export
with the private proposal apply_all_patches_CL.py and the 28 tests pass there. Also in v3: the merge-helper docstrings no
longer claim padded positions hold a "zero embedding" (they do not; the isolation argument is that padded keys stay
blocked for every real query and for CLS). Accepted corrections carried into this document: C-11 is not fixed by
re-vendoring 1.1.1; C-18's reference PLR includes ReLU; C-20 wording; history attribution and transcription scope
in Section 1. v3 awaits CH's independent verification; the GPU memory and throughput cost of the merged
causal mask (a dense `[B*H, L, L]` boolean tensor, converted to float by PyTorch) is unmeasured and must be
benchmarked at the intended batch, history and precision before any causal-mode run.
