# TF: my evaluation, experiment-status and test-review record

> Public first-person adaptation of documents originally prepared by CH on 2026-10-05. I retain the findings and their original review scope; CH performed the reported independent checks. Source paths are illustrative references to my private project, with `my_work/` representing my work and Ya identifying my collaborator. Source code, private evidence files and bank data are not included in this document set. Later changes require a separate review.

Audit date: 2026-10-05. Scope: `my_work/scoring/`, experiment documentation and grids, `my_work/run_all.sh`, model/SupCon test coverage, and reconstruction evidence. This review did not change application source, run bank workloads, or access bank data. Paths and line references below refer to the audited checkout.

## What the reported results actually mean

`history=5` means five earlier transactions plus the transaction being judged. `history=16` means sixteen earlier plus current: seventeen transactions in the model input. **100 files means the size of the training dataset; it does not mean history length 100 or 100 evaluation repetitions.** The repo's known history sweep is 5, 10, and 16.

The latest evidence available here is a copied pod-log snapshot dated **2026-10-04 23:09**, not access to the current running jobs. No `ablations/` directory, result CSVs, prediction Parquets, trained checkpoints, or live pod outputs exist in this checkout. The snapshot is useful evidence of what was reported, but its numbers cannot be independently recomputed here.

| Run | Reported training size | Reported best validation AP | Reported evaluation AP | What is actually established by the snapshot |
| --- | --- | --- | --- | --- |
| Flat h5, linear/bidirectional/AdamW | 10 files | 0.1365 | 0.0893 | Reported winner among eight one-seed h5 configurations |
| Flat h10, same knobs | 10 files | 0.139942 | 0.0902 | Reported completed; best AP at epoch 6 |
| Flat h16, same knobs | 10 files | 0.138256 | 0.1171 | Reported completed; best AP at epoch 8 |
| Flat h5, larger training confirmation | 100 files | 0.141585 | Not yet scored | Reported completed; best AP at epoch 1, stopped after epoch 6 |
| Hierarchical B h5/h10/h16 | 10 files planned | None recorded | None recorded | h5 just started; h10/h16 queued in that snapshot |
| Phase 5 masked SSL or SupCon | Configured, not demonstrated by this snapshot | None recorded | None recorded | Source exists; no experimental outcome can be inferred |

Sources: `my_work/docs/my_current_standing.md:3`, `:16`, `:46`, `:47`, `:85`, `:94`, `:100`, `:110`, `:151`.

The early-stopping explanation is internally consistent: best epoch 1 followed by five non-improving epochs 2–6, patience 5. Choosing on validation AP instead of the largest evaluation AP is the right discipline. h16's evaluation 0.1171 does not establish it as the winner. “Current” statements such as “running now” must retain the snapshot date.

Ya’s reference numbers have actual documentation counterparts. `docs/phase3_streaming_plan.md:25` reports the 100-file single-transaction model at validation AP 0.1247, and `:31`–`:37` reports its 11-file evaluation AP 0.0940. The original in-memory baseline also has AP 0.0940, but on **one** evaluation file, so those two equal numbers must not be conflated. `docs/phase4_sequential_plan.md:127`–`:130` records his history-5 validation AP 0.1293. These are prior documented results, not reverified results of this audit.

## Confirmed findings and their impact

### E1 — High: the checked-in evaluation range contradicts the claimed 11-file comparisons

Both current Phase 4 base configs score **only part_8000** (`my_work/phase4a/configs/sequential.yaml:81`–`:87`; `my_work/phase4b/configs/sequential_hier.yaml:88`–`:94`). None of the three sweep specs overrides `eval.file_end`. Their validation ranges are explicitly widened to 4950–4954, while evaluation is not widened. The scorer uses precisely the configured bounds (`my_work/phase4a/scripts/score_sequential.py:62`; same in 4b).

The standing document calls the evaluation an already inspected 11-file set (`my_work/docs/my_current_standing.md:79`, `:154`). Its supposedly ready command for scoring h5 trained on 100 files also omits `eval.file_end=8010` (`:142`–`:150`). Running that command from this checkout would score one file and could then be compared to Ya’s 11-file result.

**Conclusion:** this is a reproducibility discrepancy, not proof that the historical pod was scored incorrectly. The pod may have had local overrides absent here. Obtain the exact resolved configs, commands, prediction row counts and unique ID counts before treating evaluation APs as comparable. For the historical 11-file cohort, the reference count is 2,466,461 rows and 3,818 positives, subject to identical eligibility rules.

**Suggested fix:** put the intended evaluation range explicitly in each experiment config and the scoring command; record it and the cohort fingerprint in results. Do not rerun or overwrite existing prediction objects until their identity is preserved.

### E2 — High: sweep runs overwrite each other's predictions

The runner gives each experiment a distinct checkpoint directory, but does not give it a distinct `eval.predictions_base` (`my_work/phase4a/scripts/run_ablations.py:114`–`:139`; 4b has the same logic). Both scorers always write `sequential_preds.parquet` at the base configured S3 prefix (`my_work/phase4a/scripts/score_sequential.py:125`–`:128`; 4b same lines).

Consequently, every flat run uses the same object name, and every hierarchical run uses another shared object name. The eight h5 results and the h10/h16 runs are not backed by independently preserved predictions under the default sweep. A later run or the h5@100 scoring command can overwrite the object from an earlier run. Separate local checkpoint/log folders do not prevent this.

This blocks trustworthy per-run bootstrapping, paired comparisons, calibration, and ensembles from those S3 paths: the CSV can say “h5” while the shared prediction object now contains h16 scores.

**Suggested fix:** assign immutable prediction prefixes containing architecture, history, seed, data range and run ID. Write the full object URI and checkpoint identity to the result row. Preserve validation and test predictions in separate destinations.

### E3 — Medium: ensemble ranking creates information that tied scores do not contain

`my_work/scoring/ensemble.py:37`–`:39` computes percentiles with double `argsort`. Equal model scores are assigned different ranks by array order. Those arbitrary differences feed both weight optimization and the printed single-model AP/AUROC (`:50`, `:81`–`:95`). This is especially relevant to quantized outputs and to probabilities calibrated with isotonic regression, which naturally creates ties.

**Reproduced on the actual helper:** four identical scores with labels `[0,0,1,1]` have AP/AUROC 0.5/0.5. The helper changes them to 1.0/1.0. Reversing row order gives AP 0.416667. No model information improved.

**Suggested fix:** use average ranks for ties and test permutation invariance; show single-model metrics on original scores. The reproduction is in `CH/checks/evaluation_checks.py`.

### E4 — Medium: cohort and label integrity are not enforced before comparisons or joins

`ensemble._align` silently inner-joins model outputs and trusts labels only from the first file (`my_work/scoring/ensemble.py:42`–`:51`). It neither enforces unique IDs nor checks that all models contain exactly the intended IDs and agree on targets. Missing predictions disappear; duplicated IDs create multiple joined rows and reweight metrics. Different model cohorts can look stronger simply because harder transactions vanished.

**Reproduced:** first-model IDs `[1,2]`, second-model IDs `[2,2]` produce joined IDs `[2,2]`: ID 1 vanishes and ID 2 is duplicated. Conflicting labels in the second file are ignored.

The leaderboard independently scores each file and does not read transaction IDs at all (`my_work/scoring/leaderboard.py:80`–`:108`), so equal cohort membership is only an assumption. The temporal evaluator similarly inner-joins timestamps without cardinality validation (`my_work/scoring/temporal_eval.py:66`–`:68`) and silently drops invalid dates.

**Suggested fix:** validate unique transaction keys, identical labels and intended cohort membership; use validated merge cardinality; fail on unintended losses/duplication and log matched/unmatched/invalid counts. If a common intersection is intentionally used, label it explicitly and report exclusions and its fraud prevalence.

### E5 — Medium: leaderboard confidence intervals do not describe the displayed multi-seed mean

For multiple seed paths, the leaderboard averages AP/AUROC across all paths (`my_work/scoring/leaderboard.py:101`, `:104`) but bootstraps only `last_y,last_s` (`:92`–`:95`). It reports `n` and positives from that same last file (`:99`–`:100`). The interval therefore belongs to the last seed, not the adjacent mean across seeds.

The bootstrap itself samples individual transactions independently (`:51`–`:56`). In this project, multiple transactions belong to the same account and neighboring history windows overlap; naive row resampling may understate uncertainty. It also does not quantify the paired AP difference between two models, so comparing separate intervals does not demonstrate equivalence or superiority.

**Suggested fix:** label per-seed sampling intervals honestly, report seed variation separately, and compare models using paired resamples of a common cohort. Choose account/time blocks according to the deployment estimand and dependency structure. A claim of statistical equivalence needs a chosen practical equivalence margin and an appropriate interval for the difference.

### E6 — Medium: “statistically tied,” “only real effect,” and “saturated” exceed the evidence

The grids use one seed, 42 (`my_work/phase4a/configs/ft_decisions.yaml:16`; long-history `:22`; hierarchical `:13`). The standing document asserts a real embedding effect and noise-level mask/optimizer differences (`my_work/docs/my_current_standing.md:30`) and calls h10 and h16 statistically tied (`:79`). No repeated-seed results or confidence interval for these differences is supplied. The phase guide also declares Architecture A saturated (`my_work/docs/my_phase4a_guide.md:23`).

The eight h5 settings are different configurations, not four independent repetitions of the same embedding treatment; interactions and training randomness remain unresolved. A small observed gap does not prove equivalence. “Numerically close in this one-seed sweep” is supported; statistical significance, equivalence, and architectural saturation are not.

Also, transferring Architecture A's winning knobs to B is a sensible economical first comparison, but it does not find B's best knobs or prove the globally best architecture. Say “matched-knob comparison” until tuning budgets and seeds are adequate.

### E7 — Medium: calibration guidance confuses score ranking with probability calibration

The roadmap says downsampled models **must** be recalibrated before comparing AP (`my_work/docs/my_roadmap.md:39`). AP and AUROC depend on ordering, not whether a probability of 0.1 really means a 10% event rate. A strictly increasing prior correction or positive-slope sigmoid leaves ranking unchanged. The essential requirement for AP comparisons is evaluating the same untouched, naturally distributed cohort. Calibration matters for meaningful probability thresholds and expected loss.

Conversely, the calibrator documentation claims all isotonic/Platt calibration preserves ranking metrics (`my_work/scoring/calibrate_and_report.py:7`–`:12`, `:84`). Isotonic regression can collapse unequal scores into ties, changing AP/AUROC. The unconstrained logistic fit (`:49`–`:51`) can have a negative slope and reverse the ranking.

**Reproduced:** isotonic calibration fitted on a separate synthetic label pattern changes test AP 1.0 to 0.833333 and test AUROC 1.0 to 0.875. The script prints only original-score ranking and original-score alert-budget metrics (`:85`–`:86`, `:94`), so its outputs are raw-ranking metrics, not a verification of preserved calibrated ranking.

**Suggested fix:** report raw and calibrated ranking separately, keep raw ranking for top-budget alerts where intended, and document the difference. Fit calibration on a separate representative natural-rate calibration period or suitable out-of-fold predictions. Avoid estimating calibration on the final test labels.

### E8 — Medium: calibration and ensemble inputs can overlap validation with test

The calibration example passes `--val .../seq/part_4950.parquet` and `--test .../seq` (`my_work/scoring/calibrate_and_report.py:14`–`:18`). If the latter directory contains the former, the validation rows used to fit calibration are included in the test set. `_read` loads a ParquetDataset and nothing verifies disjointness (`:40`–`:41`, `:75`–`:82`). Ensemble input paths likewise have no disjoint-ID checks between fitting and evaluation sets (`my_work/scoring/ensemble.py:81`–`:86`).

This is an accepted-input/documented-example failure mode, not a claim that a historical run used overlapping data. The absence of prediction artifacts prevents checking historical overlap.

**Suggested fix:** use separate calibration/validation/test manifests and reject overlapping target transaction IDs. Keep any allowed prior historical context separate from scored target membership. Save the fitted calibrator, ensemble weights and their cohort provenance; the present scripts only print rounded weights or emit predictions, which is insufficient to reproduce the transformation later.

### E9 — Medium: temporal breakdown is useful monitoring, not an unseen temporal holdout

`temporal_eval.py` reads predictions already created, attaches times, and groups them (`my_work/scoring/temporal_eval.py:56`–`:84`). It never defines chronological train/calibration/test cutoffs, checks training timestamps, excludes already-used rows, or retrains a model. Running it on the repeatedly inspected evaluation set does not meet the standing document's pending temporal-holdout requirement (`my_work/docs/my_current_standing.md:154`).

Its drift claim is stronger than the diagnostic: AP depends on prevalence; a negative line fit to a few rounded bucket values does not establish model drift or a need to retrain (`my_work/scoring/temporal_eval.py:97`–`:107`). Single-class buckets disappear completely (`:76`–`:77`), and gaps between periods are collapsed into consecutive indices (`:99`).

**Suggested fix:** first define an untouched later target cohort with label-maturity rules and only historically available features. Separately show temporal metrics, all bucket counts/prevalence (including undefined metrics), actual date spacing and uncertainty. Treat a negative slope as a monitoring signal needing investigation.

### E10 — Medium: ablation summary can pair metrics from different checkpoints

The runner takes maximum validation AP and maximum validation AUROC separately (`my_work/phase4a/scripts/run_ablations.py:52`–`:65`, `:126`–`:127`; 4b same logic). Its evaluated checkpoint is selected on AP by the trainer (`my_work/phase4a/scripts/train_sequential.py:107`–`:122`). This is not wrong if the columns are explicitly independent maxima, but it is wrong to describe both values as that model checkpoint's validation performance.

The provided raw rows demonstrate the distinction. h10's best-AP checkpoint is epoch 6: AP 0.139942, AUROC **0.946059**, while the summary AUROC **0.948879** belongs to epoch 9. h16's best-AP checkpoint is epoch 8: AP 0.138256, AUROC **0.947182**, while summary AUROC **0.948226** belongs to epoch 9 (`my_work/docs/my_current_standing.md:58`, `:61`, `:75`, `:76`).

**Suggested fix:** report `selected_epoch`, `selected_valid_ap`, and `selected_valid_auroc` from one selected checkpoint; keep independent maxima in separately named optional columns.

### E11 — Medium: runner does not propagate failed experiment status to the shell

`run_ablations._run` returns failure codes, but the loop records them and continues (`my_work/phase4a/scripts/run_ablations.py:68`–`:78`, `:119`–`:151`; 4b same). Scoring return codes are not included in the CSV. Unless the Python runner itself raises, `run_all.sh` sees success even when training or scoring subprocesses failed; shell `set -e` (`my_work/run_all.sh:4`) cannot catch failures already swallowed by Python.

Additionally, training uses `common + overrides`, but scoring uses `overrides + common` (`run_ablations.py:114`, `:136`). A future axis that overrides a common key can train one configuration and score another. No such duplicate key collision was found in the three current canonical grids; this is a latent runner bug.

**Suggested fix:** use identical resolved configs for training/scoring; persist them; record training and scoring statuses; return a nonzero overall status when required jobs fail. Do not rank stale metrics from failed reruns.

## Lower-priority documentation and robustness corrections

- **Incorrect small arithmetic:** reported h5 values imply linear mean **0.132579**, not 0.132604, and bidirectional mean **0.1289745**, not 0.128987 (`my_work/docs/my_current_standing.md:25`–`:28`). Periodic 0.1243505, causal 0.127955, AdamW 0.128346 and RAdam 0.1285835 agree after rounding. The roadmap sampling formula evaluates to **1,051,145.333** non-fraud rows, not 1,051,813 (`my_work/docs/my_roadmap.md:35`). These do not change the qualitative experiment choice.
- **Stale progress documents:** roadmap still says h10/h16 “to run” (`my_work/docs/my_roadmap.md:98`–`:99`) despite the newer standing snapshot reporting completion. It references absent `my_work/run_remaining.ps1` (`:69`, `:109`). Keep a single dated experiment ledger.
- **Overbroad reproducibility claim:** `run_all.sh:2` says “reproduces every run,” but lines 9–18 invoke only the eight h5 finder runs, two flat long-history runs, and three hierarchical runs. It does not reproduce the Ya rerun, h5@100 confirmation, Phase 5 training, calibration, or final temporal evaluation. That is a **13-run Phase 4 sweep**, not the whole project history. Rerunning uses existing deterministic names and overwrites results/logs/checkpoints; archive before rerunning.
- **Mixed-scale table heading:** “10-file scale, all apples-to-apples” includes 100-file rows (`my_work/docs/my_current_standing.md:126`–`:136`). Training sizes are visible in the rows, but the heading overpromises. Validation/evaluation cohort, feature preprocessing and checkpoint-selection equivalence must also be verified, not inferred from similar scores.
- **Same fraud prevalence does not prove no shift or overlap:** `my_work/docs/my_roadmap.md:26` claims separate buckets and no overlap. Configs use different prefixes under the same raw-data bucket. Neither separate prefixes nor similar class rates prove disjoint IDs, time separation, feature stability or lack of leakage. This needs data-level evidence.
- **Gradient accumulation is sensible but not identical in every implementation:** effective batch size 4096 is consistent in the long-history configs. “Mathematically equivalent” (`my_work/docs/my_phase4a_guide.md:21`) assumes equal full microbatch sizes, equivalent loss normalization and identical optimizer-step timing. Short streaming tails, dropout/randomness, numerical precision and callback schedules can differ. Treat this as a practical comparable-batch recipe, not bitwise or unconditional mathematical identity.
- **SupCon sampler rationale overstates a hard requirement:** with batch 8192 and prevalence 0.155%, about 12.7 fraud examples are expected per independent full batch; the actual stream can be clustered. Two-view contrastive learning can also have a positive pair from the same example. Balanced sampling is a useful experiment, not a proven prerequisite merely from the global rate (`my_work/docs/my_phase5_guide.md:27`; contrastive config `:51`). Measure per-batch positives and representation behavior; do not treat the institutional 12% as a validated optimum for this loss.
- **Phase 5 comparison defaults are not matched:** SSL validation ends at 4950 (`my_work/phase5_pretrain/configs/ssl.yaml:25`–`:27`), SupCon at 4954 (`contrastive.yaml:24`–`:26`). Both default evaluation to only 8000. Standardize cohorts before comparing recipes. A from-scratch control should use the same encoder/head, features, data, training budget and final evaluation.
- **Alert-budget edge cases:** `_at_budget` and `_precision_at` do not validate budgets (`calibrate_and_report.py:54`–`:60`; `leaderboard.py:39`–`:42`). On four rows, a 0% budget still chooses one alert; 200% reports eight alerts and divides by eight although only four rows exist. Reject invalid fractions and define zero budget/rounding explicitly. The default positive fractions on large cohorts do not hit the 200% issue.
- **“Cost-sensitive” is currently alert-volume analysis:** calibration reports precision and recall at top fractions, but does not estimate false-positive cost, fraud amount recovered, expected loss or a deployable fixed threshold. These are useful operational metrics, but the label does not establish actual cost optimization.
- **Stale executable examples:** all four scoring module docstrings still use nonexistent `fraud/scripts/<scoring-file>.py` paths (`ensemble.py:9`, `leaderboard.py:9`, `temporal_eval.py:13`, `calibrate_and_report.py:15`). Correct paths start with `my_work/scoring/`. `my_work/tests/test_models.py:8` similarly should use `my_work/tests/test_models.py`.
- **Scalability:** scoring utilities eagerly load whole prediction sets into pandas. Ensembling repeatedly optimizes AP over 4,000 random weight samples; the leaderboard performs hundreds of full-row bootstrap sorts. Benchmark time/RAM before applying these settings to large bank cohorts. This is a scalability caveat, not a demonstrated failure at the reported 11-file size.

## What the existing model tests establish, and what they do not

`my_work/tests/test_models.py` contains ten tests; `test_supcon.py` contains three. The parent audit runs the available suite and records actual runtime outcomes separately; this section is a coverage review.

| Area | Existing assertion | Important remaining evidence |
| --- | --- | --- |
| Flat linear model | Output shape and backward pass (`test_models.py:61`) | Finite logits/loss and gradients for the intended parameters; a tiny learnable signal |
| Periodic embedding | Output shape (`:70`) | Numeric formula/reference correctness; gradients; checkpoint round trip |
| Causal attention | Output shape (`:75`) | Mask semantics and invariance to inputs that should be masked |
| Hierarchical model | Classifier/embedding shape (`:85`) | Backward flow; padded-history behavior; causal mode; temporal order signal |
| History padding | All test model masks are true (`:40`) | Outputs invariant to changes in masked padding; short/empty history finite |
| SSL reconstruction | Scalar loss and backward (`:102`) | Loss uses only masked fields as intended, finite empty-mask/one-field cases, no reconstruction shortcuts |
| SSL transfer | No exception and output shape (`:109`) | Exact transferred encoder tensor equality and allowed missing/unexpected key assertions |
| Optimizers | Creation and parameter groups (`:116`) | Actual parameter updates and unsupported/optional backend behavior |
| EMA | Swaps away from raw moved weights, restores raw (`:123`) | Exact expected average, accumulation/optimizer-step behavior, checkpoint loading and resume |
| SupCon loss | Scalar and some gradient exists (`test_supcon.py:31`) | Finite gradients, reference-formula comparison, same-class/one-positive/edge batches, mixed precision |
| SupCon transfer | Missing keys don't contain “projection” (`:47`) | That check does not prove any encoder tensor was loaded; assert encoder keys and tensor equality, inspect unexpected keys |

The model-test docstring claims coverage of TabM member averaging (`test_models.py:4`), but no TabM test or import exists; the roadmap says TabM was dropped. Tests use random roughly balanced labels, not the rare-event deployment distribution. That is fine for unit shape checks but does not validate rare-class learning, calibration, AP quality or full pipeline correctness. The test suite also does not exercise these four scoring utilities, cohort overlap, metric tie behavior, run manifest identity, or multi-seed intervals.

## Reconstruction evidence: fidelity is not runtime correctness

The newest evidence lives in `reconstruction/HANDOFF.md`, which explicitly supersedes older checkpoints. Do not use the top paragraph of `STATUS.md` or the old JSON totals as the latest project inventory.

- `HANDOFF.md:5`–`:17` records 68 Tests photos, two run_all photos and 44 distinct Current Standing photos reconciled to the six named outputs. The recorded status is **VISIBLE_SOURCE_VERIFIED** with separate artifact comparison and static parsing; tests/code were not executed during reconstruction.
- `HANDOFF.md:27` records 54 scoring photos reconciled to five scoring-package files, also source-verified and parsed without execution.
- `HANDOFF.md:43` records 121 Phase 5 photos reconciled to 17 source files. This does not validate training outcomes.
- `HANDOFF.md:75` still lists five normalized Markdown documents and three notebooks awaiting fidelity audits, the historical shell-header gap, seven historical `src/pytorch_tabular` source-origin gaps and expanded Phase 4 membership/fidelity reconciliation.
- `reconstruction/queue.json` explicitly says COMPLETE means visible contents recovered, **not runtime correctness or original-byte identity**. Its historical missing-file list should not be read as saying the present files are absent: those seven Python files now exist, but their source origin still needs reconciliation according to the handoff.

This audit reviewed current text and behavior; it did not re-inspect the original photos. A bug may be faithfully transcribed from the original laptop code. Evidence of a code bug does not establish a transcription mistake.

## Suggested decision sequence

1. Preserve the pod's actual resolved configs, logs, checkpoints and uniquely identified prediction outputs; reconcile the one-file versus 11-file evaluation discrepancy first.
2. Resolve correctness issues found by the full audit, especially history construction/model behavior, before spending more training time. Passing tensor-shape tests alone is insufficient.
3. Treat the existing repeatedly viewed evaluation set as a development benchmark. Select an untouched later, naturally distributed cohort for final confirmation, with label availability and historic feature availability established.
4. Compare flat and hierarchical models on identical target IDs, feature artifacts, train scale and validation splits. Include the existing single-transaction baseline and the internal XGBoost predictions on that same cohort when available. Keep validation-based selection.
5. Confirm promising settings over several seeds; report paired uncertainty, operational precision/recall at budget, counts/prevalence and resource cost. Use appropriate account/time grouping for uncertainty.
6. Evaluate Phase 5 against a matched from-scratch encoder control; then assess an explicit transfer into history models if that is the intended next phase. Pretraining success cannot be inferred from the current standings.
7. Fit and save calibration and any ensemble only on designated development/calibration data, then evaluate the frozen result once on the final holdout. Keep run metadata and thresholds reproducible.

## Executed local checks

`python CH/checks/evaluation_checks.py` passed. It extracts the actual pure helper definitions using Python AST, supplies synthetic NumPy/pandas inputs, and reproduces tied-rank metric distortion, silent join duplication/drop/label conflicts, isotonic tie changes and invalid alert-budget handling. It does not import the S3-dependent modules or access external data. Assertions intentionally verify the current undesirable behavior; this file is audit evidence, not a replacement for corrected production regression tests.

Additional static/arithmetic checks verified the marginal means, sampling formula, current config ranges, actual absence of `ablations/` and `run_remaining.ps1`, and the distinction between maximum AUROC and AUROC at the AP-selected epoch. No historical bank performance claim was independently verified.
