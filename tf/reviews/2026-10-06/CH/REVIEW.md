# CH independent review — current TF code and plan, October 6

**My latest update:** Both of my bank training scripts use their Phase 4A/4B sequence builders; CH has verified that GitHub still contains the old imports. I need to synchronize those files so the repository reproduces my bank setup. I have created h0 and plan to run it; there is no completed h0 result yet. CL also recorded that my corrected knob finder is still running and AdamW is my chosen fidelity baseline. The newer plan table remains provisional.

**Later peer update:** [Additional review and patch verification](PEER_UPDATE.md). Coverage now totals **63 unique changed files**. Application code is unchanged, but the latest default test command reports **31 passed, 2 failed** because proposal-only tests enter discovery. The initial **30 passed** result below remains scoped to the earlier application snapshot.

> Public first-person project copy prepared by CH on 2026-10-06. I keep my original findings and guides unchanged. CH performed the checks reported here. Source paths and commands are illustrative private-project aliases; code, evidence programs and the candidate patch remain in the private review record. Exact approved scientific details are retained.

**The project has improved, but the checked-in training path is not ready for trustworthy new comparisons yet.** The main blocker is precise: both training scripts still import the original faulty sequence builder. The new, corrected builders work when used. The model masking fixes and several scoring/training changes pass independent checks. Statistical superiority, production benefit and GPU readiness remain unestablished.

My review covers the October 6 source snapshot against the previous review; exact source identities remain in the private evidence record. CH reviewed **all 35 changed files** across six independent workstreams and checked the surrounding execution paths. This is a review of current application code, not an endorsement of the different candidate patches in CL's frozen folder. Original findings and application source remain unchanged.

## Current standing in different views

| View | Assessment | What supports it / what remains |
|---|---|---|
| Model implementation | **Improved; verified within CPU scope** | Flat and hierarchical causal/padding behavior passes expanded independent checks. Bank GPU kernels and resource limits remain untested. |
| Data entering training | **GitHub synchronization required** | GitHub trainers still select the old loader: ten synthetic targets produced eight distinct IDs. Bank trainers reportedly use the corrected phase-owned builders. |
| Train → checkpoint → score | **Works on small valid local fixtures** | Four actual Phase 4 entrypoint round trips completed; all 11 target IDs per case were retained. These used zero workers and do not override the four-worker failure. |
| Experiment records | **Partially repaired** | Train/score override order and distinct run-name outputs work. Repeated attempts overwrite evidence; failed attempts can inherit old metrics. |
| Statistical comparison | **Not yet established** | Historical cohorts differ. The original six grids omit h0; my newly created h0 control is pending execution. Repeated-seed results and corrected analysis utilities are still needed. |
| Updated plan | **Useful direction; needs corrections** | AdamW baseline and history axes match the intended downstream settings. Some claims about incorporated fixes, timing, architecture and executable paths exceed current source. |
| Packaging / original pipeline | **Still not reproducible from declared package alone** | Wheel build fails; a backend-only repair still omits my modules/config assets. Original streaming API and checkpoint-loading failures remain. |
| Phase 5 pretraining | **Mechanically runnable on fixtures; objectives still defective** | Small pretrain/fine-tune/score round trips work, but SupCon alignment and masked numeric field identity failures remain. |
| Business use | **Research challenger** | Offline scoring is useful. There is no matched production-model comparison, measured business lift, final temporal confirmation or live-serving validation. |

## What the updates actually changed

- New phase-owned sequence builders implement shared file shuffling before worker slicing.
- Flat and hierarchical models repair the CLS attention relay and the padded-query NaN case.
- Standard epoch-end EMA checkpoint timing now saves the evaluated weights.
- Ablation training and scoring use the same override precedence. Scoring receives the run name and preserves different run-name outputs.
- Main A/B evaluation defaults now cover files **8000–8010**. The historical scores have not thereby been recomputed.
- The executable downstream plan is **AdamW**, A **h10/h15/h20**, B **h5/h10**, and B long **h15/h20**, plus time-gap and capacity controls. The h5 finder still intentionally compares AdamW/RAdam.
- Six sweep files expand to **20 training configurations**, **15 scoring commands**, nominal effective batch **4096**, and one seed **42**. None contains h0.

The long B file remains `my_work/phase4b/configs/ft_decisions_hier_h16.yaml`; its contents now test h15/h20. The later filename mentioned in the chat-derived overview is not present. Use the checked-in filename until an intentional rename occurs.

## Findings to act on first

### 1. The corrected loader is not connected to training — high priority

Both `my_work/phase4a/scripts/train_sequential.py:27` and `my_work/phase4b/scripts/train_sequential.py:27` import `fraud.data.sequence_builder.SequenceBuilder`. The new class is only selected in the scorers. With four workers, the actual training loader factories emitted ten rows from ten synthetic source targets, but omitted IDs 7 and 9 and duplicated 3 and 6. Counting rows alone does not detect the problem.

The phase-owned implementations passed 24 direct file-coverage controls. Replacing only each trainer's imported class in an explicitly temporary test passed another eight controls through the actual factory. A two-import candidate patch is prepared in the private review record; the [fix queue](NEXT_STEPS.md) describes it and passes `git apply --check`; it is **not applied**. It does not change Ya's original source or silently alter timestamp/label contracts.

Acceptance: both real training entrypoints select their phase-owned builder and preserve the exact target-ID multiset over multiple workers, unequal shard sizes and multiple epochs. Add that entrypoint-level regression; current sequence tests still import the old class and mainly use one file. Detailed [data review](details/DATA.md).

### 2. Repeated and failed runs can corrupt the experiment record — high priority for reliable comparisons

Both runners reuse `output_root/run_name`, truncate logs, and reuse checkpoint, metric and prediction names on another invocation. A failed rerun can read the previous run's metric CSV and rank stale numbers: both probes produced `train_rc=1` alongside old AP **0.9**. That is a synthetic lifecycle example, not a bank score. Scoring failure lacks an explicit result status, and the outer runner does not reliably propagate required-stage failure.

Distinct names within the currently checked-in grids are preserved; no collision among those six grids was established. The remaining problem is repeated attempts or cloned grids/seeds that share labels. Add immutable experiment/attempt identity, config/artifact/checkpoint fingerprints, explicit train/score status, and successful-run-only ranking. [Training review](details/TRAINING.md).

### 3. Separate historical results, current defaults and the newer reported sweep

The attributed October 5 answer resolves the old sweep: evaluation used **part_8000, 290 fraud cases**. Ya's **0.0940** reference used **8000–8010, 3,818 fraud cases**. The old h5/h10/h16 evaluation APs **0.0893 / 0.0902 / 0.1171** cannot establish which model beats that eleven-file reference.

The newer October 6 summary reports linear **0.1260** versus periodic **0.1202**, and AdamW **0.1189** versus RAdam **0.1273**. Those are a separate reported sweep, not the old result table. Its raw eight-row results, exact source/config/artifacts and predictions are unavailable in this checkout. The plan's description of a “clean” rerun is not established by code whose training imports still bypass the fix. My later answer resolves the import question: my bank copies use the phase-owned builders. Completed rerun provenance is still pending; the table is provisional.

Keep AdamW as the stated fidelity baseline; assess RAdam separately if useful. Neither a small observed gap nor “won four pairs at one seed” establishes statistical equivalence or a durable optimizer advantage. [Statistical review](details/STATISTICS.md).

### 4. Repair analysis utilities before using their summaries to choose a winner

| Tool issue | Reproduced consequence | Required correction |
|---|---|---|
| Ensemble double-argsort tie ranks | Equal scores with true AP/AUROC **0.5/0.5** become **1.0/1.0**, depending on row order | Average ranks for ties; permutation-invariance check; preserve raw individual metrics |
| Unchecked joins/cohorts | Targets silently drop or duplicate; label disagreement is ignored; unequal cohorts are ranked together | Unique IDs, validated one-to-one joins, matching target/label manifests and explicit exclusion counts |
| Leaderboard interval from last seed | Two seed AUROCs 1 and 0 display mean **0.5** with CI **[0,0]** | Define the estimand; separate seed variation from sampling uncertainty; implement paired, appropriately blocked comparisons |
| Temporal grouping mistaken for holdout validation | Existing predictions are regrouped; invalid timestamps disappear and single-class periods can be omitted | Define a real time split, mature labels and as-of inputs; show all periods/counts and undefined metrics |
| Calibration/reporting contract | Isotonic ties alter ranking; calibrator is not persisted; overlap and invalid alert budgets are unchecked | Separate selection/calibration/test roles, save the fitted object, validate budgets, report raw ranking and calibrated probability metrics separately |

These are small executable counterexamples, not claims that these exact errors occurred in bank reporting. They show the tools cannot yet support the stronger conclusions requested. Details and line references are in the statistical review.

### 5. Add the experiments that identify the source of improvement

The original six checked-in sweep files omit same-architecture **h0 versus h5** and use only seed 42. I have now created h0 and plan to run it; completed results are pending. Comparing a custom history encoder to the library single-row model also changes architecture/training, so the difference does not isolate history. Run matched controls, then repeat shortlisted settings with **3–5 seeds** and paired account/time-appropriate uncertainty. Three seeds are a starting budget, not a guarantee of adequate power.

Freeze the evaluation population and checkpoint-selection rule across comparisons. If validation expands when training scales, re-evaluate every comparator on the same expanded cohort. Overlapping windows affect uncertainty even though AP for fixed predictions is order-invariant. Treat files 8000–8010 as a development benchmark that has already been consulted; reserve a later untouched cohort with mature labels for final claims.

### 6. Fix the new h10 configuration mismatch before scheduling those controls

Time-gap and capacity configs set h10 but inherit **batch4096 × accumulation1**. The long-history h10 setting uses **1024 × 4**, following the documented memory problem. The large capacity row also increases width/depth. The configuration difference is confirmed; a current H200 OOM was not tested here. Use a common measured-safe microbatch schedule, then profile the larger model. A matching effective batch does not by itself guarantee equal compute or identical optimization.

### 7. Retain the repaired attention code and strengthen its regression tests

The current models passed 288 behavioral/gradient cases, 48 explicit mask checks and 16 tests with the actual 30-feature-token shape. However, restoring the old CLS relay in memory while retaining padding safety still passes all three committed causal tests: they use one layer and only check shape/finiteness. Add a two/four-layer target-perturbation test asserting unchanged earlier hidden states, plus padding-value/gradient invariance.

Old checkpoints load, and identical bidirectional weights give identical outputs. Corrected causal masks intentionally change historical causal predictions, so rescoring an old checkpoint is a newly defined experiment. Record code revision as well as seed: the hierarchical constructor also changes initialization order. [Model review](details/MODELS.md).

## Important conditional issues, without blocking unrelated work

- **EMA is off by default.** Epoch-end evaluation/checkpoint alignment is repaired. When enabled, microbatch-based updating changes smoothing under accumulation; exact resume state is not saved; within-epoch validation resumes on EMA weights. Current entrypoints do not expose resume or within-epoch validation. Fix/test those capabilities before using them; they do not block ordinary EMA-off A/B controls.
- **Phase 5 objectives remain wrong.** SupCon positives are misaligned with flattened views, and masked numeric fields lose column identity. Small successful training/scoring round trips do not validate those objectives. Fix each before its respective pretraining experiment. A missing/empty pretrained state must fail instead of being accepted as pretrained. [Runtime/pretraining review](details/RUNTIME_AND_PRETRAINING.md).
- **Package/environment reproducibility remains open.** The declared build backend fails; correcting that line alone still leaves my modules and configuration resources out of the wheel. Original streaming construction and modern-Torch checkpoint loading also fail in this environment. A documented, verified bank environment may differ. Freeze and prove the environment used for the selected path; do not infer that passing tests in the supplemented audit environment means a clean install works.
- **Input contracts need bank evidence.** Raw non-ISO timestamps can sort incorrectly; invalid timestamps/labels and infinite numbers are not rejected consistently. Equal-time ID ordering differs from the plan's strict `time < target_time` promise. Validate actual schema/arrival order, account containment within files, target uniqueness, feature availability, label maturity and My-versus-Ya artifact identity. These conditional reproductions do not prove the bank data contain the bad cases.
- **Multi-GPU is a future capability.** Local worker partitioning does not partition distributed ranks; full-cohort metrics are not gathered across ranks. The current single-GPU scope needs explicit treatment before scaling.
- **Architecture wording matters.** A has CLS plus feature blocks and transaction-position embeddings, but no explicit SEP token. B builds a fresh transaction encoder; it does not load a trained single-transaction backbone. The default B 2+4 blocks versus A 4 is not a parameter-matched comparison. The periodic implementation is a shared-projection variant, not a general verdict on the paper's PLR family.

## What to do next, in order

1. Synchronize both GitHub trainer imports with the reported corrected bank copies. Add entrypoint coverage and causal-invariance regression checks. Preserve every attempt and reject stale/failed metrics.
2. Pin code, environment, ordered feature/preprocessing hashes, source/target manifests and evaluation identities. Confirm the timing/label/account contracts on the bank data. Use EMA off until its optional lifecycle issues are addressed.
3. Run a small smoke on the exact bank GPU/software stack, including checkpoint reload/scoring, finite gradients, exact target counts, peak RAM/VRAM and throughput. Use the known safe h10 settings for time-gap/capacity tests.
4. Run same-recipe A h0/h5 controls, then repeated-seed confirmation. Continue Ya's A/B h5/10/15/20 comparison and time-gap/capacity studies with explicit parameter/compute/resource reporting.
5. Repair comparison utilities, retain identified predictions, and compute paired uncertainty. Keep the old and newer sweep summaries separate until their run provenance is established.
6. Compare finalists with the bank's existing model on identical targets: AP, prevalence/counts, precision/recall at the actual review budget, fraud-value capture where appropriate, calibration and latency/cost. Use untouched later data for final confirmation.
7. Scale leading candidates to 100/200/500 files when the benefit and cost warrant it. Add pretraining after objective-specific fixes and matched scratch controls. Consider automated search after the trusted baseline/proxy protocol exists; previous evaluation looks cannot be erased by hiding a cohort from an agent.

The detailed [plan review](details/PLAN_AND_DOCUMENTS.md) covers every changed document/config, exact operational-reference errors, the autoresearch proposal and the proposed experiment sequence. No unrelated Phase 5 or disabled-EMA repair is a prerequisite to planning or executing a valid corrected supervised comparison.

## What CH actually verified

- **35/35 changed files accounted for** in the [coverage register](COVERAGE.md).
- **30 repository tests passed**, 113 warnings, in the existing CPU audit environment.
- **183 Python files parsed**, including prior audit/evidence scripts. Critical Ruff checks passed for application/vendor code.
- Models: 288 causal/padding/gradient cases, 48 explicit masks, 16 full-feature-shape cases, checkpoint compatibility and a regression-test mutation check.
- Data: actual-factory defect reproduction, 24 corrected-loader controls, eight proposed-import-substitution controls, plus history/metadata/schema boundary cases.
- Training: 14 real Lightning lifecycle fits; all 35 generated commands across the six grids; rerun/failure tests. Assertions reproducing a defect are evidence of the defect, not passing product behavior.
- Four actual Phase 4 train → checkpoint → score round trips, using local synthetic Parquet, zero workers, small CPU models, h0/history, time-gap on/off, nondefault class weights and one EMA/accumulation case. Every run retained all 11 evaluation IDs and correct labels with finite scores.
- Actual scorer/leaderboard/temporal exercises and unchanged analysis-utility counterexamples.
- Small SSL/SupCon pretrain → fine-tune → score round trips and embedding export mechanics, alongside independent objective counterexamples and isolated packaging probes.

Evidence uses Python 3.12.14, Torch 2.14.1+cpu, PyTorch Lightning 2.6.6, NumPy 2.5.3, pandas 2.3.3, scikit-learn 1.9.1 and PyArrow 25.0.1. It does not establish compatibility with every permitted version or the bank's installed versions. [Evidence index](README.md).

No bank datasets, live jobs, private checkpoints/predictions, original photographs or H200 execution were available to this review. AP figures are documented or attributed observations, not recomputed bank performance. Completed current bank run results remain pending. The trainer-import mismatch is reported; other source differences remain unverified. This is strong evidence for specific repairs and remaining defects; it is not a claim that every possible runtime path or statistical assumption is correct.
