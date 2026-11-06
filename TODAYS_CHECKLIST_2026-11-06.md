# Today's checklist — 6 November 2026

**My goal:** finish a clear first fraud-model deliverable: one selected feature set, a reproducible trained model, fair comparisons, and evidence of whether its embeddings help a downstream tree model.

**Today's execution target:** finalize the inputs and data roles, prepare the reduced commands, run the 10-file feature pilot and the current-model embedding pilot, and prepare the 100-file comparison. Training and confirmation runs finish according to their measured runtime; this document does not promise that all of them finish today.

**Status:** the unchecked items below are work to complete. This is a detailed checklist, not a new runnable package or a claim that the current pod was inspected. Completed items are identified separately. The earlier ZIPs contain a broader experiment program; their full queues do not automatically implement this reduced sequence.

The detailed boxes describe the steps inside two pilots and the follow-on comparison. They are not a request for a separate experiment, review or owner decision for every box. Engineering and Copilot should resolve routine implementation details from the working files.

## 1. What is already done

These are previously reported outcomes to retain and reuse when compatible with the next comparison.

- [x] The chosen model was rerun after project reorganization.
- [x] The light learning-rate study was completed; the chosen model retains learning rate `3e-4`.
- [x] Month-in and no-month models were trained and scored on the previously reported populations.
- [x] Broad feature analysis across the available training files was reported complete.
- [x] Earlier feature, workbench, checking and tuning tools were published.
- [x] A separate sinusoidal time-encoding experiment was published. Publication does not mean it was run or improved the model.

There is no blanket requirement to repeat these jobs. A repeat needs a concrete reason, such as changed preprocessing, corrected target labels, incompatible inputs or a genuinely unmatched control.

## 2. Who does what

| Role | Responsibility | Concrete output |
|---|---|---|
| Project owner | Confirm the feature choices, resolve business meanings that the available files cannot answer, and start pod workloads | Final feature decision and actual run outputs |
| Engineering | Implement the reduced workflow, configs, export and comparison tools; check affected behavior | One coherent set of files with verbatim commands |
| Copilot on the work laptop/pod | Map actual paths and environments, adapt the delivered files, verify imports and local execution, summarize outputs | A short local setup note and completed run/result table |

This document identifies required work; it does not claim that another contributor has accepted a new assignment. Work can proceed in parallel where files and outputs do not overlap.

## 3. Today's ordered checklist

### 1. Locate the working setup once

- [ ] Identify the active project root and Python environment.
- [ ] Identify the chosen model's current config, checkpoint, fitted preprocessing and feature list.
- [ ] Locate training, validation and evaluation files, existing scores and the output directory for this work.
- [ ] Identify any active jobs and their output folders before choosing new destinations.
- [ ] Record actual locations in the local setup note. Use those locations in the commands supplied below.
- [ ] Confirm whether one or two workbenches are currently available. One worker remains sufficient for the workflow.

**Done when:** each required input has a known local location and each new job has its own output destination. Another project reorganization is not needed to begin this study.

### 2. Confirm the second feature set

- [ ] Reuse the feature analysis already completed; reconcile it with the current exclusions and source definitions.
- [ ] Create one final table with: feature name, source or formula, data type, availability at prediction time, missing-value treatment, feature family, and include/exclude decision.
- [ ] Distinguish a field excluded because it is unavailable at decision time from a field omitted because its incremental value is uncertain.
- [ ] For engineered features, specify the entity key, lookback window, units and exact rule for including earlier events.
- [ ] Retain the strict earlier-second history rule. Same-second events are not ordered by an unverified sequence field.
- [ ] Include the chosen eligible features together for the main candidate. Related features may be useful jointly even when single-feature importance is low.
- [ ] Resolve candidate families from the earlier proposals: additional transaction/profile fields, merchant/category/country novelty, earlier decline-reason counts, and profile-event recency. Include only the families selected for this deliverable.
- [ ] Leave unresolved optional definitions out of this candidate and record why. Their resolution should not create an endless feature queue.
- [ ] Record the final feature count; do not reuse an old proposed count as if it were confirmed.

**Done when:** there is one exact current-feature control list and one exact expanded-feature candidate list, with reproducible definitions.

### 3. Resolve the supervised target and evaluation population

- [ ] Inspect the meaning of approved, declined, advice and verification-only rows in the local definitions and labels.
- [ ] Distinguish observed legitimate transactions from transactions whose fraud outcome is unobserved. An absent fraud tag alone does not settle that distinction.
- [ ] Specify which rows have usable supervised targets and which contribute to the training loss.
- [ ] Specify the eligible evaluation population and the denominators of each metric.
- [ ] Where a row is genuinely unlabeled, omit its target from supervised loss and labeled metrics; retain it as earlier history when its fields are available and relevant.
- [ ] Apply the same target policy and evaluation eligibility to both sides of a feature comparison.
- [ ] If the policy changes from the existing model, make a fresh matched baseline. Label that change separately from the feature-set effect.
- [ ] Record any unresolved label-maturity limitation. Lower observed fraud frequency is not proof that a population is intrinsically safer.

**Done when:** the training and evaluation code implement an explicit target definition. A decline-masking experiment is a separate comparison if the question remains a modeling choice rather than a demonstrated label correction.

### 4. Freeze the first comparison recipe

- [ ] Use the chosen hierarchical architecture B with all-position training and history setting `H20`.
- [ ] Train with chunks; validate and score with target-ending sliding windows.
- [ ] Retain the established strictly earlier timestamp behavior and row cap.
- [ ] Start at seed `42` and learning rate `3e-4`, with the current optimizer, model size, stopping rule and other training settings.
- [ ] Keep the current month-in recipe as the control. Read the existing month-in/no-month comparison on the designated June-only development rows; rescore saved checkpoints if those predictions are absent.
- [ ] Record any material June-specific concern. The previously reported mixed-month result does not settle the June-only question.
- [ ] Keep cyclical month and sinusoidal time encoding as distinct optional fits; neither is silently added to the feature pilot.
- [ ] Use the same training budget, target policy, effective-batch definition and checkpoint-selection rule for each matched pair. Record any unavoidable difference.
- [ ] If wider inputs require a smaller micro-batch, record accumulation and the number/weight of supervised targets per optimizer update. Matching the product of micro-batch size and accumulation alone does not prove equivalent loss normalization for variable-length chunks.

**Done when:** the feature comparison has a specified control. A change in architecture, labeling, sampling or optimizer is not accidentally bundled into the claimed feature effect.

### 5. Fix the data assignments

- [ ] Record the exact 10-file training list and the intended 100-file training list.
- [ ] Retain the existing validation, development, downstream and final-test roles in the population table below.
- [ ] Check actual transaction/account overlap for the relevant assignments. Different file names alone do not prove separation.
- [ ] Keep downstream training files out of encoder fitting. Keep final evaluation out of all fitting and selection.
- [ ] Choose the larger training set by a reproducible account-bucket rule, rather than selecting files because their evaluation results look favorable.
- [ ] Record that the existing file layout changes which accounts are included; files cover the same broad training period. More recent file numbers do not imply more recent transactions.
- [ ] Fit preprocessing only on the designated training population for that model. Completing exploratory analysis over a much larger file collection does not automatically make that collection the fitting population.

**Done when:** every job has an explicit file list, date filter, target policy and role.

### 6. Prepare the reduced runnable workflow

- [ ] Inspect the supplied tools and reuse their applicable functions rather than duplicate complete pipelines.
- [ ] Replace the old broad experiment queue with an explicit list of the selected jobs for this checklist.
- [ ] Supply configs for the 10-file current-feature control and expanded-feature candidate, plus their 100-file equivalents.
- [ ] Supply preprocessing, training, scoring, export, tree-fit and paired-comparison commands.
- [ ] Check imports, feature names, categorical cardinalities, model input width, artifact paths and output locations for the affected code.
- [ ] Provide one command sheet containing verbatim local commands, expected outputs, dependencies and restart behavior.
- [ ] Make a normal single-worker queue work. A second workbench may run an independent job using a separate output folder; this does not automatically enable distributed training of one model.
- [ ] Reuse passing evidence for unchanged code and run proportionate functional checks for changed behavior.

**Done when:** Copilot can follow one sequence without guessing commands or selecting among historical queues. No new command is implied by a planned script name in this document.

### 7. Refit preprocessing and check the entire path

- [ ] Fit the expanded feature set's vocabularies, missing-value handling and numerical transforms on its assigned training data.
- [ ] Keep the fitted artifacts associated with the exact model that uses them.
- [ ] Determine whether fitting changes the treatment of shared features. If it does, train a matched control with the same shared-feature fitting rules.
- [ ] Check feature order, categorical ranges, unknown-category handling and transformed values.
- [ ] Run a small end-to-end check: prepare input, train briefly, save, reload, score and export.
- [ ] Check usable targets, finite losses/scores/vectors, consistent keys and expected output dimensions.
- [ ] During that run, record elapsed time, GPU utilization, memory and time spent waiting on data.
- [ ] Build a local cache or pre-chunked copy only if measured loading costs justify it and the actual disk capacity supports it.

**Done when:** the actual local train-to-score/export path works. This short check establishes functionality, not model quality.

### 8. Run the 10-file feature pilot

- [ ] Reuse the existing current-feature run only if the compatibility conditions in the run matrix below hold.
- [ ] Otherwise train the matched current-feature control.
- [ ] Train the confirmed expanded-feature candidate on the same 10 files.
- [ ] Select checkpoints using the designated validation population and the existing stopping rule.
- [ ] Score both models on the same permitted development targets.
- [ ] Produce the paired result table and required uncertainty estimates.
- [ ] Keep the better-supported recipe or retain the control when no gain is supported. A retained baseline is a valid completed result.
- [ ] Use grouped permutation on a trained candidate if interpretation is needed. Reserve additional group-removal refits for a concrete ambiguity or cost decision.

**Done when:** the expanded set has a documented comparison against a valid control. There is no requirement to train one model per feature or keep searching until a positive number appears.

### 9. Run the current-model embedding pilot in parallel

- [ ] Freeze the existing chosen checkpoint and its preprocessing for this pilot.
- [ ] Export its fraud score and final per-transaction representation for downstream training and development rows.
- [ ] Use the representation immediately before the fraud head: the current model has 48 dimensions. Tokenizer-only feature vectors are a different representation.
- [ ] Fit the three matched tree arms in the experiment details below.
- [ ] Read vector-plus-score against score-only to test whether the vector adds information beyond the scalar prediction.
- [ ] Save the exported table for reuse. Repeated tree experiments do not require repeatedly scoring the encoder when the checkpoint, inputs and target population are unchanged.

**Done when:** the current encoder has a measured downstream result. The pilot does not need to wait for the expanded-feature or 100-file transformer.

### 10. Prepare the actual incumbent and foundation-model comparisons

- [ ] Locate the actual model scores and transaction IDs already reported available by the owner.
- [ ] Join scores to common eligible target transactions and reconcile duplicated IDs, missing coverage, labels and time filters.
- [ ] Use the same alert-budget and metric definitions across models.
- [ ] Keep the benchmark comparison distinct from the downstream training experiment and the untouched final test.
- [ ] Determine whether suitable incumbent scores exist for downstream training as well as benchmark evaluation.
- [ ] If suitable training scores exist, prepare the incumbent-offset comparison described below. If only benchmark scores exist, complete the benchmark alignment and record the specific missing downstream input.
- [ ] Treat foundation-model scores and foundation-model embeddings as different inputs. Having one does not establish possession of the other.

**Done when:** the comparison uses the real existing models on matched rows. The small XGBoost consumer is not presented as a replica of the incumbent.

### 11. Decide what proceeds to 100 files

- [ ] Review the feature pilot using the existing decision rule and development populations.
- [ ] Complete seed confirmation only where the agreed rule requires it. A small single-seed gain remains preliminary.
- [ ] Fix the selected feature recipe and any justified implementation corrections before the larger fit.
- [ ] Prepare the 100-file current-feature control and selected expanded-feature candidate with matched fitting rules.
- [ ] Estimate the remaining runtime from actual pilot measurements, including validation, scoring and export.
- [ ] Run the larger jobs when the pilot dependencies are complete. Keep one-worker operation available if the second workbench disappears.

**Done when:** the larger comparison is fully specified and runnable. The first deliverable stops at the agreed 100-file study; a 500-file run is a later decision.

### 12. Write the daily result and handoff

- [ ] Mark each job as not started, running, completed, failed or blocked, using actual outputs.
- [ ] Record completed changes, input locations, config names, selected checkpoints, artifact locations, scores and outstanding dependencies.
- [ ] Include verbatim commands for the next unfinished job and the expected output locations.
- [ ] Separate measured results from hypotheses and proposed experiments.
- [ ] Update the small result table below. Keep detailed logs beside their run outputs.
- [ ] List a concrete next action for every blocked item; do not leave a generic “waiting” status.

**Done when:** one short status note explains what ran, what it showed, what remains and exactly what to run next.

## 4. Exact experiment and population details

These are planned experiment roles. Row counts, account overlap, cohort eligibility, and compute outcomes still need checking; this document does not report completed fits or executed checks.

### 1. Run the minimal transformer comparison

Check whether more training data or expanded features improve the encoder under the same training setup.

| Feature set | Training files | Initial fit | Reuse condition |
|---|---:|---|---|
| Current 23 fields | 1–10 | Seed 42 | Reuse the existing 10-file baseline only if preprocessing, labels, and hyperparameters match this comparison. |
| Current 23 fields | 1–100 | Seed 42 | Fit unless an equivalent run already exists. |
| Owner-confirmed expanded features | 1–10 | Seed 42 | Fit with the confirmed feature list. |
| Owner-confirmed expanded features | 1–100 | Seed 42 | Fit with that same expanded feature list. |

- [ ] Confirm the expanded feature list and availability at prediction time before starting those rows. Record any unavailable fields explicitly.
- [ ] Use H20 hierarchical all-position chunk training, with sliding validation and scoring. Set learning rate to `3e-4`; make no other hyperparameter changes in this matrix.
- [ ] Keep label construction, preprocessing fitting procedures, epoch validation, and checkpoint selection consistent across matched rows. Refit the transforms on the designated training population at each size; the fitted values can change.
- [ ] Compare any reusable baseline against these settings. A similar run name or matching seed is insufficient evidence of equivalence.
- [ ] Start with up to four seed-42 fits. Reuse may reduce the number of new fits. Confirm with seeds 43 and 44 only when the existing agreed rule calls for them; do not promise a fixed total run count.
- [ ] Record completed, failed, and pending runs separately, including the actual settings and any missing results.

### 2. Run the current frozen-encoder pilot

Check whether the current frozen encoder's score and representation improve the tree using those fields. Keep this pilot separate from final transformer selection.

| Pilot arm | Tree inputs | Question answered |
|---|---|---|
| R | Common approved raw/engineered features | What does the tree using those fields achieve alone? |
| R + score | R plus the frozen encoder's fraud score | Does the encoder's scalar prediction add value? |
| R + score + 48-vector | R, that score, and the 48-dimensional representation | Does the representation add information beyond the score? |

- [ ] Extract the actual 48-dimensional representation immediately before the fraud head. Feature-tokenizer output alone does not satisfy this requirement.
- [ ] Use common downstream rows, raw/engineered tree inputs, preprocessing, and tuning budgets across all three arms. Check joins and missing outputs before comparing metrics.
- [ ] Use tree-internal validation for tree fitting and tuning. Do not use the downstream selection cohort as the tree's early-stopping set.
- [ ] Keep the encoder frozen throughout the pilot. Record which encoder supplied both the score and vector.
- [ ] Reproduce the comparison later with the selected encoder. Reuse R only if its rows, labels, raw/engineered features, preprocessing, tuning budget, and fitting settings are identical; otherwise refit it.
- [ ] If an arm loses rows because an encoder output is unavailable, report that loss and compare on common eligible rows.

### 3. Keep dataset roles explicit

| Dataset role | Files and extract | Intended use |
|---|---:|---|
| Transformer training, small | Training `1–10` | Small-data encoder fits. |
| Transformer training, large | Training `1–100` | Large-data encoder fits. |
| Transformer epoch validation | Training `4950–4954` | Epoch and checkpoint validation. |
| DS_TRAIN | Training `4001–4010` | Downstream tree fitting, with tree-internal validation. |
| DS_SELECT | Evaluation `8217–8227` | Downstream development comparison on June targets, with eligible strictly earlier May/June history. |
| RANK_A | Evaluation `8228–8243` | June-only feature importance and exploratory screening. |
| RANK_B | Evaluation `8244–8259` | June-only confirmatory development comparison under the existing decision rule. |
| SELECT | Evaluation `8206–8216` | The separate June development consistency check. |
| Test1 | Evaluation `8000–8010` | Reporting cohort; label full-file and June-only readings separately. |
| FINAL | Evaluation `8260–8277` | June-target final evaluation after the model and decision rules are frozen. |

- [ ] Preserve the distinction between DS_SELECT and SELECT in filenames, tables, and discussion. Similar names do not make them interchangeable.
- [ ] For June targets, eligible history can include May and earlier June transactions within the configured window. Exclude the target and any same-second or later transactions from its history; check temporal alignment and target-label isolation.
- [ ] Check account overlap directly before claiming account-disjoint cohorts. Different file ranges alone do not establish account disjointness.
- [ ] Keep FINAL unused for model choice, tuning, and threshold decisions until the planned freeze.
- [ ] Treat BENCH as a separate eligible, matched comparison between the incumbent model and payment foundation model. BENCH overlaps Test1, so it is not an independent final holdout.

### 4. Make result tables comparable

Use these results columns. Populate measured values only after the run and evaluation complete.

| Column | What to record |
|---|---|
| Model / arm | Feature set, training size, and downstream input combination. |
| Encoder / seed | Encoder configuration and fit seed; identify reused runs. |
| Evaluation cohort | Exact dataset role and common eligible population. |
| Transactions / fraud count | Evaluated transaction count and positive-label count. |
| Coverage | Eligible and scored counts, fraction covered, and exclusions. |
| UDR at 0.6% | Fraction of all fraud caught among the top-scoring 0.6% of all evaluated transactions. |
| AP / AUROC | Average precision and area under the ROC curve on the same rows. |
| Paired difference / confidence interval | Difference against the stated comparator using paired account bootstrap. |
| Seed variation | Variation across completed confirmation seeds, reported separately. |
| Status / notes | Pending, completed, failed, or reused, with relevant limitations. |

- [ ] Apply the same transaction-level ranking and existing tie rule when calculating UDR. Its denominator is all fraud in the evaluated population.
- [ ] Bootstrap accounts in pairs across compared models, retaining their transactions together. Keep sampling uncertainty distinct from training-seed variation.
- [ ] Use the existing agreed rule for statistical decisions and seed confirmation. The recorded paired-account procedure uses 2,000 draws, seed `20261102`, and a 90% interval; retain its current implementation and account for seed variation separately. Recompute the alert-budget ranking within each resample. Do not choose a new confidence level, stopping rule or adoption threshold after seeing the result.

### 5. Check the two additional comparisons before fitting

- [ ] For a real incumbent-model offset, obtain properly aligned, out-of-sample incumbent scores for the downstream training rows, plus aligned scores for evaluation. Confirm the required raw-margin or logit meaning and the fitting interface's offset semantics.
- [ ] Do not automatically convert an arbitrary incumbent score into a logit. Establish its meaning first; a score supplied as an ordinary input feature is not the same experiment as an offset.
- [ ] Before fitting with payment foundation model embeddings, verify that the downstream fitting cohorts were not included in that model's training extract. Record unresolved provenance as unresolved.
- [ ] For BENCH, verify eligible matched rows and report coverage alongside performance. Keep any broader-population results clearly identified.
- [ ] Close the day with the completed comparisons, reusable artifacts, missing evidence, and next action. Leave unverified rows and compute outcomes marked as pending.

## 5. What finishes the first deliverable

- [ ] Complete the matched 100-file comparison and any required confirmation runs.
- [ ] Choose one feature list, preprocessing fit, model recipe and checkpoint using development evidence.
- [ ] Repeat the downstream comparison with the selected encoder. Keep the tree baseline inputs and downstream populations fixed when measuring the encoder's changed usefulness.
- [ ] Freeze the full recipe, including the downstream model if it is part of the final candidate, before reading final results.
- [ ] Run the reserved final evaluation and report all planned metrics and material limitations.
- [ ] Report the actual incumbent/foundation-model benchmark separately; it has a different role and can overlap previously examined populations.
- [ ] Keep the runnable code/configs, preprocessing, checkpoint, exported features, results and exact commands together in the working project.
- [ ] Provide a simple embedding table and a working retrieval example. A row should identify the transaction, account where needed for lookup, prediction time, encoder version, fraud score, history depth and vector columns.
- [ ] For any as-of retrieval, select only information available by the requested time. Do not mix vectors from different encoder versions in one consumer without a separately tested compatibility method.
- [ ] Verify that the saved table and lookup return the expected transactions and vector dimensions.
- [ ] If storing the table in object storage is part of the selected delivery, use the configured project location and test a read-back. A new online feature-store platform is not required for this first result.

**Completion means:** a reproducible model and an honest answer to both questions: does the selected recipe improve fraud detection, and do its embeddings improve the tested downstream consumer? It does not require a positive result or establish live production readiness.

## 6. Research proposals retained outside today's required queue

These ideas are recorded so the narrower delivery does not lose them. They do not automatically add fits to today's queue.

| Area | Earlier or newly discussed work | When to revisit |
|---|---|---|
| Feature interactions | TFM-derived group, incremental incumbent-style group, combined eligible set, family-removal fits | Main candidate is tested now; extra group fits only when their answer affects selection or cost |
| Feature attribution | Grouped permutation, integrated gradients on learned embeddings, attention summaries | Explain a fitted model; attribution alone does not prove incremental generalization |
| Calendar and time | June-only interpretation; categorical/no-month/cyclical month; sinusoidal time-gap encoding | Existing June reading now; new encoding fits after the first deliverable unless a specific result changes priorities |
| Target treatment | Decline/advice masking while preserving history; population slices | Target correctness now; additional competing modeling policies as controlled experiments |
| Data loading | Epoch profiling, local cached files, prebuilt chunks | Profile within a planned run; implement only for a measured bottleneck and actual available disk |
| Data scale and sampling | 10/35/100/500-file curves; retaining fraud accounts and sampling others; non-fraud loss sampling; recency weighting | 10/100 first; larger designs later with explicit inclusion probabilities and loss objectives |
| Sequence context | Read-only prefixes at chunk boundaries; mixed chunk/sliding training; longer history | Later matched experiments; the previous fully sliding result does not prove a prefix will help or fail |
| Tuning | Broader LR, decay, dropout and clipping search; model size; other optimizers; EMA; class weights or focal loss | Retain current settings for the feature comparison; reopen for demonstrated instability or a separately budgeted study |
| Representation quality | Effective rank and spectra, head-orthogonal 47-dimensional representation, linear CKA, cross-seed alignment, zero-retrain swap, ridge orthogonalization | After a basic downstream result identifies what additional question matters |
| Additional embedding tasks | Controlled seven-day future-fraud probe; auxiliary self-supervision or multitask objectives | Separate research; these do not become required fraud-model deliverables |
| Compression and correlation | PCA to 8/16 dimensions fitted on downstream training; correlation with existing scores and foundation-model embeddings | Follow up on the managers' reuse/compression questions using available compatible inputs |
| Richer downstream features | Strictly earlier 1/7/30-day vector averages, deviations from those averages, cold-start/history-depth features | After the plain score/vector comparison, using point-in-time construction and a matched baseline |
| Further consumer arms | Vector-only addition, incumbent offset versus concatenation, foundation-model vectors, both vector sources, aligned-seed research arms | Only when the necessary inputs and a concrete question are available; one selected model remains the deployment target |
| Store and refresh | Scheduled refresh, replay on new months, cross-version compatibility, online serving, product selection | After useful offline exports and a working lookup; no new serving platform in today's queue |

The proposal record includes corrections to earlier research claims and compute assumptions. No published paper's reported gain, another dataset's fraud rate or an unmeasured speed estimate is adopted as this project's expected result.

## 7. Completion table to fill from actual runs

Use one row per selected model/checkpoint and scored population. The detail below belongs in the local result file; a short summary is enough for the daily handoff.

| Job | Status | Features / encoder | Training files | Seed | Preprocessing / config location | Checkpoint / output location | Evaluation population | UDR @ 0.6% | AP | AUROC | Comparison / uncertainty | Next action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Current features, 10 files | Reuse if compatible; otherwise pending | Current | Recorded 10-file list | 42 | To fill | To fill | Designated development rows | To fill | To fill | To fill | Matched candidate comparison | Confirm compatibility |
| Expanded features, 10 files | Pending | Confirmed candidate | Same 10-file list | 42 | To fill | To fill | Same rows | To fill | To fill | To fill | Candidate minus control | Refit and run |
| Tree: inputs only | Pending | Fixed tree inputs | Downstream training | Fixed | To fill | To fill | Downstream development | To fill | To fill | To fill | Baseline | Prepare downstream split |
| Tree: inputs + score | Pending | Current frozen encoder | Same downstream rows | Same | To fill | To fill | Same rows | To fill | To fill | To fill | Score contribution | Export and fit |
| Tree: inputs + score + vector | Pending | Same frozen encoder | Same downstream rows | Same | To fill | To fill | Same rows | To fill | To fill | To fill | Vector contribution beyond score | Export and fit |
| Current features, 100 files | Pending pilot decision | Current control | Recorded 100-file list | 42 initially | To fill | To fill | Designated development rows | To fill | To fill | To fill | Matched expanded-feature comparison | Prepare config |
| Selected expanded features, 100 files | Pending pilot decision | Selected candidate | Same 100-file list | 42 initially | To fill | To fill | Same rows | To fill | To fill | To fill | Candidate minus control | Prepare config |

For each comparison also record evaluated transaction count, fraud count, unique accounts, eligible-row exclusions, missing-score coverage, alert count and frauds caught. Retain planned secondary measures such as value capture, KS and FPR when available, but do not substitute fraud value for the primary fraud-transaction capture question. Record confirmation seeds separately when run. A completed process with no valid result file is not a completed comparison.

## 8. Copilot setup and command handoff

This checklist does not establish the current native directory layout. Fill the locations below from the actual working project rather than inventing a new folder hierarchy.

| Item | Local value to fill |
|---|---|
| Project root | Actual active working directory |
| Python | Actual environment/interpreter |
| Current model config and checkpoint | Existing compatible chosen run |
| Current fitted preprocessing | Artifacts associated with that checkpoint |
| Expanded-feature specification | Confirmed feature table and config |
| Train / validation / evaluation input locations | Existing project data locations |
| Incumbent and foundation-model scores | Actual score files and transaction-key columns |
| New job outputs | Separate destination for each job |
| Downstream export table | Transaction-keyed score/vector output |
| Results summary | One daily result table and next-command note |

The engineering handoff must supply a verbatim command and expected output for each operation:

| Order | Operation | Expected output |
|---|---|---|
| 1 | Activate/check the local environment and resolve configs | Correct imports and resolved paths |
| 2 | Validate the confirmed feature map and target policy | Final input/target definitions and usable-row counts |
| 3 | Fit the required preprocessing | Separate compatible fitted artifacts |
| 4 | Run the small train/reload/score/export check | Valid checkpoint, scores and vectors |
| 5 | Train the matched 10-file jobs | Selected checkpoints and training summaries |
| 6 | Score the development populations and compare | Matched metrics and uncertainty |
| 7 | Export from the existing frozen encoder | Reusable downstream score/vector tables |
| 8 | Fit and score the three tree arms | Downstream comparison table |
| 9 | Align the actual incumbent/foundation-model benchmark | Matched coverage and score table |
| 10 | Launch the specified 100-file jobs after the pilot decision | Larger-fit checkpoints and subsequent comparisons |
| 11 | Freeze, export and evaluate the selected deliverable | Final report and working embedding lookup |

Long-running commands must explain where their logs/results go and how to determine whether they completed. Re-running a completed job should be an explicit action, not a side effect of starting the next one. If two workbenches are used, assign disjoint jobs/output directories and keep the workflow valid when only one remains.

## 9. Existing downloads

The repository currently provides these earlier tools. Their presence is not evidence that they are installed locally, that every module is needed today, or that the reduced workflow has been assembled.

- [Existing start-here guide](COPILOT_START_HERE_REVIEWERS_SUGGESTIONS.zip)
- [Clean tuning package](tuning_package_part1_clean.zip)
- [Feature tools](features_testing_package_REVIEWERS_SUGGESTIONS.zip)
- [Shared-workbench worker add-on](features_addon_two_workbenches_REVIEWERS_SUGGESTIONS.zip)
- [Existing model/checking add-on](stage1_checks_addon_REVIEWERS_SUGGESTIONS.zip)
- [Optional time-encoding experiment](time_encoding_test.zip)

Use the reduced job list and the actual local setup to determine which existing modules to reuse. The worker add-on can also be a dependency of checking tools in single-worker operation. The optional time experiment is not a prerequisite for the feature or embedding pilots. No ZIP is changed or removed by publishing this checklist.
