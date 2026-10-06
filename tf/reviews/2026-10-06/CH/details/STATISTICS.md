# Independent statistical and scoring review — October 6, 2026

**Status qualification:** This detailed report tests the reviewed GitHub snapshot. My later update says the bank trainers already use the corrected phase-owned builders; that makes the import finding a repository sync gap. My h0 config is created, results are pending, and the corrected knob rerun table is provisional. See the [current status](../README.md).

> Public first-person project copy prepared by CH on 2026-10-06. I keep my original findings and guides unchanged. CH performed the checks reported here. Source paths and commands are illustrative private-project aliases; code, evidence programs and the candidate patch remain in the private review record. Exact approved scientific details are retained.

Reviewed immutable source `October-6-reviewed-source` in `<private-checkout>`, compared with `previous-reviewed-source`. No source, bank data, checkpoints, or S3 objects were changed. Tests used the audit CPU environment and synthetic local inputs. Severity below refers to trustworthy experimental conclusions, not a security classification.

## Conclusion

The latest scoring changes are an improvement, and both real scoring entrypoints passed a synthetic end-to-end exercise of configuration, DataLoader iteration, positive-class softmax, metadata alignment, parquet writing, and AP/AUROC. The repository does **not** yet establish that the whole project is statistically validated, that history causes the observed lift, or that a model is ready for a business decision. Important comparison/reporting utilities retain reproducible errors, historical results use a different evaluation cohort from the reference, and no untouched temporal confirmation or multi-seed paired uncertainty is recorded.

## Coverage of assigned changed files

| File | Review and result |
|---|---|
| `my_work/phase4a/scripts/score_sequential.py` | Full file + diff + actual `main()` on synthetic inputs. Correct phase-owned builder import at line 28; metadata/target/probability alignment passes. Distinct run names preserve separate outputs at lines 103–106. Full-stream AP/AUROC calculation is appropriate at lines 109–112. |
| `my_work/phase4b/scripts/score_sequential.py` | Same full review and integration test. Builder import line 28; run namespace lines 99–102; metrics lines 105–108. Same passing result. |
| `bridge/ASK_my_work.md` | Full file. Lines 8–13 resolve historical scoring: one file, part_8000, 290 fraud cases; reference 0.0940 used eleven files. Opening question lines 3–4 is now stale about current config defaults. Remaining unanswered artifact/run-log questions still matter. |
| `Update Overview/Changes Made - 2026-10-06 11-02 AM ET.md` | Full file. Separates intermediate versus final optimizer decision, and explicitly says proposed commands are not proof of execution (177). New sweep summaries at 100–108 have no committed raw eight-row results here and differ from October 4 summaries. Treat as reported separate run evidence, not replacement verified metrics. |

Also read all four `my_work/scoring/*.py` tools, `my_work/docs/my_work_current_standing.md`, phase4a/b base/sweep configs, both score/train/runner contracts, `fraud/callbacks.py`, the updated Phase 4 plan and statistical sections of the updated guide, plus the autoresearch proposal. Training/loader/model correctness is separately owned by the other reviewers; relevant cross-cutting issues are noted below.

## Confirmed progress

1. **Current A/B base configs now select eleven evaluation files.** `phase4a/configs/sequential.yaml:81–87` and `phase4b/configs/sequential_hier.yaml:89–95` specify 8000–8010. This makes future default scoring consistent with the intended development benchmark. It does not transform historical one-file results into eleven-file results.
2. **Separate run names now preserve separate prediction files.** Actual scoring tests produced `run_a_preds.parquet` and `run_b_preds.parquet` for each phase, with four correctly aligned targets, two positives, and AP/AUROC 1.0. No bank-model quality inference follows from this synthetic test. There is no established collision among the six currently checked-in grids; A/B bases differ and B short/long history names are disjoint. Repeating a run still overwrites its deterministic output, so an execution ID/immutable manifest is still needed.
3. **Training/score override precedence is now consistent.** Runners use common overrides before axis overrides on both paths. This is a resolved historical bug; the separate training review covers failure handling and other remaining runner limitations.
4. **The full validation-stream metric is correct for one process.** `fraud/callbacks.py:58–79` clears state each epoch, collects positive-class softmax and labels across batches, and computes sklearn average precision/AUROC once. It does not average batch AP, which would be inappropriate here. Multi-GPU global aggregation is not established by this code.
5. **Time-gap and capacity ablation configs now exist.** They add useful experimental controls and disable evaluation scoring during these exploratory sweeps. Their existence is not evidence the runs completed. A matched history-zero experiment remains missing from the executable grids.

## Findings requiring correction before strong claims

### S1 — High: historical results do not support the claimed matched comparisons or equivalence

Evidence: `bridge/ASK_my_work.md:8–13`; `my_work/docs/my_work_current_standing.md:30–32,79,126–136,154`; new overview `:100–120`; updated plan `docs/phase4_sequential_plan.md:50–61`.

The historical h5/h10/h16 evaluation APs 0.0893/0.0902/0.1171 are from part_8000 (290 positives). Ya's 0.0940 is from 8000–8010 (3,818 positives). They are not comparable model rankings. The standing document still says eleven files, “statistically tied,” and “all apples-to-apples,” despite mixed training scales and no paired uncertainty. The guide now correctly discloses the one-file evaluation and describes history-length/optimizer effects as unresolved.

The new overview reports AdamW 0.1189 versus RAdam 0.1273 and says RAdam won all four pairs. The old eight documented rows give AdamW 0.128346 versus RAdam 0.1285835; RAdam wins only two of those four matched optimizer pairs. These must not be silently merged. Capture the new raw result rows, resolved config, seed, code/artifact fingerprints, dates, and predictions under a separate run identity. The plan calls the new sweep “clean,” but normal training still imports the original faulty builder (`phase4a/scripts/train_sequential.py:27`, and phase4b same); only scoring is wired to the new builder in this revision. A different pod checkout could have behaved differently, but the repository cannot certify that run's training provenance.

Small arithmetic correction: recomputing exactly the eight old displayed rows gives linear mean **0.132579** and bidirectional mean **0.1289745**, rather than standing's 0.132604 and 0.128987. This tiny discrepancy does not alter qualitative interpretation; it reinforces the need to derive summaries from saved raw files.

Action: keep old results labeled historical/development, record the new sweep separately, rerun controls after training actually uses the corrected loader, and use paired uncertainty before claiming superiority or equivalence. “Effectively tied” is only a description of a small observed gap, not a statistical equivalence result.

### S2 — Medium: ensemble tie handling fabricates discriminatory power

Evidence: `my_work/scoring/ensemble.py:37–39,49–50,89–95`; unchanged known finding C-16/E3.

Double `argsort` assigns arbitrary ranks to equal scores. Executed reproduction: all scores 0.5 with labels `[0,0,1,1]` have AP/AUROC **0.5/0.5**, but `_rankpct` makes both **1.0**. Reversing row order gives AP **0.4166667**. This affects individual metrics and the tuned ensemble itself.

Action: use average ranks for ties, verify permutation invariance, and display individual model metrics from the original scores. Fit/persist ensemble transformations on development data and evaluate only on a separate intended cohort. Ranking the test cohort itself is a batch-relative scoring strategy; deployment needs an explicit frozen training/calibration rank transform or an acknowledged batch-scoring design.

### S3 — High for comparison claims: cohort identity, join cardinality, and label agreement are not enforced

Evidence: `my_work/scoring/ensemble.py:42–51`; `leaderboard.py:80–108`; `temporal_eval.py:56–68`.

Executed existing reproduction: model-one IDs `[1,2]` and model-two IDs `[2,2]` join into `[2,2]`, dropping one target and duplicating the other; conflicting model-two labels are ignored. Executed leaderboard `main()` accepts models with different cohort sizes (2 versus 20) and ranks them together without checking IDs. Executed temporal `main()` duplicates a positive transaction from duplicate timestamp rows and silently drops another positive with an invalid timestamp.

Action: require unique transaction IDs, equal labels, exact intended target membership and prevalence/counts; use validated one-to-one merges and report rejected/unmatched rows. Explicit intentional intersections require a new named cohort and exclusion report. Do not read a directory of multiple run prediction files as if it were one model cohort: the new per-run files make exact-file/manifest selection essential.

### S4 — High for inferential claims: leaderboard interval belongs to the last seed, not the displayed mean

Evidence: `my_work/scoring/leaderboard.py:45–59,84–106`; unchanged known C-16/E5.

Executed real `main()` with two synthetic seeds, AUROC 1.0 and 0.0: it displays **mean AUROC 0.5 with CI [0.0000,0.0000]**. The interval is computed from the last seed alone. Independently, the bootstrap samples individual transactions; account transactions and overlapping history windows are correlated. It is not a paired difference interval between models.

Action: keep training-seed variation separate from sampling uncertainty. Report each seed's metric and use the same resampled account blocks and target IDs for both models when estimating the difference. Choose account/time blocking to match the intended future-transaction estimand; account blocking alone does not establish robustness to calendar drift. For a seed-mean estimand, explicitly implement resampling/aggregation over seeds and accounts, or report paired per-seed differences plus clearly labeled within-seed sampling intervals. Never infer equivalence from overlap of separate intervals; define a practical equivalence margin first.

### S5 — Medium: calibration claims and operational metrics need a clearer contract

Evidence: `my_work/scoring/calibrate_and_report.py:7–12,44–60,75–103`; `my_work/docs/my_work_roadmap.md:39`; unchanged C-16/E7/E8.

The tool correctly fits a calibrator on its `--val` input and applies it to `--test` and computes Brier scores. But it does not check split overlap or save the fitted calibrator. It prints raw-score ranking and raw-score alert-budget metrics while claiming all monotonic calibration preserves ranking. Isotonic regression can create ties; the reproduced example changes AP **1.0 → 0.8333333**, AUROC **1.0 → 0.875**. Unconstrained sigmoid fitting can reverse scores if its slope is negative. Calibration is not required for AP comparison under a strictly increasing transformation, but it is required to interpret probabilities/expected losses and select probability thresholds reliably.

The default positive budgets are sensible exploratory choices; unsupported inputs are not validated. On four rows, 0% still requests one alert and 200% reports eight alerts although only four rows exist. Default large-cohort budgets do not hit that 200% edge case.

Action: separate model selection, calibration, and final testing; reject unintended target overlap; save calibrator/weights and provenance; explicitly report raw-ranking versus calibrated-probability metrics; define tie breaking and budget rounding; require valid fractions. Brier alone should be contextualized against a prevalence-only predictor and reliability bins because most examples are nonfraud. Report recall/precision at the actual bank review budget and amount-weighted fraud captured if transaction amount reliably represents the business value being estimated.

### S6 — High for final-generalization claims: temporal bucketing is not a clean future holdout

Evidence: `my_work/scoring/temporal_eval.py:56–107`; standing `:154`; guide Q18 `guide/parts/07_questions_and_answers.md:198–203`; autoresearch proposal `bridge/proposals/autoresearch_CL.md:35–40,49–56,76–79`.

The temporal tool only groups existing predictions after a timestamp join. It does not define train/test cutoffs, check whether features/labels existed at decision time, or create an unseen holdout. Single-class buckets are omitted completely; the executed main-path test omitted a whole February bucket. Slope uses successive bucket indices rather than actual time spacing and rounded AP values; a negative slope alone cannot establish model drift because AP also changes with prevalence and sampling variability.

The proposal's “held-back” pinned eleven-file set is already part of reported development evidence, including part_8000 used repeatedly. Hiding it from a future search agent does not make it a previously untouched project test set. The proposed two seeds/no-change margin reduce training-noise selection but do not eliminate adaptive validation overfitting. The proxy correlation reference should use corrected full runs, not rely only on old runs from the faulty training loader.

Action: retain 8000–8010 as a matched development benchmark; define a later untouched naturally distributed target cohort with mature fraud labels, as-of feature availability, allowed preceding history, locked selection criteria and independent final evaluation. Show all temporal bucket counts/prevalence, undefined metrics explicitly, actual calendar spacing and uncertainty. Treat trend flags as monitoring signals requiring investigation.

## Statistical plan that would support a strong project

1. **Trustworthy run provenance first.** Route training through the corrected builder; record unique input/target counts, resolved configs, code revision, ordered feature/artifact hashes, checkpoint hash and immutable execution IDs. Freeze one matched validation/development cohort and checkpoint selection rule.
2. **Separate scientific questions.** Keep AdamW as the replication baseline if that is the stated goal; report RAdam as an exploratory optimizer comparison rather than calling an intentionally fixed baseline the numerical winner. The new reported optimizer outcome and old sweep must have separate lineage.
3. **History attribution.** Run the same history-model family at h0 and h5, and compare A/B at matched histories and train budgets. Keep identical targets and preprocessing, including short/no-history targets. Run time-gap on/off and capacity controls. A removed feature changes token count/parameters; if isolating only information content, add a zeroed time-gap channel control. At h0 the current time-gap-to-target is always zero (`sequence_builder.py:269–270`).
4. **Repeated training plus paired testing.** Start with 3–5 seeds for shortlisted configurations (more if decisions remain unstable), report per-seed scores and seed spread; use paired account/time-respecting resamples for model differences on the same targets. Predeclare the minimum business-relevant improvement. More seeds do not independently create a fresh test set.
5. **Freeze a final temporal holdout.** Verify timestamp ranges, train/test target disjointness, transaction uniqueness, label maturity and point-in-time inputs. Existing-account overlap may be legitimate for predicting future transactions of existing customers; never use future transactions/labels. Define a separate new-account assessment only if that is part of the intended use.
6. **Compare the bank baseline on the exact same cohort.** Include current baseline scores with transaction identity, target labels and as-of availability. AP ≈0.14 is a ranking summary, not 14% accuracy or recall. Report AP, AUROC, positive prevalence/count, precision and recall at the alert budget, uncertainty, calibration and operational latency/history-lookup cost. No existing-bank-model lift can be established without its matched predictions.
7. **Move to pretraining/automated search after this gate.** Use matched from-scratch controls and independent search/test separation. Persist every attempt, including failed/discarded ones, so the search extent is visible and results cannot be cherry-picked.

## Reproduction and precise limits

- `PYTHONDONTWRITEBYTECODE=1 <audit-python> CH/checks/evaluation_checks.py` from the pinned repository: exit 0, all historical undesirable-behavior reproductions still hold. Output saved as `existing_evaluation_checks.json`.
- `PYTHONDONTWRITEBYTECODE=1 <audit-python> <private-evidence>/statistics/check_scoring.py`: exit 0. Saved `scoring_results.json`, `scoring_stdout.txt`, `scoring_stderr.txt`, synthetic parquet files and metric CSVs. The test intentionally asserts current defects, so passing it means the issues are reproduced, not fixed.
- The scoring integration uses actual entrypoint/config/DataLoader/softmax/parquet/metric code with an explicitly synthetic model, dataset and artifact I/O. It does not test the real bank model weights, cloud permissions, H200 precision/kernel behavior, large-cohort RAM/runtime, or on-pod execution.
- Bank checkpoints, raw predictions, newly reported result CSVs, timestamps/label maturity, artifact byte identity, production model outputs, training-seed variance, calibration and final temporal-holdout results are not available here. Historical numbers are repository reports, not independently recomputed performance. No claim of statistical significance, equivalent models, business savings or production readiness is justified from these checks alone.
