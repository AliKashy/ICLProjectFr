# CH — proposed fix and experiment queue, October 6

**My latest update:** Both of my bank training scripts use their Phase 4A/4B sequence builders; CH has verified that GitHub still contains the old imports. I need to synchronize those files so the repository reproduces my bank setup. I have created h0 and plan to run it; there is no completed h0 result yet. CL also recorded that my corrected knob finder is still running and AdamW is my chosen fidelity baseline. The newer plan table remains provisional.

> Public first-person project copy prepared by CH on 2026-10-06. I keep my original findings and guides unchanged. CH performed the checks reported here. Source paths and commands are illustrative private-project aliases; code, evidence programs and the candidate patch remain in the private review record. Exact approved scientific details are retained.

This is CH's recommendation after reviewing current source `October-6-reviewed-source`. It is not a jointly approved change or a claim that any fix has been applied. The [review](REVIEW.md) gives evidence and qualifications. Preserve source-reconstruction history separately from deliberate repairs.

## First repair batch: make the intended Phase 4 experiment trustworthy

| Order | Concrete change | Acceptance evidence |
|---|---|---|
| 1 | Synchronize both GitHub Phase 4 training imports with the reported bank copies, selecting their phase-owned `SequenceBuilder`. A minimal two-import candidate patch is prepared in the private review record, unapplied. | Actual `_make_loader` functions emit the complete target-ID multiset exactly once with 0/2/4 workers, unequal shards and at least two epochs. Include an assertion of the selected class. |
| 2 | Give every experiment attempt an immutable ID and directory/object prefix; save resolved config, source/environment identity, ordered feature/artifact hashes, target manifest and checkpoint identity. | Two executions of the same grid preserve both attempts. Different seeds and cloned grids cannot overwrite each other's predictions. Explicit overwrite/resume is a distinct operation. |
| 3 | Handle failed training/scoring explicitly. Read only the current attempt's metrics, record stage status, exclude failed runs from selection, propagate required-stage failure. | A previous successful CSV plus a new startup failure yields no new performance claim. Scoring-disabled, scoring-failed and parsing-failed are distinguishable. |
| 4 | Preserve causal/padding fixes and add behavior-based regression tests. | Two/four-layer target perturbations cannot alter earlier real states; padded values/gradients cannot affect real outputs; finite forward/backward under representative precision/padding. Deliberately restoring the old relay makes the test fail. |
| 5 | Set explicit safe microbatch/accumulation for h10 time-gap/capacity configs; validate phase-specific `model.type`. | h10 begins with the documented 1024×4 schedule, with GPU measurement for wider/deeper models; invalid architecture requests fail clearly. Identical effective batch is reported without claiming identical optimization. |
| 6 | Correct operational plan references: `buffer_windows`, actual `scripts/train_sequential.py` paths, existing B-long filename, no-SEP A and fresh-encoder B descriptions. | Every proposed command resolves to the intended phase entrypoint and its effective settings are recorded before launch. |

Run the existing suite and the focused regressions in a pinned environment. Then perform a small train/checkpoint/score smoke on the actual bank stack. The CPU evidence here does not certify CUDA kernel choice, H200 batch limits, throughput or host RAM. Capture the exact traceback and backend for the reported long-B launch failure, then record the successful retry rather than assuming the arithmetic proves readiness.

## Statistical protocol before using “better” or “tied”

1. **Name the question.** Is account history useful within the same encoder family? Is hierarchical history more useful or cheaper at a given budget? Keep these distinct from optimizer and pretraining questions.
2. **Freeze comparisons.** Select one training/validation/development manifest and ordered preprocessing version. Verify unique targets, label agreement, timestamp ranges, mature labels, decision-time feature availability and the same target population across history lengths. Preserve short-history targets too.
3. **Keep the timeline honest.** Old sweep evaluation is one-file, per the recorded user answer. Current defaults are eleven files. The newer finder summary needs its own raw rows/source/config/checkpoint record. These are separate result families.
4. **Start with A h0 versus h5.** Same corrected loader, model recipe, optimizer and target population. Add a fixed-slot masked-history control if the aim is to isolate only history information from parameter/layout changes. Keep time-gap channel handling explicit.
5. **Use repeated seeds for shortlisted models.** Start with 3–5, show every result, then increase if the decision remains unstable. This is a budget starting point, not a statistical theorem.
6. **Measure paired differences.** Resample matching account/time-appropriate blocks for both models. Distinguish sampling uncertainty from training-seed variation. Specify a practical improvement/equivalence margin before interpreting tiny differences. A last-seed interval beside a seed mean is invalid.
7. **Use validation for selection.** Hold a later untouched, naturally distributed cohort with mature labels for final confirmation. Files 8000–8010 are a development benchmark already consulted; not showing them to an automated agent does not reset that history.
8. **Evaluate business utility.** Use matched scores from the bank's existing model. Report AP, AUROC, counts/prevalence, precision/recall at the real review capacity, fraud-value capture where justified, calibration, history-retrieval cost, peak memory and latency. AP 0.14 is not 14% accuracy or recall.

Before this protocol uses the scoring toolkit, fix tie ranks, exact-cohort/label joins, seed-mean intervals, temporal row retention and calibration/budget validation. Save fitted calibration/ensemble objects and keep their fitting targets separate from final evaluation. Show undefined metrics for single-class periods instead of silently omitting those periods.

## Then follow the updated research plan

| Study | Recommended comparison | Interpretation limit |
|---|---|---|
| Knob confirmation | Linear/periodic variant, bidirectional/corrected causal, AdamW baseline; optional separately labeled RAdam challenger | One seed or one embedding implementation cannot settle the whole method family. |
| History and architecture | A/B at matched h5/10/15/20, with code/config/compute records | A4 versus B2+4 blocks is not parameter matched; report parameters and GPU hours. |
| Time-gap ablation | Same selected history, safe batch and recipe with time gap on/off; optionally retain a zeroed channel | Removing a token changes layout/parameter count as well as information. |
| Capacity | Current three width/depth choices, followed by explicit LR/schedule/regularization tuning if useful | A fixed-LR losing larger model does not establish a capacity ceiling. Warmup is not currently implemented just by adding a config key. |
| Data scale | 100 files first, then 200/500 if worthwhile | Keep validation common or re-evaluate all candidates on a declared new version. |
| Pretraining | Corrected SSL/SupCon versus identical scratch encoders and downstream budgets | Objective fixes, strict checkpoint loading, correct class-weight scorer reconstruction and corrected loader are prerequisites for the affected path. |
| Automated search | Verified proxy versus corrected full-run references; immutable data/metric/cohort code, all-attempt ledger and cost cap | Repeated adaptive selection still overfits development data; retain independent final testing. |

## Optional capabilities need their own gate

- EMA remains disabled during baseline comparisons. Before enabling it across batch ladders, update on optimizer progress and test partial accumulation, saved raw/shadow state, exact resume, callback order and within-epoch validation.
- Packaging requires a valid build backend, inclusion of my implementation/config resources, declared dependencies and an installed-package smoke outside the source checkout. Alternatively, record and validate an explicitly supported source environment; do not call that a successful clean package install.
- The original streaming constructor and modern-Torch load path need their own compatibility fixes before rerunning that original baseline in this environment.
- Multi-GPU requires rank-aware partitioning, valid step synchronization and globally aggregated metrics. Single-process worker fixes are insufficient.
- Truly unlabeled SSL needs a loader contract without a required fraud-label column. Optional cyclical features need their separate streaming-loader correction.

## Bank facts still needed

- My bank training imports are answered: they use the phase-owned builders. I need to synchronize GitHub and record the exact source/config for each new run; other differences remain unverified.
- Which October 6 runs actually finished, including B and the 100-file confirmation? Preserve exact cohorts and result rows.
- Are My artifacts byte-identical to Ya's? Which ordered feature schema and artifact hashes pair with each checkpoint?
- What are split dates, label-maturity rules, equal-time availability order and account-per-file guarantees?
- Are prior predictions/checkpoints/logs still retained, and what is the weekly GPU budget and actual alert capacity?

The historical evaluation-range question is answered; do not ask it again as if unresolved. Remaining run provenance can be supplied without publishing raw transaction records.

## Late peer verification

CL's h0/h5 three-seed config is a useful proposal; my separate h0 config is created and its execution/results are pending. No need to delay a corrected EMA-off h0 control for optional EMA or Phase 5 repairs. Keep h0/h5 on the same data, validation, recipe and selection rule; save every attempt's config/checkpoint/metrics and identified validation predictions. Do not compare h0 against an absolute historical AP cutoff.

The latest default pytest command also collects two failing proposal-only tests. Separate unapplied proposals from normal discovery, then retain explicit installed-proposal checks. The proposed EMA callback needs its last-step counter synchronized after Lightning restores optimizer progress; verify uninterrupted versus resumed training with accumulation before accepting it. Details are in the peer review.
