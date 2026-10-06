# Training lifecycle and experiment-runner review — 2026-10-06

**Status qualification:** This detailed report tests the reviewed GitHub snapshot. My later update says the bank trainers already use the corrected phase-owned builders; that makes the import finding a repository sync gap. My h0 config is created, results are pending, and the corrected knob rerun table is provisional. See the [current status](../README.md).

> Public first-person project copy prepared by CH on 2026-10-06. I keep my original findings and guides unchanged. CH performed the checks reported here. Source paths and commands are illustrative private-project aliases; code, evidence programs and the candidate patch remain in the private review record. Exact approved scientific details are retained.

Reviewed application source at `October-6-reviewed-source`, compared with `previous-reviewed-source`. This is a review of current application files, not the stronger, separate patches stored under `CL/patches/`.

## Conclusion

The training/scoring override-order defect is fixed. The standard end-of-epoch EMA checkpoint now contains the weights actually evaluated. Current six ablation specs expand coherently to 20 training runs, of which 15 score; nominal effective batch size is 4096, and scoring receives the history/batch overrides and full 8000–8010 range.

The experiment runner still lacks durable attempt identity and reliable failure accounting. Fresh runs overwrite prior evidence, a failed rerun can report old validation results, and failed scoring is not represented by an explicit score status. Optional EMA remains unsafe for exact resume, optimizer-step-based averaging with accumulation, or within-epoch validation. EMA is false in all current base/sweep configs; its remaining defects do not establish contamination of documented historical runs. Within-epoch validation and resume also are not exposed by current training entrypoints, so those lifecycle findings are future-use limits rather than current default-run blockers.

The separate data review owns the more immediate active-path problem: both training entrypoints still import the old `fraud.data.sequence_builder.SequenceBuilder`, while both scorers import the new phase-local loader. This review inspected that wiring and avoids duplicating the data review's finding.

## Evidence and scope

- Python interpreter: `<audit-python>`.
- Actual runtime: `pytorch_lightning==2.6.6`, `torch==2.14.1+cpu`.
- `check_training.py`: 14 real scalar-model Lightning fits (seven cases per phase), and the actual runner's `main()` over every current sweep with only subprocess execution replaced by a recorder. Real configs, OmegaConf merging and runner branching execute. No S3, bank data, production training, or GPU was accessed.
- `ema_results.json`: raw training starts, optimizer steps, EMA state, validation weights and checkpoint contents.
- `sweep_results.json`: all 35 generated subprocess commands represented as effective settings, plus overlap/rerun/failure probes.
- `verify_evidence.py` / `assertions.json`: 93 passed evidence assertions, including assertions that reproduce defects. This count is not a claim of 93 passing product tests.
- `run.log`: actual Lightning and runner output. Diagnostic metric named `valid_ap` is a scalar parameter for callback testing, **not a fraud-model AP measurement**.
- Original application files remained unchanged.

## T1 — Preserve an immutable identity for every attempt (P2; active on fresh reruns)

**Source:** both `scripts/run_ablations.py:45–57,112–125`; 4a `scripts/score_sequential.py:103–106`, 4b `scripts/score_sequential.py:99–102`; both train scripts `119–121`; `fraud/callbacks.py:34–37`.

The run name contains only axis labels. `output_root/run_name` is reused on a subsequent invocation, log files open with `w`, the metrics callback truncates the prior CSV on first validation, `best.ckpt` and `last.ckpt` reuse their names, and predictions write to the same `<run_name>_preds.parquet`. The requested fresh reruns therefore replace earlier evidence unless the operator manually moves it first.

Two consecutive executions of each real runner produce identical train/scoring command identities and checkpoint paths. A second grid with another output root but the same axis labels also shares its prediction filename under the same predictions base: `ablations/seed42/history-h10` and `ablations/seed43/history-h10` both write `history-h10_preds.parquet`.

**Scope correction:** there are **no cross-grid prediction collisions among the six currently checked-in sweep specs**: A and B have different prediction bases; the short and long B grids have disjoint histories. The confirmed issues are repeated attempts and future/cloned grids sharing labels; a same-named A/B row by itself does not establish collision.

**Recommendation:** derive identity from the full experiment, effective config and attempt ID; record source revision, artifact hashes, seed and file ranges. Refuse existing destinations unless explicit resume/overwrite mode is selected. Resume should load a checkpoint rather than silently restart. The existing roadmap already describes canonical grids as from-scratch; absence of resume itself is not a contradiction.

## T2 — Failed runs can inherit old metrics and remain ranked (P2; active failure path)

**Source:** both `scripts/run_ablations.py:125–136`; 4a `162–175,178–182`; 4b `156–175`; both scripts end by calling `main()` without forwarding aggregate subprocess failure as exit status.

After a successful run leaves `training_metrics.csv`, a rerun failing before its first validation does not clear that file. The runner reads its maxima even when `train_rc != 0` and ranks the row alongside successes. Reproduction for both runners: simulated startup failure leaves a new result row `train_rc=1,best_valid_ap=0.9,best_valid_auroc=0.95`, values produced by the previous synthetic run. Console summary omits `train_rc`, so it presents those stale metrics as this attempt's result.

Scoring has a similar accounting gap: `s_rc` is checked but never stored in results, so blank evaluation metrics can mean scoring disabled, scorer failed or output parsing failed. The outer process reports normal completion despite failed subprocesses. This can mislead unattended experiment orchestration and comparison.

**Recommendation:** isolate attempts, require metrics from that attempt, suppress selection/ranking of failed training runs, store train/score status and parsing status, and return nonzero if a required stage fails. Preserve partial success rows rather than deleting them.

## T3 — EMA still averages each microbatch, changing the method with accumulation (P2, conditional on EMA enabled)

**Source:** both `callbacks.py:46–49`; trainer accumulation at both `scripts/train_sequential.py:129`.

The hook updates EMA on every batch, even when no optimizer update completed. In both real Lightning runs with four batches, accumulation two and decay 0.5, final raw weight is 0.36. Optimizer-step EMA should be 0.23; implemented EMA is 0.255 because the non-stepping third microbatch averaged raw 0.20 a second time. This changes smoothing when batch/accumulation choices change.

**Do not transplant the old v2 verdict:** the current code has no modulo guard. Three batches with accumulation two update the final partial group correctly (EMA 0.19); three batches with accumulation four also update the final group (0.075). Its defect is extra updates, not the old candidate's missing-final-group update.

**Recommendation:** update only when observed optimizer-step progress advances, including partial final groups; test no-op/scaler-skipped updates when adding CUDA mixed-precision coverage. Defaults currently have EMA false.

## T4 — The EMA checkpoint is inference-correct but not exact training-resume state (P2, conditional on EMA and resumed training)

**Source:** both `callbacks.py:37–44,51–72`, with no `state_dict`, `load_state_dict`, `on_save_checkpoint` or `on_load_checkpoint` implementation. Current `train_sequential.py:136` does not supply a resume checkpoint argument.

The end-of-epoch checkpoint now saves evaluated EMA weights (0.23), correctly fixing the old raw/EMA mismatch. It does not save the optimizer's corresponding raw model weights (0.36) as separate resume state. No EMA callback state exists in the checkpoint. Resuming with actual Lightning `ckpt_path` starts the optimizer from the EMA weight, unlike uninterrupted training:

| Two-epoch test | Final raw weight | Final EMA |
|---|---:|---:|
| Uninterrupted | 0.59040 | 0.47470 |
| One epoch, save, resume | 0.50720 | 0.40710 |

Identical result in 4a and 4b. The test uses deterministic SGD and no stochastic model layer; the difference is a lifecycle issue rather than sampling noise.

**Recommendation:** persist raw weights and EMA shadow with explicit restoration semantics, leaving the scoring state dict unambiguous. When adding CLI resume, ensure optimizer, scheduler and callback states resume together. This does not block scoring the current EMA checkpoint, and current entrypoints do not yet offer resume.

## T5 — Delaying raw-weight restoration until the next epoch breaks within-epoch validation (P2 conditional; not configured by current entrypoint)

**Source:** both `callbacks.py:51–69`.

After validation the model remains on EMA weights until next epoch start. With actual Lightning `val_check_interval=1`, first optimizer update reaches raw 0.20, validation swaps in EMA 0.10, and the next training batch starts at 0.10 rather than 0.20. The final raw weight becomes 0.28 rather than 0.36; training has silently changed. A subsequent validation overwrites the prior raw backup.

**Controls:** standard epoch-end validation passes, checkpoint and evaluated EMA agree, raw weights restore at the next epoch and `on_fit_end`; initial sanity validation does not perturb the fresh-run initial parameter. Lack of a sanity guard alone is not reported as a demonstrated fresh-run defect.

**Recommendation:** restore raw weights before the next training batch or optimizer action while explicitly controlling checkpoint serialization, rather than relying on the next epoch. Test callback ordering and sanity validation alongside this path. No `val_check_interval` is passed or read in current training scripts, so this is a feature limitation for future trainer changes.

## Confirmed improvements and config coverage

| Current file(s) | Review outcome |
|---|---|
| `my_work/phase4a/callbacks.py`, `my_work/phase4b/callbacks.py` | Standard epoch-end checkpoint alignment fixed; optional lifecycle limits above remain. |
| Both `scripts/run_ablations.py` | Train and score now merge common overrides before axis overrides; conflict test trains and scores history 10 over common history 5. Scorer receives run name. |
| `phase4a/configs/ft_decisions.yaml` | Eight h5 knob combinations; correct common/axis precedence; still evaluates every candidate because `run_score: true` (statistical reviewer owns holdout-use recommendation). |
| `phase4a/configs/ft_decisions_flat_long.yaml` | Three h10/h15/h20 runs; batches 1024/256/128 with accumulations 4/16/32; each product4096; score receives same batch/history. |
| `phase4a/configs/ft_decisions_timedelta.yaml` | Two on/off runs at h10; no scoring; all fixed knobs are explicitly prior-result placeholders requiring reconfirmation. OmegaConf interprets YAML keys `on`/`off` as booleans, making run labels `time_delta-True`/`time_delta-False`; values correctly toggle the feature. |
| `phase4a/configs/ft_decisions_capacity.yaml` | Three width/depth choices32/3,48/4,64/6, all widths divisible by eight heads; no scoring. Prior winner placeholder is acknowledged. |
| `phase4b/configs/ft_decisions_hier.yaml` | Two h5/h10 runs, batch4096. |
| `phase4b/configs/ft_decisions_hier_h16.yaml` | Despite filename, h15/h20; batch2048×accum2; score receives2048. B×T is32768/43008 as comments imply. CPU review cannot verify which CUDA kernel/bank GPU hits a grid limit or prove these settings fit memory. |
| `phase4a/configs/sequential.yaml`, `phase4b/configs/sequential_hier.yaml` | Default eval expanded to8000–8010; EMAfalse; old base loader still wired into training, owned by data reviewer. |
| Both inherited `scripts/train_sequential.py` | Uses full-stream metrics callback before CSV/checkpoint callbacks; loss/model construction delegated to model review. Seeds set. No config-to-resume path. Current launch has no explicit GPU device count or distributed metric gathering guarantees; multi-GPU should remain a separate tested capability. |
| Unchanged `sequential_smoke.yaml` and `tune.yaml` | Inspected as inherited context; smoke intentionally retains8000-only evaluation. Several documentation usage lines still point to nonexistent `fraud/...` variants and should be updated to phase paths; primary changed sweep command comments are correct. |

Equal nominal batch size is useful comparison control; it does not establish exact mathematical equivalence under dropout, floating-point ordering or incomplete final accumulation groups. Neither those figures nor passing CPU lifecycle probes certify GPU throughput, bank schema compatibility, calibration or statistical superiority.
