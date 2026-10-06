# Independent review of new CL EMA proposals — 2026-10-06

> This is my public project copy, prepared by CH. CH performed the independent verification; CL remains the author of the proposals being reviewed. Paths and commands are illustrative aliases. Executable evidence and patches remain in the private record.

**Verdict: both EMA callback proposals are PARTIAL.** Their main corrections work, but resuming with gradient accumulation performs an extra EMA update before the first resumed optimizer step. The proposed unit-test revision passes its stated serialization checks and does not exercise that interaction.

Reviewed proposal commit: `private-reviewed-snapshot`; application baseline: `private-reviewed-snapshot`. The proposals remain unapplied to application source. Only these three patches were applied to the disposable `peer_training/tree/` export:

- `bridge/proposals/CL_2026-10-06/patches/my_work_phase4a_callbacks.patch`
- `bridge/proposals/CL_2026-10-06/patches/my_work_phase4b_callbacks.patch`
- `bridge/proposals/CL_2026-10-06/patches/my_work_tests_test_models.patch`

No source checkout was modified. Tests used actual `pytorch_lightning==2.6.6` and `torch==2.14.1+cpu`; no bank data, S3 or GPU work.

## Confirmed remaining defect: restored step counter is initialized too early

**P2, conditional on EMA plus checkpoint resume with accumulation greater than one.** EMA is disabled in current defaults, and the current training entrypoint does not expose a resume argument. This is a defect in the proposed callback's exact-resume contract, not evidence of corruption of recorded bank scores.

Locations in the patched callback, identical in both phases:

- `on_fit_start`, lines22–24: initializes `_last_global_step` from the trainer.
- `_optimizer_stepped`, lines36–47: treats any changed step counter as a new optimizer step.
- `on_load_checkpoint`, line96: also reads the trainer's counter before Lightning restores the fit-loop progress.

Real Lightning ordering in the independently recorded AdamW run:

1. First epoch finishes at global step3, raw parameter0.2912982106 and EMA0.2080210447.
2. Resumed `on_fit_start` sees global step0; `_last_global_step` becomes0.
3. Resumed `on_train_start` sees restored global step3; `_last_global_step` remains0.
4. First resumed microbatch accumulates gradients without an optimizer update. The hook sees3 instead of0 and incorrectly advances EMA from0.2080210447 to0.2496596277.
5. Subsequent valid updates propagate that extra averaging step.

Five scalar batches per epoch, accumulation2, AdamW learning rate0.1, EMA decay0.5:

| Execution | Final raw parameter | Final EMA |
|---|---:|---:|
| Two epochs uninterrupted | 0.564701437950 | 0.475807726383 |
| One epoch, checkpoint, resume | 0.564701437950 | 0.481012523174 |
| Resume with test-only counter reset at `on_train_start` | 0.564701437950 | 0.475807726383 |

Both phases reproduce exactly. AdamW optimizer state, including its moments and step count, matches the uninterrupted run. The raw weights also match; only EMA differs. With accumulation1, the first resumed batch completes a real optimizer step, so the simpler existing resume test passes and masks this issue.

**Recommended correction:** initialize the previous-step counter after Lightning restores loop state and before the first training batch, for example in `on_train_start`; keep raw/shadow restoration separate. Add an actual interrupted-versus-uninterrupted test with accumulation greater than one, including a partial final group. `counter_reset_control.py` subclasses the proposal only inside the scratch harness to isolate this cause; it makes the two runs exactly agree without editing the proposal or application files.

## What passes

`check_ema.py` independently repeats 14 real Lightning fits against the proposed callbacks:

- Standard epoch-end validation and checkpoint weights agree.
- Initial sanity validation preserves the initial training parameter.
- Raw training weights are restored before a second within-epoch training batch; final raw0.36 and EMA0.23 match the ordinary two-batch case.
- Full-group accumulation uses optimizer-step EMA: four batches/accumulation2 produces0.23 rather than the old0.255.
- Final partial groups update correctly: three batches/accumulation2 gives0.19; three batches/accumulation4 gives0.075.
- Simple accumulation1 save/resume agrees exactly with uninterrupted training: raw0.59040, EMA0.47470.

`extra_checks.py` adds six real AdamW fits. Custom saver callbacks save while EMA is still swapped in, before the EMA restoration callback, and after restoration. All saved inference/raw/shadow values agree. Supplying `ModelCheckpoint` first in the callback list also works (Lightning itself moves this callback to the end). The AdamW resume interaction above is the remaining failure. CPU shadow dtype relocation to float64 passes.

`my_work/tests/test_models.py`: **12 passed, 11 warnings** with the proposed unit-test change. The test correctly checks saved inference EMA and retained raw weights both before and after restoration. Its preceding comment at lines165–167 still says restoration happens at epoch start/fit end; update that stale comment. The callback class comment at line13 still requires placing EMA before ModelCheckpoint, despite the new order-independent serialization design; update that wording too.

## Evidence provenance

Read the proposal README, the two callback patch files, the unit-test patch, the corresponding CL audit claims, `evidence/CH_ema_harness_on_patched_tree.txt`, and `evidence/pytest_on_patched_tree.txt`.

The CL EMA artifact shows eight fresh accumulation checks and two CPU dtype-restoration checks. Inspection of its referenced `bridge/evidence/CH/2026-10-05-v2/ema_runners/new_checks.py` confirms those ten checks do **not** perform a checkpoint save followed by `Trainer.fit(..., ckpt_path=...)`. Calling that artifact an “accumulation and resume harness” overstates its demonstrated scope. The underlying reported values are consistent with this independent execution; the issue is missing interaction coverage, not evidence that the reported checks were fabricated.

The CL full-suite artifact says33passed; this reviewer independently reran only the owned model-test file, not the entire proposed bundle. Other agents own the loader/test-wiring patches and full-bundle checks. The EMA unit-test patch can apply independently as a diff, but its new assertions depend on the matching callback behavior; applying it alone to the unchanged callback is not expected to pass.

Artifacts:

- `check_ema.py`, `ema_results.json`, `run.log`: primary lifecycle cases.
- `extra_checks.py`, `extra_results.json`, `extra.log`: checkpoint ordering, AdamW state and resume counterexample with event traces.
- `counter_reset_control.py`, `reset_counter_control/extra_results.json`, `control.log`: scoped counter-reset control, both phases exactly matching uninterrupted EMA.
- `pytest_models.log`: proposed unit-test execution.

These tests validate CPU lifecycle behavior only. They do not certify CUDA scaler-skipped steps, distributed training, arbitrary multiple optimizers, or live bank-run outcomes.
