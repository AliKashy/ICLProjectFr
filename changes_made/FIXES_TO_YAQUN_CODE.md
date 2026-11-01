# Code changes we made to Yaqun's code to run it (tabular_fraud)

## 1. Defects in Yaqun's code, fixed in our copies

1. Sharding (workers skipped or repeated training files)
   - Problem: each of the 4 data workers shuffled the file list in its own order, then took every fourth file. On average about 31.6% of training files were skipped each epoch and others were read twice (mathematical expectation, not measured per run).
   - Fix: single-transaction copy: each worker takes its own slice first, then shuffles it. Arch A and B: one shared seeded shuffle, then split. V2 uses the run seed.
   - Where: `waqas/base/data/parquet_loader.py`, `waqas/base/data/sequence_builder.py`, `waqas/phase4a/sequence_builder.py`, `waqas/phase4b/sequence_builder.py`, `waqas/v2/scripts/train_v2.py`.

2. Causal mask leak through CLS
   - Problem: every transaction could read the summary token (CLS), and CLS reads every position, so from the second attention layer on earlier positions could see later ones. This broke the causal versus bidirectional comparison. It did not leak labels, and in sliding windows the scored transaction is last, so it saw nothing after itself.
   - Fix: CLS still reads everything; no transaction reads CLS. Padding is merged into the mask so short histories no longer produce NaN.
   - Where: `waqas/phase4a/sequential.py`, `waqas/phase4b/sequential_hier.py`, `waqas/base/sequential_model.py`.

3. Same-second history
   - Problem: a transaction in the same second counted as history if its id sorted first, and ids were compared as text.
   - Fix: option `history_time_order`, default `strict`: history uses strictly earlier times only (Yaqun's own written rule). `tiebreak` keeps his old behaviour. The rule is stored in the checkpoint and scoring refuses the other rule.
   - Where: `waqas/phase4a/sequence_builder.py`, `waqas/phase4b/sequence_builder.py`.

4. Silent bad inputs
   - Problem: a missing fraud label became "not fraud"; an unreadable timestamp became a meaningless number.
   - Fix: the run stops on a missing label, a missing account id, an unreadable timestamp, a numeric timestamp with unclear units, or mixed time zones. It does not reject every unexpected label value. The account check runs only where history windows are built; in the single-transaction loader the timestamp check runs only when the feature list uses time features.
   - Where: `waqas/base/data/input_contracts.py`, used by the base loaders and scripts and by the phase4a and phase4b builders.

5. Time handling
   - Problem: accounts were sorted on the raw time column; cyclical time encoding with missing flags crashed (`KeyError('hour_sin')`).
   - Fix: sort on the parsed timestamps; build time features only when the feature list needs them, before the missing flags.
   - Where: `waqas/base/data/sequence_builder.py`, `waqas/base/data/parquet_loader.py`, `waqas/base/scripts/train_baseline.py`, `waqas/base/scripts/score_eval.py`.

6. Scorer and fraud weight
   - Problem: the scorers rebuilt the model with the default fraud weight, so a model trained with a different weight could not load.
   - Fix: rebuild with the configured weight; the streaming scorer checks its settings before a strict load. (Never mattered in practice: every run used weight 1.)
   - Where: `waqas/base/scripts/score_sequential.py`, `waqas/base/scripts/score_streaming.py`.

7. Prediction overwrites
   - Problem: every run of a sweep wrote to the same prediction file.
   - Fix: each run writes its own prediction file. Base scorers refuse an existing destination and stop on empty, non-finite or out-of-range scores.
   - Where: `waqas/phase4a/scripts/score_sequential.py`, `waqas/phase4b/scripts/score_sequential.py`, `waqas/base/scripts/score_streaming.py`, `waqas/base/scripts/score_sequential.py`.

8. Several GPUs
   - Problem: training could start on several GPUs, but the data is not split per GPU.
   - Fix: training refuses more than one GPU and pins one device.
   - Where: `waqas/base/scripts/train_streaming.py`, `waqas/base/scripts/train_sequential.py`, `waqas/phase4a/scripts/train_sequential.py`, `waqas/phase4b/scripts/train_sequential.py`.

9. Weight averaging (EMA)
   - Fix: weights update once per optimizer step; validation and the saved model use the same weights; raw and averaged weights are both kept for resume. Matters only when EMA is on, and it never was.
   - Where: `waqas/phase4a/callbacks.py` (same in `waqas/phase4b`).

## 2. Fixes to our own code: the all-position model

10. All-position model (every transaction a target)
   - Problem: the first version gave earlier transactions a time gap to the last transaction in the window (a value from the future), used overlapping windows (transactions counted more than once) and allowed same-second history.
   - Fix: the gap is to the latest strictly earlier time; chunks do not overlap and are counted in whole seconds (H20 means 20 earlier time points, which can be more than 20 rows); padded slots are left out of the loss and the loss is averaged over real transactions; position numbers do not reveal same-second neighbours; an optional size check rejects oversized chunks rather than trimming them.
   - Where: `waqas/phase4b/sequence_builder.py`, `waqas/phase4b/sequential_hier.py`, `waqas/phase4b/ALL_POSITION_CONTRACT.md`.

## 3. Experiment settings (not bug fixes)

11. Validation: Yaqun's history-model config used file 4950 only; we use 4950-4954 (from your own first sweep configs). His single-transaction config already used 4950-4954.
12. Evaluation: his configs scored file 8000 only; we score 8000-8010.
13. Batch sizes: long histories use smaller batches with accumulation (for example 1024 x 4, 2048 x 2), keeping the effective batch at 4096. Not guaranteed identical to one batch of 4096; settings are recorded per run.
14. Storage paths point to your folders (`waqas/base/configs/streaming.yaml` and the A and B configs). All other settings kept.

## 4. Keeping results traceable

15. Every run writes a metrics file with its checkpoint's hash.
16. Checkpoint contracts: a checkpoint records its features, artifacts, mode and history rule; scoring refuses a mismatch. In the single-transaction copy, old causal checkpoints from before the mask fix are refused. In A and B, old single-target checkpoints load with a warning; incompatible all-position checkpoints are refused.
   - Where: `waqas/base/data/contracts.py`, `waqas/base/data/checkpoint_contracts.py`, `waqas/base/sequential_model.py`, `waqas/phase4a/factory.py`, `waqas/phase4b/factory.py`, `waqas/phase4b/sequential_hier.py`.
17. Leaderboard: both scores come from the same epoch, and models are ranked by validation AP only, never by the test score (V1 analysis tools in `runbook/2026-10-07/`).

## 5. New V2 code (branch `claude/v2-integration-2026-10-19`, package 009)

18. `waqas/v2/scripts/train_v2.py`: resumes from the last completed epoch on a compatible setup; explicit seed; chunk models validated on one window per transaction.
19. `waqas/v2/scripts/score_v2.py` and `waqas/v2/target_ending.py`: one window per transaction, ending at it; the row cap is set per model from a measurement (32 rows for the H20 model); each scoring run writes a receipt.
20. Guards: a width check on each batch; a refusal to overwrite artifacts, only in the new no-month artifact builder (not in every preprocessing tool).

## 6. Pod-only changes (patch text is in the uploaded reports in the repo; not part of the package code)

21. Three patches: training on one window per transaction (`train_v2.py`); the loss on each window's labelled transaction (`models.py`); chunk batches routed back to the original training step (`models.py`). Scripts: `patch_train_v2_sliding.py`, `patch_target_ending_training.py`, `patch_chunk_training_route.py`.
22. Split patch (`select_splits.py`): widened where downstream training files may come from; 4001-4010 was the block chosen (files 5001-7999 do not exist).

## 7. Operational changes

23. The 4 + 4 GPU queue split for the V1 runs.
24. The hand-restarted B-H15 and B-H20 runs, folded into the V1 analysis.
25. Later helper and config scripts run by hand on the pod, such as `13_h20_sliding.py` and `seed_robustness.py`.

## Known faults in Yaqun's code that we did not fix (none affects the results)

- The training-loss column in his metrics log is misleading (`fraud/callbacks.py`, still used by the copies and V2).
- His baseline training script empties its metrics folder when it saves the model (`waqas/base/scripts/train_baseline.py` keeps it).
- His S3 connection turns TLS certificate checking off (`fraud/data/s3_utils.py`).
- An outdated library default when loading checkpoints, and slow data encoding.

## Two copies that lag behind (neither ran in V1 or V2)

- `waqas/base/data/sequence_builder.py` (his single-transaction history builder copy) does not have the strict same-second rule.
- `waqas/base/scripts/score_eval.py` can still overwrite earlier predictions.

## Open point

- Every V1 and V2 run read preprocessing artifacts version 1 from your storage folder, not Yaqun's. Nobody has confirmed they are an exact copy of his. The original pre-flight script (`runbook/2026-10-07/preflight_pod.py`) prints that comparison.
