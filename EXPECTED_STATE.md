# What ChatGPT and Claude think the pod looks like

This is the last known state of the pod, as ChatGPT (CH) and Claude (CL) understand it from every record and returned report up to 1 November 2026. Each item has an ID.

Copilot's job is to compare the real files with this list and report only what differs. The rules for the reply are in the "How to answer" section at the end.

Pod project root: `/home/jovyan/k194855/tabular_fraud` (all paths below are relative to it). Python environment: `/home/jovyan/k194855/tabular_fraud_s3_venv_FAIfNI`. One H200 GPU. Data and preprocessing artifacts are in S3, not on the pod.

## X. Root layout

| ID | Path | What we expect |
|---|---|---|
| X01 | `waqas/` | Waqas's code: `base/`, `phase4a/`, `phase4b/`, `v2/`, `scoring/`, `source_identity.py` |
| X02 | `fraud/` | Yaqun's code, identical to his `main` and `yw_seq` branches |
| X03 | `src/` | Yaqun's vendored library (pytorch_tabular), identical to his branches |
| X04 | `runbook/` | Launch tools, including `runbook/2026-10-07/` (V1) and `runbook/2026-10-19/quick_checks.py` |
| X05 | `tests/`, `docs/` | Yaqun's tests and design documents |
| X06 | `ablations/` | V1 run folders: the 15 selected attempts (M01 to M15), plus retries and failed attempts |
| X07 | `runs/` | `run_log.csv`, `returns/`, `install/`, `v2/` |
| X08 | `reviewer_packages/` | Package zips and folders, including `10_21_2026_009/` |
| X09 | `.tf_install`, `tf_package` or `tf_packages` | Package install leftovers |
| X10 | Root scripts | `13_h20_sliding.py`, `seed_robustness.py`, `cpu_reads.sh`, `patch_train_v2_sliding.py`, `patch_target_ending_training.py`, `patch_chunk_training_route.py`, plus `.out`/`.log` files |
| X11 | `v1_analysis_outputs/` | V1 run table and reports (`run_table.csv`) |
| X12 | `test.yaml` | Root file, never read by our tools |
| X13 | `pod_reorg_20261027_v3.zip`, `tools/organize_pod.py` | The clean-up tool; our record says "plan" was never run |
| X14 | `models/`, `shared/`, `archive/` | Should not exist yet, because the clean-up tool was never applied |

## K. Code

| ID | Item | What we expect |
|---|---|---|
| K01 | `fraud/`, `src/` | Identical to Yaqun's own `main` and `yw_seq` branches. Package 009's install comparison found 10 `fraud/` and 48 `src/` files that differ from the GitHub repository copy but match his branches |
| K02 | `waqas/base/`, `waqas/phase4a/`, `waqas/phase4b/` | Our corrected copies as in V1 (package 005/006). `phase4b/` holds all-position B (`ALL_POSITION_CONTRACT.md`) |
| K03 | `waqas/v2/` | The files package 009 installed, exactly as listed with hashes in `runs/install/009_20261021_180056_UTC_5ctj616u/INSTALLATION.json`, plus the patches K04 to K06 |
| K04 | `waqas/v2/scripts/train_v2.py` | Patched once by `patch_train_v2_sliding.py` (target-ending windows during training) |
| K05 | `waqas/v2/models.py` | Patched twice: `patch_target_ending_training.py` (final hash starts `d206e62c`), then `patch_chunk_training_route.py` (final hash starts `6134ccc3`) |
| K06 | `waqas/v2/scripts/select_splits.py` | Patched by `patch_select_splits_ds_train.py` (DS_TRAIN uses files 4001 to 4010); receipt in `runs/v2/patches/select_splits_ds_train_20261022T034405Z/` |
| K07 | `waqas/source_identity.py` | Replaced by package 009 |
| K08 | Two older copies | `waqas/base/data/sequence_builder.py` (no strict same-second rule) and `waqas/base/scripts/score_eval.py` (can overwrite predictions); neither ran in V1 or V2 |
| K09 | `runs/v2/env.sh` | Environment variables for V2 runs |

## M. Models and runs

Test 1 / DEV_OOT is evaluation files 8000 to 8010. SELECT is files 8206 to 8216 (June targets). AP is shown to 4 decimals.

| ID | Model | Folder | Recorded AP (Test 1) |
|---|---|---|---|
| M01 | REF-SINGLE | `ablations/ref_single/ref_single_seeds/REF-SINGLE-53e82983...` | 0.0925 |
| M02 | A-H00 | `ablations/arch_a_flat/history_h0_vs_h5/A-H00-985f76d5...` | 0.0816 |
| M03 | A-H05 | `ablations/arch_a_flat/history_h0_vs_h5/A-H05-d836626f...` | 0.0728 |
| M04 | A-H10 | `ablations/arch_a_flat/history_h10/A-H10-33a2d72e...` | 0.0824 |
| M05 | A-H15 | `ablations/arch_a_flat/history_h15/20261013T044303Z-4bcdf2dd...` | 0.0928 |
| M06 | A-H20 | `ablations/arch_a_flat/history_h20/20261013T044636Z-57c3b2c8...` | 0.0817 |
| M07 | A-GAP05-OFF | `ablations/arch_a_flat/time_delta_off_h5/A-GAP05-OFF-2f38276b...` | 0.0767 |
| M08 | A-GAP10-OFF | `ablations/arch_a_flat/time_delta_on_vs_off_h10/20261013T050127Z-0416d54e...` | 0.0808 |
| M09 | A-MASK05-C | `ablations/arch_a_flat/mask_causal_h5/A-MASK05-C-b398041f...` | 0.0779 |
| M10 | B-H00 | `ablations/arch_b_hier/history_h0_scored/B-H00-7815ffe1...` | 0.0822 |
| M11 | B-H05 | `ablations/arch_b_hier/history_h5_h10/B-H05-e5ea8664...` | 0.0674 |
| M12 | B-H10 | `ablations/arch_b_hier/history_h5_h10/B-H10-8a5832cc...` | 0.0904 |
| M13 | B-H15 | `ablations/arch_b_hier/history_h15_h20/B-H15-4d71f33d...` | 0.0955 |
| M14 | B-H20 (Arch B H20 sliding, seed 42) | `ablations/arch_b_hier/history_h15_h20/B-H20-1dfa4d00...` | 0.0955 |
| M15 | B-ALLPOS-H5 (first chunk model) | `ablations/arch_b_hier/allposition_strict_h5/B-ALLPOS-H5-8e61fcd6...` | 0.0927 |
| M16 | ALLPOS-H20-CHUNK (Arch B H20 chunk, chosen, seed 42) | `runs/v2/L3/ALLPOS_H20_CHUNK/` | 0.1020 (SELECT 0.1103) |
| M17 | ALLPOS-H20-SLIDE | `runs/v2/L3/ALLPOS_H20_SLIDE/` | 0.0743 (SELECT 0.0615) |
| M18 | ALLPOS-H5-SLIDE | `runs/v2/L3/ALLPOS_H5_SLIDE/` | 0.0866 (SELECT 0.0923) |
| M19 | Arch B H20 chunk, seed 43 | `runs/v2/seed_robustness/ALLPOS_H20_CHUNK_seed43/` | 0.0912 |
| M20 | Arch B H20 chunk, seed 44 | `runs/v2/seed_robustness/ALLPOS_H20_CHUNK_seed44/` | 0.1013 |
| M21 | B-H20, seed 43 | `runs/v2/seed_robustness/B_H20_seed43/` | 0.0743 |
| M22 | B-H20, seed 44 | `runs/v2/seed_robustness/B_H20_seed44/` | 0.0753 |

The folder names end with a long ID; only the first 8 characters are shown here. Every V1 folder (M01 to M15) has `ablation_spec.yaml`, `results.csv` and a `history-h<N>/` subfolder with `best.ckpt`, `last.ckpt`, `resolved_train.yaml`, `run_manifest.json`, `training_metrics.csv`, `train.log` and `post_score/` or `manual_score/`. M13 and M14 were scored by hand; M14's score is `manual_score/20261013T142910Z/`. V2 folders (M16 to M22) hold `best.ckpt`, the as-run config, `resolved_train_v2.yaml`, `run_manifest_v2.json`, `training_metrics.csv` and `score_*` receipts.

## S. Shared results and records

| ID | Path | What we expect |
|---|---|---|
| S01 | `runs/v2/L3/allpos_h20_chunk.yaml`, `allpos_h20_slide.yaml`, `allpos_h5_slide.yaml` | As-run L3 configs |
| S02 | `runs/v2/L3/scores_slide/` | Score receipts and predictions for L3 and the V1 anchors |
| S03 | `runs/v2/L3/paired_inputs_followup/`, `paired_DEV_OOT_followup/`, `paired_SELECT_followup/` | Paired comparisons behind the presented results |
| S04 | `runs/v2/L3/measure/` | `window_rows_h20.json`, `window_rows_h5.json`, `chunk_widths_h20.json` (caps 32 for H20, 8 for H5) |
| S05 | `runs/v2/L3/logs/` | L3 training and scoring logs |
| S06 | `runs/v2/arch_a_scores_retry3/` | Final Arch A rescore: `individual_scores.csv` with 24 rows (8 models, 3 populations). `runs/v2/arch_a_scores/` is a failed earlier attempt |
| S07 | `runs/v2/step1/L2_selected/`, `runs/v2/L2_inventory/` | Population split and file inventory (inventory hash starts `9c3f54e9`) |
| S08 | `runs/v2/data_look/`, `runs/v2/step0/`, `runs/v2/step0_logs/` | Data look `data_look_20261021T231206Z` and step 0 logs |
| S09 | `runs/returns/quick_checks_20261021T232641Z/`, `quick_checks_20261021T235529Z/` | Two quick-check returns that complement each other |
| S10 | `runs/v2/patches/` | Patch receipts with before and after hashes |
| S11 | `runs/install/` | One folder per installed package, each with `INSTALLATION.json` and `REPLACED_FILES.zip` |
| S12 | `runs/run_log.csv`, `runs/returns/parallel_v1_005_all/`, `runs/returns/analysis_<UTC>/`, `runs/paired_report*` | V1 run log, returns and paired reports |
| S13 | `waqas/scoring/allpos_sliding_h5/<run>/`, `LATEST_RUN`, `LATEST_RUN.json` | The sliding rescore of M15 |
| S14 | S3 `predictions/sequential/` | Arch A rescore predictions (UUID file names), not `predictions/arch_a_holdout/` |

## P. Plans and state

| ID | Item | What we expect |
|---|---|---|
| P01 | Clean-up | `organize_pod.py` version 3 delivered; "plan", "apply" and "verify" never run; no `shared/reorg/` files. The owner also moved, renamed and deleted files by hand, which we have no record of |
| P02 | Running jobs | Nothing training or scoring |
| P03 | Next work (not started) | Next feature model: drop month, keep time gap, hour, day of week and day of month, add new features; then 100-file training; then downstream use of the embedding (15298 offset model, correlation with TFM V22B) |

## How to answer

Write `handoff/SEND.md` as a list of differences from this file only:

1. **Changed items.** One line per ID that is not as we expect: `ID: moved to <path>`, `ID: not found`, `ID: different (<what>)`, or `ID: partly there (<what is missing>)`. Items that match get no line. End with one line counting how many items matched exactly.
2. **New items.** Anything on the pod that is not on this list, at folder level: path, size, file count and a one-line role. Name individual files only for code, configs and receipts.
3. **Code differences.** For each code file that differs from its known version: path, closest known version, and which functions changed. Include a diff only if it is 10 lines or fewer.
4. **Moves.** Old path to new path, at folder level, with same or different bytes.
5. **Questions**, each with why it matters.

**Limits:** at most 300 lines, each at most 90 characters, with a line `--- page N of T ---` every 40 lines.

Keep all full details (complete lists, diffs, full file text) in `handoff/detail/`, under the same IDs, for later requests.
