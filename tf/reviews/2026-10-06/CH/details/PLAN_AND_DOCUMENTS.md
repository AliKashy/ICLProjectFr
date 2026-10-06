# October 6 documentation and experiment-plan review

**Status qualification:** This detailed report tests the reviewed GitHub snapshot. My later update says the bank trainers already use the corrected phase-owned builders; that makes the import finding a repository sync gap. My h0 config is created, results are pending, and the corrected knob rerun table is provisional. See the [current status](../README.md).

> Public first-person project copy prepared by CH on 2026-10-06. I keep my original findings and guides unchanged. CH performed the checks reported here. Source paths and commands are illustrative private-project aliases; code, evidence programs and the candidate patch remain in the private review record. Exact approved scientific details are retained.

Reviewed snapshot `October-6-reviewed-source` against `previous-reviewed-source`. Source was read-only. Scope: all 16 changed Markdown files plus all 8 changed/new YAML files. Static OmegaConf expansion covered all 20 sweep configurations. The script and machine-readable resolved values are `inspect_configs.py` and `effective_configs.json`; changed-file accounting is `inventory.json`. No bank data, training jobs, S3, or GPU were accessed by this reviewer.

## State that is actually supported

The repository now contains updated A/B model/callback/scoring code, separate phase-owned sequence builders, 11-file evaluation defaults for both principal phase configs, and explicit long-history/time-delta/capacity sweeps. This is implementation progress, not an end-to-end functional or statistical certification.

The downstream optimizer decision and actual YAML agree: AdamW throughout A long history, B short/long history, time delta and capacity. The h5 finder deliberately still compares AdamW/RAdam (8 combinations). This is not an AdamW-only contradiction: the statement is downstream-only. Config comments suggesting updating all winner knobs if the finder changes should explicitly keep the fidelity baseline AdamW and treat a RAdam challenger separately.

A long history is h10/h15/h20; B main is h5/h10; B long is h15/h20, stored under `ft_decisions_hier_h16.yaml`. That filename is deliberately preserved by reconstruction/HANDOFF.md:44. The overview reports a later local rename to `ft_decisions_hier_long.yaml` but explicitly marks it chat state (Overview:134–136); no such filename exists in this snapshot. Use the existing filename until a deliberate rename is made.

The latest detailed bank-result table remains the October 4 23:09 snapshot in `my_work/docs/my_work_current_standing.md`. The October 6 overview/plan adds a newer reported finder summary (linear .1260 versus periodic .1202; AdamW .1189 versus RAdam .1273), but contains no per-run rows, source commit/config digests, checkpoints, or execution logs with which to verify the result. Preserve it as a reported later observation. Do not merge it with the older finder (.1365 best row) or call it a reproduced clean rerun of this GitHub snapshot.

The October 5 bridge answer resolves the old sweep evaluation range: those recorded eval scores were one-file part_8000 scores, according to the attributed user answer. Changing defaults today to 8000–8010 does not retroactively rescore those checkpoints or turn the old scores into eleven-file results.

## Findings and actionable gaps

### PD-01 — P1: the plan says the loader fix is incorporated, but training does not use it

- Claim: `docs/phase4_sequential_plan.md:20–23`, component table `:234/:240`, finder comment `my_work/phase4a/configs/ft_decisions.yaml:3–7`.
- Actual entrypoints: `my_work/phase4a/scripts/train_sequential.py:27` and `my_work/phase4b/scripts/train_sequential.py:27` import `fraud.data.sequence_builder.SequenceBuilder`, then construct it at `:36`.
- The changed scorers import the phase-owned copies, but scoring has shuffle disabled; this does not fix the training path. Both base configs still use 4 training workers.
- Consequence: running the new comparison configs from this checkout still exercises the old independently shuffled file partition during training. A comment saying “fresh rerun under fixed code” is not evidence of correction.
- Required before new comparative training: wire the intended builder into both entrypoints and run an integration check through the actual loader creation path. Root/data reviewer owns the detailed reproduction.

### PD-02 — P2: new h10 time-delta/capacity controls restore the unsafe old micro-batch

- `my_work/phase4a/configs/ft_decisions_timedelta.yaml:16–26` and `ft_decisions_capacity.yaml:15–25` set h10 but have no batch/accumulation override.
- They resolve to `batch_size=4096`, `accumulate_grad_batches=1` from `sequential.yaml:56/:61`. A long-history h10 explicitly uses `1024 × 4` (`ft_decisions_flat_long.yaml:37`).
- The bank status documents a prior long-history full-batch CUDA OOM and the smaller-batch workaround (`my_work/docs/my_work_current_standing.md:40–42`). The capacity “large” row additionally grows 48×4 to 64×6.
- Static configuration regression is confirmed; whether a particular current GPU/kernel run OOMs was not tested here. Use the known h10 micro-batch schedule for these comparisons, then profile the large row and decrease micro-batch further if needed while retaining the same effective batch. Compare time delta on/off at the same safe micro-batch rather than changing memory settings opportunistically.

### PD-03 — P2: the configured experiment set still omits the no-history control and repeated seeds

- The plan explicitly requires `history_len` 0 versus 5 (`docs/phase4_sequential_plan.md:290–291`). None of the 20 expanded changed sweep rows has history 0; all set seed 42.
- h0 is technically configurable, so this is a missing comparison in the proposed run order, not a model inability. The overview execution order `:169–176` goes directly to longer histories/B/delta/capacity/100 files.
- Without same-architecture h0, observed lift over the library baseline does not isolate history from encoder changes. Without repeated seeds/paired uncertainty, “effectively tied” (plan:55) and “N=5 lift ... real” (plan:319–320) are overstrong.
- Prioritize h0/h5 under the same corrected pipeline before expanding h15/h20 or pretraining. For a stricter context test, optionally include target-only/masked-history at a fixed slot layout so positional/table-size differences are also controlled.

### PD-04 — P2: the plan's strict time contract and the implemented equal-time ordering differ

- Plan `:103` promises history time `< target_time`; verification `:292` requires excluding any time `>= target_time`.
- Its own construction also says timestamp plus transaction-ID tie-break (`:87`). The builder accepts preceding rows under that lexicographic order, which can include equal timestamps; the changed business guide correctly says this explicitly (`business_guides/CL/README.md:77–80`).
- Confirm bank-side availability order for ties or implement strict timestamp exclusion. Until then describe the contract as ordered preceding rows, not proven strictly earlier observations. Root/data reviewer owns code reproduction.

### PD-05 — P2: preserve cohort comparability when scaling, and handle correlated observations in uncertainty

- Plan `:121–122` calls correlation a training-only concern because validation/test metrics aggregate over the whole set. Empirical AP for fixed scores is order-invariant; that narrow claim is fine. Account/window dependence still changes uncertainty and effective independent sample size. Use paired account/appropriate time-block resampling, not independent-transaction intervals.
- Plan `:324–327` suggests growing validation as training grows. Expanding validation is reasonable if fixed before comparison, but AP changes with prevalence and cohort mix. Freeze a manifest for the comparison; if widening it, rescore/re-evaluate all controls and candidates on the widened manifest. Do not compare a candidate's enlarged-cohort AP with the old baseline number.
- Retain files 8000–8010 as the matched development reporting cohort. It has been consulted; an untouched later-time cohort with mature labels is still needed for final generalization claims.

### PD-06 — P2: the autoresearch proposal should calibrate against clean controls and retain a genuinely fresh final cohort

- `bridge/proposals/autoresearch_CL.md:30–36` proposes proxy rank correlation against the 8 finder runs “that already exist”; those original runs used the old loader/mask. Compare proxy and full-scale runs from the same corrected implementation/config/seed family before inferring fidelity.
- `:37–38` calls the eleven-file cohort held back from the agent. Agent access can indeed be withheld, but that does not erase previous human development looks. A separately untouched temporal cohort is needed for final claims.
- Other core boundaries are sensible: immutable loader/metric/cohort, all-trial ledger, seeds, cost cap, review of accepted changes, full-scale confirmation. This is explicitly a proposal, not a current runtime defect. Defer until controlled baselines are reproducible.

### PD-07 — P2/P3 operational documentation: current plan uses the wrong buffer knob and entrypoint

- Plan `:117/:273/:308` calls `read_batch_size` the sequence-window buffer. Actual phase training code reads `training.buffer_windows` (`phase4a/scripts/train_sequential.py:51`, same in B), and actual configs use it at A:57/B:58. Adding the plan's `read_batch_size` override will not enlarge the sequence shuffle pool.
- Plan component table `:236` names nonexistent `my_work/phase4a/train_sequential.py`; actual location is `my_work/phase4a/scripts/train_sequential.py`.
- B config `my_work/phase4b/configs/sequential_hier.yaml:4` tells readers to run the original `fraud/` training/config instead of the phase-owned entrypoint; those original paths bypass phase-owned changes.
- Correct these operational instructions before using the plan as a runbook. Raising buffer_windows should follow measured host-memory headroom, not simply scale mechanically with history.

### PD-08 — P3: cleanup and current-status claims exceed the checked-in evidence

- Plan `:43–46` and Overview `:94/:144–154` say Phase 6 paths/docstrings were cleaned. `my_work/phase4a/configs/sequential.yaml:1/:79/:85` still uses Phase 6/phase6_sequential; both training docstrings still say Phase 6. This is documentation drift, not a model failure.
- Overview `:162–164` reports `my_work/docs/phase4_progress.md`, `project_writeup.md`, `meeting_notes.md`; none exists in this snapshot. They may exist on the bank machine; do not claim reviewed/verified.
- `reconstruction/HANDOFF.md:42/:68` explicitly limits recovery checks to syntax/visible source and says no tests/jobs ran. Source-photo fidelity is not runtime verification.
- The Overview's reported test count 22→24 (`:79`) is a chat claim, not a current suite result. Use the separate runtime review's actual collection/pass counts.

## Historical documentation and bridge handling

The six changed guide parts plus the assembled GUIDE consistently incorporate the user's one-file evaluation answer. All 10 part texts are present unchanged in the assembled guide (script assertion passed). This is a valid correction to historical results. GUIDE is timestamped October 5, and the business guide explicitly pins its code snapshot and pod snapshot (`business_guides/CL/README.md:3–6/:377`). Their “not applied”, h16, old test count and one-file-default descriptions are historical, not proof of October 6 state. Preserve audit history; add a clearly dated current-state addendum and links instead of silently rewriting frozen findings.

`bridge/ASK_my_work.md:8–12`, `bridge/errata/CL.md:4–6`, and `bridge/threads/000-kickoff.md:218–222` correctly record attribution and the consequence of the cohort answer. `bridge/STATUS.md:8–9` still says it awaits Q1/the eval range despite that answer; close that particular question and retain unresolved artifact provenance/B results/Phase 5 status. New main code is not necessarily byte-equivalent to CL's proposed patch v3; the historical pending-v3 verification cannot be treated as validation of the current implementation.

The rewritten business guide has sound caution about causality, capacity mismatch, production comparisons and unverified benefits. Its first table's introduction says “same five files for every run” (`:138`) despite rows `:144–146` explicitly using only validation file4950; restrict that statement to the comparison subset. Exact old measured rows remain usable as historical observations, not a new shared-cohort experiment.

## Concrete next experiment sequence aligned with the updated plan

1. **Make the actual execution path reliable.** Resolve entrypoint/builder integration and the model/training/runtime findings; verify CLI train+score with synthetic/local fixture data; save code/environment/artifact identity and transaction/cohort checksums. Keep original source recovery evidence separate from any deliberate repair.
2. **Freeze the protocol.** Pin training1–10, validation4950–4954, AdamW baseline, preprocessing manifest, target availability/time-tie policy, natural label distribution, safe batch schedule, metrics, and run IDs. Confirm account containment/split dates/label maturity. Treat eval8000–8010 as development comparison. Set aside later-time final holdout before more searching.
3. **Rerun matched h0 and h5 controls first.** Same A architecture recipe and data; start with a smoke seed, then at least3 confirmation seeds (5 if feasible). Record per-seed selected-epoch metrics plus identified predictions. Do not use the old library FT number alone as the history control.
4. **Reconfirm optional knobs under fixed code.** If paper fidelity is the decision, AdamW remains baseline. RAdam can be a separately labeled challenger; choose embedding/mask on validation, with the corrected causal path. Do not transplant old winner labels blindly.
5. **Run the planned history/architecture matrix.** A h5/10/15/20 and B h5/10/15/20. Current safe candidates: A1024×4 at10,256×16 at15,128×32 at20; B2048×2 at15/20. Measure actual throughput/RSS/GPU memory. Effective-batch equality alone does not prove capacity/compute matching; report parameters, optimizer steps, examples, and GPU time. Keep h16 old results historical unless specifically retested.
6. **Time-gap attribution and capacity.** Use the chosen history and common safe schedule for on/off. Capacity32×3/48×4/64×6 is a sensible initial fixed-recipe diagnostic. The plan's larger48→64→96/128 and4→6→8 grid/LR/warmup/regularization retuning is a later search, not implemented by the three-row config. Introduce warmup explicitly if desired; merely placing a config key does not implement it. Current model scheduler is ReduceLROnPlateau(valid_loss), not warmup.
7. **Scale the leading controlled candidates.** Run100 files first, then200/500 only if benefit/cost justifies it. Hold validation fixed or evaluate all models on a newly declared common version. Recompute uncertainty with paired observations and repeated seeds. Preserve all run outputs so failures/restarts cannot masquerade as new successful results.
8. **Business confirmation.** Same identified transactions for transformer/control/production-score comparisons; report AP, precision/recall at the bank's actual review budget, fraud-value capture, calibration on an appropriate disjoint split, and latency/resource cost. Evaluate final frozen candidates on untouched time with mature labels. Phase5/autoresearch follow demonstrated reliability and measurable controlled benefit.

No statistical test can certify “everything works.” A strong outcome is a reproducible pipeline, explicitly limited evidence, controlled performance estimates, and a next-run protocol that answers one question at a time.
