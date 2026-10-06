# TF — my current project review, October 6

**My latest update:** Both of my bank training scripts use their Phase 4A/4B sequence builders; CH has verified that GitHub still contains the old imports. I need to synchronize those files so the repository reproduces my bank setup. I have created h0 and plan to run it; there is no completed h0 result yet. CL also recorded that my corrected knob finder is still running and AdamW is my chosen fidelity baseline. The newer plan table remains provisional.

**My project has improved, but the current GitHub training path still needs a key fix before I can trust new comparisons.** Both trainers still import the original data loader instead of the corrected phase-owned copy. CH reproduced missing and duplicated training targets through the actual entrypoints.

This is CH's independent review of all **63 unique changed files**, including a late peer upload, Ya's updated plan and the surrounding implementation. I keep the earlier CH/CL findings and guides unchanged. These are new public copies using my first-person project voice and the approved identity aliases; CH performed the reported verification. This review does not imply a new CL endorsement.

- [Complete current review](REVIEW.md)
- [Prioritized fixes and experiment plan](NEXT_STEPS.md)
- [All 63 changed files and review coverage](COVERAGE.md)

[Late peer review and proposal verification](PEER_UPDATE.md): the loader proposal passes scoped checks; the EMA proposal still has a resumed-accumulation defect. None is applied.

## Verified progress and test status

- The original **30 application tests pass**. The later peer upload makes the default repository-wide command **31 passed, 2 failed** because it collects proposal-only tests; the [peer update](PEER_UPDATE.md) explains both failures.
- Expanded independent tests support the new causal-attention and padding fixes in A and B.
- Small actual training → checkpoint → scoring runs preserve target IDs, labels and finite scores.
- Override precedence, different-run-name output preservation, and standard epoch-end EMA checkpoint alignment are repaired.
- Current main evaluation defaults select eleven files; downstream AdamW and the planned history axes align with Ya's updated intentions.

## What I need to resolve

- Synchronize both GitHub trainers with my reported bank imports and verify target coverage through those exact entrypoints.
- Preserve separate attempt outputs and prevent failed runs from reporting old metrics.
- Separate historical one-file evaluation scores from the eleven-file reference and newer reported sweep.
- Run my newly created no-history control against matched h5, then add repeated seeds, paired uncertainty and untouched temporal confirmation.
- Correct analysis-tool errors before using their rankings or confidence intervals.
- Validate the exact bank software/GPU environment and data contracts. Fix Phase 5 objectives before pretraining comparisons; optional EMA and multi-GPU need their own checks.

## Detailed reports

| Area | Report |
|---|---|
| Data, loader integration and input contracts | [Data](details/DATA.md) |
| Flat/hierarchical architectures, masking and tests | [Models](details/MODELS.md) |
| Training, EMA, experiment commands and failures | [Training](details/TRAINING.md) |
| Metrics, cohorts, uncertainty and evaluation | [Statistics](details/STATISTICS.md) |
| Every changed document/config and alignment with Ya's plan | [Plan and documents](details/PLAN_AND_DOCUMENTS.md) |
| Packaging, original pipeline and pretraining | [Runtime and pretraining](details/RUNTIME_AND_PRETRAINING.md) |

The private review contains executable synthetic evidence and an **unapplied two-import candidate patch**. Public source-path aliases and commands are explanatory; the private application code is not included here. No bank job, bank dataset or H200 execution was available. Passing CPU checks does not establish statistical superiority or production readiness.

The old evaluation-range question is now answered: the historical sweep used file8000. The [new bank questions](NEXT_STEPS.md#bank-facts-still-needed) concern current source/run provenance, completed results, artifacts, timing and business capacity. I should not treat every pending item in the older question snapshot as still unresolved.
