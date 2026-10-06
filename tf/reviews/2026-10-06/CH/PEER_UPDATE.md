# CH — late peer-update review, October 6

**My latest update:** Both of my bank training scripts use their Phase 4A/4B sequence builders; CH has verified that GitHub still contains the old imports. I need to synchronize those files so the repository reproduces my bank setup. I have created h0 and plan to run it; there is no completed h0 result yet. CL also recorded that my corrected knob finder is still running and AdamW is my chosen fidelity baseline. The newer plan table remains provisional.

> This is my public project copy, prepared by CH. CH performed the independent verification; CL remains the author of the proposals being reviewed. Paths and commands are illustrative aliases. Executable evidence and patches remain in the private record.

CL published a new review and candidate patches while CH was completing the [application review](REVIEW.md). This addendum reviews the **35 additional new/changed files** through `private-reviewed-snapshot`. There are **63 unique changed files** since the preceding October 5 review. The actual application source remains byte-identical to `private-reviewed-snapshot`; new proposals are not applied fixes.

## Latest test result: the repository-wide command now fails

The original application suite passed **30 tests**. On the newer peer snapshot, unrestricted `python -m pytest -q` reports **31 passed, 2 failed, 115 warnings**. Both failures come from the standalone proposed `bridge/proposals/CL_2026-10-06/test_wiring_and_sharding.py`, which pytest discovers before the proposals are installed:

1. Its path calculation assumes installation under `my_work/tests/`. From its proposal location it looks for the nonexistent `bridge/my_work/.../train_sequential.py`.
2. Its order-variation assertion requires an optional, unapplied per-epoch shuffling patch. Current fixed-seed builders already pass the partition/coverage condition when called directly.

This is a real failure of the current default repository test command, not a newly demonstrated model-forward regression. Store proposed tests in non-collected form until installed, or deliberately scope default discovery to active test directories while retaining explicit proposal verification. A copied test must not silently validate the wrong root. The raw latest pytest output is retained in the private evidence record.

## Independent proposal verdicts

| Proposal | Verdict | Evidence and limits |
|---|---|---|
| Both training import changes | **Confirmed within tested scope** | Correct the reproduced active-path loader defect. Applied only to a disposable export; actual factories preserve exact target coverage. |
| Both per-epoch worker-seed changes | **Confirmed for tested coverage; optional enhancement** | A/B × zero workers/nonpersistent four/persistent four × three epochs × two repetitions: 36 exact-coverage checks pass. Nonpersistent workers change file order; zero/persistent workers retain it. Persistent workers use their initial base seed, not the `shard_seed` fallback claimed in the comment. |
| Installed wiring/sharding/causal regression tests | **Useful with dependencies** | Three pass on the relevant patched tree; two fail on unpatched application control, one fails with imports-only. The remaining imports-only failure is optional order variation, not broken coverage. Standalone proposal collection has the separate failure described above. |
| Both EMA callback patches | **Partial** | Epoch-end checkpoint content, accumulation, within-epoch restore and simple resume improve. A real resumed AdamW run with accumulation two still performs an extra EMA update before any resumed optimizer step. |
| Revised EMA unit test | **Passes but incomplete** | Twelve model tests pass on the disposable EMA-patched tree. The test does not cover the resumed-accumulation counterexample; one old restore-timing comment is stale. |
| Proposed h0 control config | **Sound starting comparison, not an executed result** | Six rows: h0/h5 × seeds42/43/44, AdamW, linear, bidirectional, train1–10/validation4950–4954, validation-only selection. It remains a proposal; no bank result or paired validation-prediction export is established. |

The seed enhancement is not necessary to repair missing/duplicate file coverage; CH's two-import candidate deliberately stays narrower. Tests and patches may apply textually independently but have behavioral acceptance dependencies.

### Remaining EMA counterexample

In both proposed callbacks, `on_fit_start` initializes the last-seen optimizer-step counter while Lightning's global step is still **0**. Lightning later restores it to **3**. The first resumed microbatch performs no optimizer update, but the callback sees the apparent step change and updates EMA anyway. It moves the shadow from **0.20802104** to **0.24965963** before a new optimizer step.

The uninterrupted versus resumed experiment finishes with identical raw model weights (**0.564701438**) and AdamW state, but EMA differs: **0.475807726** versus **0.481012523**. A test-only synchronization of the counter at `on_train_start`, after restoration, eliminates the discrepancy in both phases. This is a scoped candidate repair, not a change applied to source.

CL's supplied ten-check log contains eight fresh accumulation checks and two restored dtype/shadow-value checks. Those do not establish actual resumed-training equivalence. CH's stronger resumed-accumulation probe explains why both statements can coexist: the supplied checks pass, yet the new interaction fails. EMA remains disabled in current configs, so this does not block corrected EMA-off supervised controls. [Detailed independent patch review](details/PEER_EMA.md).

## Agreements and corrections to CL's new review

We independently agree on the active training-import problem, causal/padding repairs, standard EMA checkpoint improvement, repeated-run evidence risk, missing executed controls, timestamp/architecture wording, and the limits of CPU verification. The proposed seeded h0 experiment is useful.

CH does not agree with several stronger claims:

- **No universal 0.01 AP rule is supported.** The claim that every smaller single-seed difference is unresolvable has no demonstrated uncertainty calculation. Report conditional paired uncertainty, training variation and a prespecified practical margin.
- **The finder is not yet verified “under fixed code.”** CL's R16 contradicts its own R01 until training wiring and bank run provenance are established.
- **h16 and per-epoch reshuffling are optional.** Fresh matched A/B h10/h15/h20 experiments satisfy the stated plan. Historical A h10 already exists; h16 is not the only prior long-history result.
- **Rerun/failure accounting is not mere hygiene.** CH reproduced stale AP ranked after failed training. Prioritize trustworthy records before interpreting new comparisons.
- **Keep dated result families.** A newer finder can coexist with an older table; neither must be erased. AdamW is an explicit fidelity baseline, and a RAdam challenger is an optional measured follow-up.
- **Scope evaluation and class-weight claims.** Eleven-file defaults apply to the principal Phase 4 configs, not every smoke/Phase 5 config. Weighted Phase 4 round trips pass; remaining class-weight loading defects affect specific original/Phase 5 paths.
- **Photo fidelity is still unverified here.** Matching line counts, imports and tests do not independently compare the original photographs. GPU kernel explanations also remain hypotheses until reproduced on the relevant stack.

The updated guide/business notes correctly distinguish old one-file scores from the new eleven-file defaults. Some other status prose remains historical; a current-state addendum is preferable to treating a new assembly timestamp as a fresh runtime certificate. Detailed [peer-document review](details/PEER_DOCUMENTS.md) and [data-patch review](details/PEER_DATA.md) retain file references and scoped checks.

Application fixes remain unapplied. This addendum does not establish full acceptance of the new patch set, exact bank execution, a statistical win or production readiness. The [fix queue](NEXT_STEPS.md) remains valid, with proposal-test discovery added as a repository hygiene/CI repair and the new EMA interaction retained under its optional-feature gate.

## Subsequent owner answers and final scope

CH also reviewed all three document updates through the latest reviewed documentation update: the peer review addenda A1/A2, question answers and thread 003 messages [3]/[4]. They introduce no application changes and no additional unique paths; coverage remains 63 files. The bank imports are now reported corrected, h0 is created/pending execution, the corrected finder is still running, and AdamW is my deliberate baseline. Earlier snapshot statements about missing imports or missing configured controls must be read with those updates. No current bank run is invalidated solely by the GitHub import mismatch.

One additional statistical correction: CL's suggested h0 threshold of about **0.135** cannot establish that most lift comes from the encoder. Compare contemporaneous h0/h5 (and the original baseline for encoder attribution) on identical cohorts, with paired differences and uncertainty. Three same-direction seed pairs are descriptive evidence, not a significance test; an absolute historical AP threshold does not identify a causal contribution.
