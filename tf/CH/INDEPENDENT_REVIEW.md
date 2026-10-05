# TF: my record of CH’s independent review of CL’s v2 work

> Public first-person adaptation of documents originally prepared by CH on 2026-10-05. I retain the findings and their original review scope; CH performed the reported independent checks. Source paths are illustrative references to my private project, with `my_work/` representing my work and Ya identifying my collaborator. Source code, private evidence files and bank data are not included in this document set. Later changes require a separate review.

Reviewer: CH. Date: 2026-10-05.

I retain this as the **2026-10-05 v2 review**, with three separately recorded private source identities: the main guide/bridge snapshot, CL’s v2 documents and patch proposals, and CH’s frozen original findings. Exact hashes remain in the private audit. This public copy does not claim that later v3 or other revisions were reviewed.

## Overall assessment

CL's documents identify many real defects and provide useful explanations. The assembled guide contains all ten current parts exactly once and in order. Most underlying audit findings agree with source and CH’s earlier reproductions. The revised patches address several problems found in the first peer review.

**I retain CH’s qualified assessment of these documents and the v2 patch set.** Some explanations contradict code or the guide's own table; some infer experimental conclusions that the available evidence cannot establish. Two EMA patches and two runner patches still have reproducible gaps. Proposed repairs are repeatedly described as already applied, although application source is unchanged at the reviewed main commit.

The independent v2 patch verdict is **7 confirmed within scope, 4 partial**. All 28 existing tests pass on a fresh export after applying the full patch set. Additional checks, rather than that passing suite alone, reveal the remaining issues.

The detailed reports, exact source identities and runnable evidence remain in the private audit. Every verdict applies only to this dated v2 review; later changes need their own review.

## Independence and the collaboration rules

I asked CH and CL to act as **equal, independent verifiers**. CH found no rule giving CL supervisory authority: the private bridge protocol explicitly makes the assistants peers, forbids unilateral assignment and puts evidence ahead of assertion. Original-folder ownership protects each author’s text; it does not give either author a veto over the other verifier’s report.

My collaboration rules are:

- Either verifier can investigate, challenge, publish a finding, propose an alternative or decline an unsupported endorsement without the other’s approval.
- Both acknowledgments are needed to call a recommendation **jointly agreed**, not to publish an independent review or perform an already authorized review. Disagreement stays visible.
- Workboard claims coordinate effort; they do not grant exclusive ownership of an investigation or authority to override my direct instructions.
- I decide goals and business tradeoffs. An unresolved technical claim needs evidence; asking me to choose a reviewer does not settle its scientific truth.
- Author ownership, reviewer independence, joint endorsement and deployment authorization remain separate. This review changed no application code and merged no proposed fix.

In the reviewed v2 recommendations, CL described a broader restriction as jointly agreed: no work from later sections was recommended until all priority repairs were complete, including unrelated pretraining and disabled-EMA work. CH rejected that attribution. The actual decision made prerequisites depend on the experiment. I retain that distinction: measurement planning can proceed, and a valid supervised comparison need not wait for unrelated pretraining repairs.

Suggested replacement: “D-004 requires verified loader coverage, a reproducible environment, an identified evaluation cohort and unique prediction artifacts before new comparative runs. Additional fixes gate only experiments that use the affected path. Measurement planning and independent verification can proceed in parallel.”

## Agreements supported by this pass

- The underlying loader, SupCon, masked-numeric-SSL, checkpoint, evaluation-range, override and reporting defects are real within their documented scope. The 28-finding matrix (private source reference) records agreement and qualifications for every C-ID.
- The flat model concatenates feature tokens across the history window; the hierarchical model first encodes each transaction, then attends across transaction representations. The main diagrams and default shapes are useful.
- Keeping identifiers and historical fraud labels out of predictors, preserving chronological/availability contracts, using full-cohort metrics, recording artifacts, and doing matched comparisons are sound goals. Actual bank-side conformance still needs evidence.
- The v2 causal fixes remove the prior NaN counterexample and pass the expanded CPU checks. The v2 checkpoint save/resume behavior is improved. Loader coverage, SupCon alignment, dependency declarations and packaged config resources check out.
- CL supplied the requested raw reruns and upstream sdist evidence. CH independently compared five JSON outputs with the frozen results: all 210 leaf values agree within `atol=rtol=1e-6`; 14 floating values differ slightly. This supports numerical agreement, rather than literal bit-for-bit equality. The other two text outputs were not included in this numeric comparison.
- The two independently downloaded PyPI sdists match CL's hashes and field lists: `output_cardinality` is absent in the checked 1.1.0 release and present in 1.1.1.

## Material document corrections

The private detailed reviews retain exact quotes, source locations and replacement paragraphs. I use the following high-priority corrections when preparing my project explanation.

| Claim or explanation | Independent assessment | Correct framing |
|---|---|---|
| History's benefit “stands firmly”; every history configuration beats 0.1247; sharding cannot change the direction or fairness of comparisons | Unsupported attribution, plus a direct table contradiction: two history settings score about **0.1215** and **0.1224**. The baseline and history paths also differ in architecture/training. | The best documented sequential runs are promising. Isolate history with a same-architecture h0 control, time-gap ablation, matching data/artifacts and corrected-loader reruns. Rankings are not guaranteed to survive the repair. |
| Selected history-length, mask or optimizer differences are treated as ties/noise using a marginal AP standard error; every one-file score has a ±0.03 margin | A published marginal approximation exists, as acknowledged in CH’s prior correction. Its bank-cohort coverage and the uncertainty of a **paired difference** have not been established. | Label the formula illustrative; do not infer equivalence or no effect. Use shared identified predictions, a justified resampling unit and repeated training seeds. |
| Natural-prevalence softmax training means probabilities are already calibrated; all monotone recalibration preserves ranking metrics | Natural prevalence does not guarantee calibration. Isotonic calibration can introduce ties and change AP/AUROC. | Measure held-out calibration. Strictly increasing transformations preserve ordering and ties; discuss isotonic separately. |
| AP is described as generic PR area and as primarily measuring the top of the ranking | The callback uses sklearn **average precision**. It can differ from trapezoidal PR area and does not specify one operating point. | Define the implemented metric and separately report precision/recall at the bank's alert budget. |
| Baseline and streaming both use frozen files 1–5 artifacts and select on validation AP every epoch | Incorrect for the earlier in-memory baseline: it fits its own imputer/library preprocessing, selects on validation loss and computes ranking metrics after fitting. | Explain the baseline and later streaming/history paths separately. |
| Hierarchical attention grows linearly with history | Stage 1 is linear in transaction count; stage 2 still has quadratic attention over a much shorter sequence. | Approximate attention cost is `T*(F+1)^2 + (T+1)^2`, with widths/layers held fixed. Cheaper does not mean wholly linear. |
| Pretraining happens before labels are seen | True of the masked objective, not SupCon. SupCon explicitly uses fraud labels. | Define masked self-supervised and supervised contrastive pretraining separately. |
| Both pretraining bugs are fixed; loader fix already applied; Phase 5 never ran | Main application source is unchanged. SupCon has a verified candidate patch; masked SSL still needs an implemented, verified correction. Missing run artifacts cannot establish no bank-side run. | Distinguish identified, proposed, independently checked, applied and operationally verified. Say no Phase 5 result is documented in the reviewed snapshot. |
| History is strictly earlier in timestamp; every transaction is seen exactly once; smoke padding rate generalizes to h16 | Equal-time rows can precede the target by ID; availability ordering needs a contract. Exactly-once traversal depends on sharding and stride. The h5 smoke accounts were deliberately selected. | State timestamp/tie-order assumptions and limit smoke results to their actual sample/history setting. |
| Broken causal attention is equivalent to bidirectional, cosmetic, or cannot change the target score | The attention graphs still differ. A same-weight synthetic counterexample changes logits by about **0.1451 flat / 0.04469 hierarchical**. This says nothing about bank AP magnitude. | The CLS relay breaks strict history-state causality; it does not prove equivalence or explain the observed AP gaps. No post-target transaction or target-label leakage was established. |
| Re-vendoring 1.1.1 also resolves modern Torch checkpoint loading | Independently inspected 1.1.1 still omits `weights_only` in its helper. | Constructor compatibility (C-10) and checkpoint loading (C-11) need separate fixes and acceptance checks. |
| FT-Transformer fidelity requires the first block to be post-norm; per-feature linear projection alone is PLR | Author reference code omits the **first attention prenormalization**, not all first-block prenorm. PLR also includes ReLU. | Match the exact reference architecture if making a fidelity claim; otherwise clearly name the implemented variant. |
| Passing imports/tests certifies all screenshot transcription and bank execution | Those checks establish neither full pixel fidelity nor all runtime paths or a clean install. | State what was parsed, imported and executed, in which environment, and which source images/run artifacts were unavailable. |

Detailed guide corrections: Parts 03–05 and CL's project guide (private source reference); Parts 00–02 and 06–09, plus assembly (private source reference). Detailed audit/recommendation corrections: all 28 findings and cross-document issues (private source reference).

Other qualifications in those reports include numeric channel counts, the still-active dropout in frozen-head training, independently maximized AP/AUROC, sampled padding rates, unverified latency/throughput claims, and stale machine-readable patch status. Proposed stacking needs out-of-fold or time-respecting base-model predictions and untouched meta-model evaluation. Suggested cost budgets and stop thresholds are planning heuristics, not measured costs or derived statistical tests.

## V2 patch verdicts

All paths below are illustrative patch names from CL’s private v2 proposal set reviewed on 2026-10-05. “Confirmed” is a scoped correctness finding, not a clean-install, full-size GPU or deployment certificate.

| Patch | Verdict | Evidence |
|---|---|---|
| `fraud_data_parquet_loader.patch` | **CONFIRMED** | Actual 100-file synthetic reads have no omissions/duplicates over three epochs in tested 0/4-worker settings. Persistent-worker order repeats as now documented. |
| `fraud_data_sequence_builder.patch` | **CONFIRMED** | Same independently repeated actual-file coverage result. |
| `fraud_sequential_model.patch` | **CONFIRMED** | Prior padded NaN counterexample fixed; causality, padding isolation and finite-output checks pass in tested CPU modes. |
| `my_phase4a_sequential.patch` | **CONFIRMED** | Same for flat A. |
| `my_phase4b_sequential_hier.patch` | **CONFIRMED** | Same for hierarchical B. |
| `my_phase5_pretrain_supcon.patch` | **CONFIRMED** | Independent sample/view loss and gradient reference still agrees. |
| `pyproject.patch` | **CONFIRMED** for identified corrections | Built wheel declares `requests`, includes 47 Python files from my work and 15 config resources; imports/resource reads pass from a fresh installed target using existing dependencies. Full clean dependency resolution is still untested. |
| `my_phase4a_callbacks.patch` | **PARTIAL** | Save/resume now works; final partial accumulation groups can complete an optimizer step without an EMA update. |
| `my_phase4b_callbacks.patch` | **PARTIAL** | Same newly reproduced accumulation gap. |
| `my_phase4a_scripts_run_ablations.patch` | **PARTIAL** | Effective prediction-root overrides work. Full output roots with identical basenames, and repeated executions, still collide. |
| `my_phase4b_scripts_run_ablations.patch` | **PARTIAL** | Same remaining output-identity gap. |

The rewritten apply-all helper is **confirmed for clean-export application and preflight-failure nonmutation**. It applies all 11 patches; an incompatible final patch leaves every file unchanged. It is not documented as an idempotent successful no-op or an atomic rollback system, and this review does not claim either.

### New EMA counterexample

In actual Lightning 2.6.6 CPU training, both callbacks with three batches and `accumulate_grad_batches=2` finish with EMA **0.10**, whereas updating at each observed optimizer step gives **0.19**. With accumulation 4, the final optimizer step occurs but EMA remains **0**, versus **0.075** expected. The modulo-based callback guard misses incomplete groups. A four-batch, accumulation-two control passes.

Update from actual optimizer-step completion, and test a partial final group. EMA remains disabled in checked-in defaults; this is not evidence that the documented historical scores used broken EMA.

### Remaining prediction collision

The patched namespace uses `basename(output_root)`. Experiments at `experiment_one/same_grid` and `experiment_two/same_grid` both produce `s3://<bank-storage>/axis/same_grid/history-h16`. Repeating an execution reuses that path too. No S3 writes were made in these checks.

Use full experiment identity plus a run/attempt identifier, or an explicit policy distinguishing deliberate overwrite/resume from a new run. Distinct configured A/B roots are not alleged to collide merely because this counterexample exists.

### Causal fix: correctness improves; GPU cost remains unmeasured

All 144 prior model cases and 15 added execution cases pass, including CPU bf16 and unusual padding layouts. The revised mask explicitly allocates a dense batch-by-head attention mask. Benchmark the intended GPU batch/history/precision before asserting memory or throughput readiness. The baseline already creates a dense floating merged mask internally, so its entire cost must not be attributed to v2.

Also correct the “own zero embedding” explanation: padded input values are zero, but embeddings and positions need not be. The safety property is that real queries/CLS cannot read padded keys, while padded query rows retain a valid self-key.

Patch detail: models (private source reference), EMA/runners/helper (private source reference), data/packaging/version evidence (private source reference).

## Validation and remaining limits

The full 11-patch suite independently reports **28 passed, 111 warnings in 3.73s**. The models' smaller suite reports 13 passing tests. Additional executable evidence covers causal/padding boundaries, loss/gradient alignment, actual Parquet coverage, real Lightning save/resume/accumulation, runner command construction, installed-wheel resources, guide assembly and metric counterexamples. Public upstream source archives were inspected without executing their build hooks.

No bank datasets, S3 objects, GPU workloads, current pod jobs, original photographs or private internal model-review materials were inspected in this pass. Recorded bank figures remain documentary observations. CH did not verify every bibliography entry or every allowed dependency version in this pass. The review does not silently convert a proposal, a configured command or a passing synthetic test into a completed bank-side experiment.

I keep both assistants’ independent findings available whether or not they agree. CL can respond to or challenge every item. Corrections belong in subsequent review entries; the original findings remain frozen.
