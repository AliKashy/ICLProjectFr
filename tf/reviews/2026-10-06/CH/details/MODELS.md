# Current Phase 4 model review — 2026-10-06

> Public first-person project copy prepared by CH on 2026-10-06. I keep my original findings and guides unchanged. CH performed the checks reported here. Source paths and commands are illustrative private-project aliases; code, evidence programs and the candidate patch remain in the private review record. Exact approved scientific details are retained.

Reviewed immutable commit `October-6-reviewed-source` against `previous-reviewed-source`, in `<private-checkout>`. Source remained clean. Owned changed files: `my_work/phase4a/sequential.py`, `my_work/phase4b/sequential_hier.py`, `my_work/tests/test_models.py`; also read both tokenizer/numerical-embedding/factory implementations, relevant architecture configurations, and updated design plan. EMA runtime is assigned to another reviewer.

Owned changed-file count: **3/3 reviewed**. Source links: `my_work/phase4a/sequential.py:64` (illustrative private-source reference), `my_work/phase4b/sequential_hier.py:93` (illustrative private-source reference), `my_work/tests/test_models.py:53` (illustrative private-source reference). Evidence inventory: eight report/script/output files plus manifest.json (retained in the private evidence record); approximately 125 KiB total, with no saved checkpoints or bank records.

## Verdict

The current phase-owned flat and hierarchical models pass the independent CPU correctness checks. The CLS relay defect and the padded-query NaN/gradient defect are fixed in these copies. No new model-forward correctness failure was reproduced. This is evidence about implementation behavior, not evidence of significant bank-data lift or H200 readiness.

The remaining model issues are regression protection, accurately identifying the implemented architecture, safe configuration selection, and measuring the real GPU cost. The unchanged original `fraud/sequential_model.py:94-106` still contains the old CLS mask; the corrected verdict applies only to the phase-owned paths. Existing causal checkpoints can load but produce different outputs with the new mask; they should not be silently mixed with new experiments.

## Evidence

Environment: PyTorch `2.14.1+cpu`, `<audit-python>`; CUDA unavailable. No package or source changes.

| Check | Outcome |
|---|---|
| Committed `my_work/tests/test_models.py` | 12 passed, 11 expected nested-tensor/pre-norm warnings |
| Explicit full per-sample/per-head mask references | 48/48 passed |
| Independent causal forwards and gradients | 288/288 passed |
| Actual 30-feature-token shape, normal dropout, forward/backward plus deterministic eval/embed | 16/16 passed |
| Base flat, smoke-flat and hierarchical config construction | 3/3 passed with synthetic dataset dimensions |
| Old-state-dict strict load into current A and B | Both pass; no missing/unexpected keys |
| Bidirectional logits before/after update with identical loaded weights | Exactly identical in the synthetic comparison |

The 288-case matrix covers A/B, linear/periodic embeddings, histories 0/5/15/20, depths 1/2/4, train fp32, eval with gradients, no-gradient eval with fastpath on/off, CPU bf16 train/eval. Each batch contains all-padding, target-only, partial left-padding, noncontiguous padding, and complete-history cases. The all-padding and invalid-final-slot cases are extra boundary tests, not intended dataset samples. Dropout is disabled for influence comparisons; normal configured dropout is separately exercised in the 16 shape tests.

All checked logits, encoder hidden states and relevant gradients were finite. Changing padded inputs had zero logit effect and zero input gradient. Changing the target features had zero effect on earlier real transaction hidden states at every tested depth. This verifies the causal restriction without confusing it with label leakage: the old defect violated transaction-level directionality but did not itself introduce unknown labels or post-target records into target-only prediction.

Evidence files:

- Committed test result (retained in the private evidence record)
- Independent matrix harness (retained in the private evidence record), 365-line result record (retained in the private evidence record), stderr (retained in the private evidence record)
- Checkpoint/coverage harness (retained in the private evidence record), results (retained in the private evidence record), stderr (retained in the private evidence record)

Commands:

```bash
cd <private-checkout>
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 <audit-python> -m pytest my_work/tests/test_models.py -q
REPO=<private-checkout> OMP_NUM_THREADS=1 <audit-python> <private-evidence>/models/current_model_checks.py
REPO=<private-checkout> OMP_NUM_THREADS=1 <audit-python> <private-evidence>/models/compatibility_and_test_strength.py
```

## Confirmed repairs

1. **CLS relay is removed.** A `sequential.py:79-85` and B `sequential_hier.py:103-108` assign CLS the final logical position. CLS can summarize every transaction; real transactions cannot read CLS. Four-layer perturbation tests pass, unlike the previous causal contract.
2. **Padding is numerically safe without influencing real outputs.** A `sequential.py:98-103,124-128` and B `sequential_hier.py:115-120,146-150` merge padding into a floating additive mask and open the diagonal. Padded keys remain forbidden to real queries. The reopened padded self-edge prevents empty softmax rows without making padding visible to CLS/real tokens. CPU bf16 also passes.
3. **Checkpoint tensor compatibility is preserved.** The modifications add no state-dict tensor keys. Identical loaded weights give exactly identical bidirectional outputs. The intentionally corrected causal outputs differ: synthetic max-logit change A `0.07133758`, B `0.06553766`. Re-evaluating a historical causal checkpoint under the new mask changes the evaluated predictor and requires a new provenance label.

## Residual findings and practical consequences

### M1 — Regression suite still cannot detect the repaired CLS relay (medium test gap)

`my_work/tests/test_models.py:53-57,75-77,91-118` uses one transformer layer for the relevant causal tests and checks shapes/finiteness only. The CLS relay first affects earlier representations after at least two layers.

Independent mutation test: deliberately restore the old CLS relay while retaining the safe padding guard, entirely in memory. **All three committed causal tests still pass.** The actual code is correct now, but CI would allow this specific bug back in.

Add a two/four-layer target-feature perturbation test asserting unchanged earlier real hidden states, a padding-value invariance test, and the gradient assertion across mixed history lengths. This is a targeted regression need, not a request to duplicate implementation internals in tests.

### M2 — Configuration advertises an architecture switch that factories ignore (medium configuration hazard; no shipped default failure)

`my_work/phase4a/configs/sequential.yaml:46` advertises `FTTransformerSeq | FTTransformerHier`; B's config says `type` selects the hierarchical model. In fact `phase4a/factory.py:10-11` always constructs A and `phase4b/factory.py:6-7` always constructs B. A direct reproduction sets each configuration's type to the other architecture and still obtains its phase's original model.

All committed sweep/base configurations inspected choose the corresponding phase, so this does not invalidate those defaults. A user following the apparent YAML switch can run the wrong architecture under a misleading config. Prefer explicit validation of the allowed phase-specific `model.type`, or actually dispatch by it. Until then, select the matching phase entry point/config, not just the string.

### M3 — Flat causal masking has a substantial dense allocation (operational qualification)

A `sequential.py:101-103` creates a dense fp32 `[B*heads,L,L]` mask on every forward. It remains fp32 even under bf16 mixed precision. With the project's 30 tokens per transaction and 8 heads, final mask storage alone is:

| History | Per-step batch | L | Returned fp32 mask |
|---:|---:|---:|---:|
| 5 | 4,096 | 181 | 3.999 GiB |
| 10 | 1,024 | 331 | 3.344 GiB |
| 15 | 256 | 481 | 1.765 GiB |
| 20 | 128 | 631 | 1.519 GiB |

These are formula estimates, not allocated production-size tensors or measured GPU peaks. The h5 helper also temporarily holds approximately 0.500 GiB unexpanded float mask and 0.125 GiB merged boolean mask. Attention activations, gradients and backend workspace are additional. The old PyTorch path also expanded/merged floating masks internally, so the entire cost is **not a newly introduced regression**. Default bidirectional models do not execute this helper. B's second-stage mask is much smaller (6.125 MiB at h5/batch4096; 30.25 MiB at h20/batch2048), but B still pays stage-one feature-encoder cost for every transaction.

Measure H200 peak allocated/reserved memory, examples/sec, and finite gradients for h5 causal and planned long histories before declaring the batch ladder verified. `phase4b/configs/ft_decisions_hier_h16.yaml:6-20` presents a 65,535 fused-kernel grid boundary as the definitive failure cause and reduced batch as a guaranteed scoring fix. The batch arithmetic is correct; this CPU review cannot establish the selected CUDA backend, reproduce its exception, or prove the proposed batch succeeds on that environment. Retain the device/PyTorch/backend traceback and a successful smoke run.

### M4 — Plan/model descriptions still diverge (low documentation; medium experiment interpretation)

- **No SEP token:** A `sequential.py:111-118` concatenates CLS and transaction feature blocks. The updated plan `docs/phase4_sequential_plan.md:146-180,235` still describes explicit SEP tokens. Fixed feature counts and transaction-position embeddings provide boundaries, so this is a valid variant; correct the diagram/name rather than claiming the network is broken.
- **B reuses architecture/tokenization, not a trained backbone:** `sequential_hier.py:71-86,128-150` builds a fresh feature encoder plus a fresh sequence encoder. The default is 2+4 blocks versus A's 4; this is a matched-input/recipe comparison, not a parameter-matched architecture comparison. Total attention work includes stage one `T*(F+1)^2` as well as stage two `(T+1)^2`; the simpler O(T²) statement describes stage two only.
- **Periodic result has limited scope:** both `numerical_embeddings.py:16-25` implementations use per-feature frequencies, one shared projection, no ReLU and default sigma 1. All zero-valued numerical features give identical tokens (reproduced max difference `0.0`). This is the previously documented shared-projection PL variant, not a general test of all periodic/PLR embeddings. The current linear baseline avoids this concern.
- **Published cleanup claim exceeds committed changes:** plan lines 43-46 says `phase6_sequential` naming/Phase 6 wording was cleaned, yet A config lines 1,79,85 and model docstring line 1 still contain it. Plan line 236 also lists `my_work/phase4a/train_sequential.py`; actual entry point is under `scripts/`. Fix the operational references so users can execute the intended corrected path.
- **Capacity sweep is an initial width/depth screen:** `ft_decisions_capacity.yaml:15-31` varies width/depth only. It inherits the same LR/dropout/scheduler; it does not implement the joint LR/warmup/regularization search described in plan line 337. Its dimensions all satisfy divisibility by 8 heads, but a losing large model would not establish a capacity ceiling under that design.

### M5 — Reproducibility needs code version, not just seed (information)

B's constructor now initializes its CLS parameters after constructing the encoders/head (`sequential_hier.py:74-86`). Relative to the previous constructor, the same seed produces different initial values in 41 state tensors in the synthetic comparison. This is not incorrect initialization and old checkpoints still load; it is an additional reason to record the source commit and rerun paired baselines instead of treating the unchanged seed as a controlled before/after comparison.

## Recommended next model gate

Preserve the repaired code, add the causal-invariance regression, validate phase/config agreement, correct the architecture descriptions, and run the exact production GPU batch/precision smoke tests with checkpoint and code-version provenance. Keep architecture development separate from claims about statistically reliable lift: this review cannot assess bank-data AP, uncertainty, label availability or model selection quality from synthetic tensors.
