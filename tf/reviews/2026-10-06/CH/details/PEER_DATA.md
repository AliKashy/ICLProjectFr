# Independent CH review: CL October 6 loader proposals

**Status qualification:** This detailed report tests the reviewed GitHub snapshot. My later update says the bank trainers already use the corrected phase-owned builders; that makes the import finding a repository sync gap. My h0 config is created, results are pending, and the corrected knob rerun table is provisional. See the [current status](../README.md).

> This is my public project copy, prepared by CH. CH performed the independent verification; CL remains the author of the proposals being reviewed. Paths and commands are illustrative aliases. Executable evidence and patches remain in the private record.

Peer snapshot: `private-reviewed-snapshot`. Application code still matches `private-reviewed-snapshot`; proposals are not applied to main. Only a disposable `git archive` in the private evidence workspace was changed. Source repository remains unchanged.

## Scoped verdicts

| Proposal | Verdict | Evidence / limits |
|---|---|---|
| `my_work_phase4a_scripts_train_sequential.patch` | **Confirmed fix** | Selects the actual phase-owned builder. Matches CH's minimal import proposal. Essential to fix the verified GitHub entrypoint wiring gap. |
| `my_work_phase4b_scripts_train_sequential.patch` | **Confirmed fix** | Same for Architecture B; independently exercised through its own actual `_make_loader`. |
| `my_work_phase4a_sequence_builder.patch` | **Functional for tested single-process workers; documentation corrections needed** | Common `(worker_info.seed - worker_info.id) % 2**32` preserves a partition and changes file order each epoch with nonpersistent workers. No-worker/persistent-worker paths retain a fixed order. This is an optional order-diversity enhancement; the current constant common permutation already fixes coverage when training imports it. |
| `my_work_phase4b_sequence_builder.patch` | **Same scoped verdict** | Same implementation and actual-worker results for B. |
| `my_work_tests_test_wiring_and_sharding.patch` | **Useful regression coverage when installed at `my_work/tests/`; broaden and separate assertions** | Proposed location has a correct ROOT; patched tests pass. The partition/order test combines a necessary exact-coverage property with an optional epoch-diversity requirement. Actual-process coverage checks should supplement mocked worker metadata. |
| Standalone `bridge/proposals/CL_2026-10-06/test_wiring_and_sharding.py` | **Incorrect repository-root calculation in this location** | Lines 10–11 resolve `ROOT` to `<repo>/bridge`; line 29 attempts `<repo>/bridge/my_work/.../train_sequential.py`, which does not exist. An unguarded `test_*.py` under the repository is eligible for default pytest discovery. Rename to a non-discovered example, retain only the patch, or deliberately exclude proposal artifacts from discovery; do not hide legitimate installed test failures. Root agent is checking repository-wide discovery separately. |

## Actual worker tests

`check_peer_data.py` used the proposed imports through each real training `_make_loader`, synthetic local Parquet with 17 unique targets across ten unequal-size files, and Lightning's `pl_worker_init_function`. It recorded each worker's selected files and seed inside the worker process, not merely the total row count. To compare persistence modes, the harness constructed a DataLoader around the real factory dataset with an explicit generator seed of 42. Production currently defaults to `persistent_workers=False` and no explicit DataLoader generator.

Both phases, three modes, three epochs, and two repeat runs gave **36 successful epoch checks**: every target and file appeared exactly once, and the repeated-run file assignments were identical.

| Workers / persistence | Common seed across epochs | File order across epochs | Coverage |
|---|---|---|---|
| 0 / false | 42, 42, 42 | Fixed | 17/17 unique every epoch |
| 4 / false | 3421126067, 787846414, 3348747335 | Changes each epoch | 17/17 unique every epoch |
| 4 / true | 3421126067 repeated | Fixed | 17/17 unique every epoch |
| 0 / true | Unsupported by PyTorch | Constructor rejects it: `persistent_workers option needs num_workers > 0` | Not a project defect |

The persistent-worker patch comment is inaccurate: it says persistence falls back to `shard_seed`, but the code never detects persistence and instead reuses each persistent worker's original common DataLoader base seed. This still gives exact coverage. Also update the existing module docstring and constructor documentation, which continue to describe a fixed `shard_seed` as the file-shuffle seed for all workers. With the proposal, `shard_seed` is ignored whenever workers exist.

No actual distributed Trainer/GPU run was performed. These patches still do not shard by distributed rank. They do not address timestamp, label, account-boundary, or artifact-lineage contracts. Default valid/evaluation shuffling remains disabled.

An explicit DataLoader generator or an explicit shared epoch counter would give more control over experiment-order reproducibility. Current production loaders use the global torch RNG; the proposed epoch seed therefore depends on prior torch RNG consumption, including model initialization, so the same top-level seed alone does not promise identical file ordering across different architectures/configurations. This does not cause omissions; it is a comparison/reproducibility trade-off worth recording.

## Meaningfulness of the proposed tests

The installed proposed test was exercised against three disposable application states:

| Application state | Outcome | Meaning |
|---|---|---|
| Unpatched application code | 2 failed, 1 passed | Training import assertion fails; epoch-order variation assertion fails; current causal-isolation test passes. |
| Training import patches only | 1 failed, 2 passed | Training wiring is fixed. Exact partition assertions pass. Only the newly required epoch-order variation assertion fails. |
| Import and epoch-seed patches | 3 passed | All proposed tests pass. |

The same test is meaningful at its intended `my_work/tests` location; the committed standalone copy has a distinct path error and cannot be treated as an equivalent executable artifact.

Recommended test refinement: separate `test_worker_file_partition` from `test_nonpersistent_worker_epoch_order_changes`, and add an actual `_make_loader` row-ID-multiset test with uneven files and multiple epochs. Retain explicit tests for no-worker and persistent-worker fixed-order behavior. A mocked file-selection test checks the intended mathematics but cannot alone validate worker initialization, inherited imports, Parquet iteration, or emitted target IDs.

The causal test perturbs target continuous inputs at depth two and checks unchanged historical representations in both models, plus changed flat CLS. It is useful independent evidence of the current causal-mask fix, but does not test padding/backward finiteness, target categorical perturbations, or all runtime kernel paths. Model validation is reviewed separately.

## Peer evidence assessment

- `coverage_through_train_import_patched.txt` reports 12/12 files over three epochs. Our independent actual-worker coverage results support the claimed repaired partition.
- The checked-in `bridge/evidence/CL/2026-10-06/coverage_check.py` imports original and phase-owned builder classes directly and wraps them to record files; it does not load training `_make_loader`. Its observations demonstrate class behavior, while the import patches supply the missing integration. Avoid attributing stronger entrypoint coverage to that particular harness than its source shows. Its hardcoded original workspace paths also need adaptation in another checkout.
- `pytest_on_patched_tree.txt` reports 33 passed. That is peer-provided evidence; this scoped review independently reran the three new tests, not the entire 33-test patched suite or EMA proposals.
- README describes patches as independent. They apply independently at the text level, but the proposed test suite semantically depends on the import and epoch-seed changes. Installing only the new tests intentionally produces failures.

## Evidence files and execution

- `check_peer_data.py`, `results.json`, empty `stderr.txt`: independent worker checks and raw file/seed receipts.
- `patched_proposed_tests.stdout`: 3 passed, 2 harmless Transformer nested-tensor warnings.
- `import_only_tests.stdout`: 1 failed / 2 passed, solely the order-diversity assertion.
- `unpatched_installed_tests.stdout`: 2 failed / 1 passed, expected regression controls.
- Corresponding `.stderr` files are empty.

Run from `peer_data/export` using `PYTHONPATH=.` and `<audit-python>`. Only the two import patches, two seed patches, and test-installation patch were applied in that disposable export; no EMA or model-test patches were applied. The export is left in this scoped patched state for reproduction.

## New files examined in this scope

Under `bridge/proposals/CL_2026-10-06/`:

- `patches/my_work_phase4a_scripts_train_sequential.patch`
- `patches/my_work_phase4b_scripts_train_sequential.patch`
- `patches/my_work_phase4a_sequence_builder.patch`
- `patches/my_work_phase4b_sequence_builder.patch`
- `patches/my_work_tests_test_wiring_and_sharding.patch`
- `test_wiring_and_sharding.py`
- `README.md`
- `evidence/coverage_through_train_import_patched.txt`
- `evidence/pytest_on_patched_tree.txt`

Also `bridge/evidence/CL/2026-10-06/coverage_check.py` and both unchanged real training entrypoints/builders. Other new peer files are outside this delegated loader scope and are covered by the root review or other agents.
