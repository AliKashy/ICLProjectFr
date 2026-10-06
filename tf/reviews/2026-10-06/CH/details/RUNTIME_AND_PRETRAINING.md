# CH runtime, packaging and Phase 5 review — 2026-10-06

> Public first-person project copy prepared by CH on 2026-10-06. I keep my original findings and guides unchanged. CH performed the checks reported here. Source paths and commands are illustrative private-project aliases; code, evidence programs and the candidate patch remain in the private review record. Exact approved scientific details are retained.

Reviewed immutable source `October-6-reviewed-source` at `<private-checkout>`. Source was not edited. All new harnesses, build copies and outputs are in the private evidence directory. No bank data, S3 access, bank checkpoint or GPU was used.

## Verdict

The Phase 5 pipeline has usable mechanics: both recipes can pretrain, fine-tune, save, reload and write correct row-aligned predictions on a small CPU fixture; SSL embedding export also works. **The two pretraining objectives remain incorrect in the active source, so successful execution does not establish scientifically valid pretraining.** Package installation and the reconstructed original baseline/streaming entrypoints still have reproducible blockers.

No code changes occurred since the prior `previous-reviewed-source` snapshot in `pyproject.toml`, `requirements.txt`, `fraud/`, `src/`, `my_work/phase5_pretrain/`, or `my_work/scoring/`. The only change in the broader scoped diff including `my_work/tests/` is `test_models.py` (48 additions, one deletion), which belongs to the Phase 4 work. Therefore the defects below are **reconfirmed outstanding issues**, not newly introduced Phase 4 regressions. Candidate patches living elsewhere in the repository are not active repairs.

## Findings that block their respective workloads

| ID | Priority and workload | Current evidence | Source |
|---|---|---|---|
| RT-01 | High before any SupCon result is trusted | Features flatten sample-major while labels repeat view-major. Correctly paired class views receive loss **14.2857151**, while incorrect cross-class pairing receives **0.0000011921**; an independent formula gives approximately **0.0000009537** and **14.2857151**, respectively. Gradient existence is not an objective-correctness test. | `my_work/phase5_pretrain/supcon.py:37-42`; `my_work/tests/test_supcon.py:31-44` |
| RT-02 | High before masked SSL pretraining | Whole tokens are replaced with the same mask token, with no remaining numeric field identity and one shared numeric decoder. Simultaneously hidden targets +2 and -2 both predict **0.5007388** in the probe. Dropout noise cannot supply stable field identity. | `my_work/phase5_pretrain/ssl_pretrain.py:50-53,65-78` |
| RT-03 | High for a reproducible installed project | A fresh isolated wheel build fails with `BackendUnavailable: Cannot import 'setuptools.backends._legacy'`. A diagnostic throwaway copy changing only the backend to `setuptools.build_meta` builds a wheel containing **zero `my_work/` files and zero YAML/CSV configuration assets**. Fixing only the backend is insufficient. | `pyproject.toml:3,29-31` |
| RT-04 | High before reproducing original streaming baseline | `build_inferred_config([3], 1, 1)` raises `TypeError: InferredConfig.__init__() got an unexpected keyword argument 'output_cardinality'`. Both original streaming train and score use this builder. The vendored package identifies itself as 1.1.0 and lacks that field. Align the recovered code with the intended library API, while preserving the distinction between reconstruction compatibility and historical bank execution. | `fraud/streaming_model.py:27-37,90`; `src/pytorch_tabular/config/config.py:206-230`; `src/pytorch_tabular/__init__.py:5` |
| RT-05 | High before baseline train/save/load reproducibility on modern PyTorch | Original baseline trains a 32-row CPU fixture for four optimizer steps, then its default best-checkpoint load fails because the vendored helper calls `torch.load` without an explicit serialization policy. Modern PyTorch's default restricted loader rejects checkpoint `OmegaConf.DictConfig` objects. The helper is used by baseline best-model reload and standalone model load. | `src/pytorch_tabular/utils/python_utils.py:74-84`; `src/pytorch_tabular/tabular_model.py:389-398,664-665,1485-1494` |
| RT-06 | Medium, before class weighting is enabled in affected paths | A fine-tuner trained with non-default `pos_weight` has a `loss_fn.weight` state buffer. Both Phase 5 scorers reconstruct with default weight 1, then load strictly: `Unexpected key(s) in state_dict: "loss_fn.weight"`. Original sequential scoring has the same issue; original streaming scoring also omits its training weight. The default Phase 5 weight 1 avoids the issue. | `my_work/phase5_pretrain/ssl_finetune.py:50-53`; `scripts/score_ssl_finetune.py:60-69`; `scripts/score_contrastive.py:61-70`; `fraud/scripts/score_streaming.py:70-79` |
| RT-07 | Medium, before claiming a run used pretraining | Missing checkpoint paths merely warn and train from scratch; an empty state is accepted with `strict=False` and **18 missing keys** in the tiny probe. The output naming can still imply SSL or SupCon fine-tuning. Legitimate complete encoder transfers do match tensor-for-tensor, so the issue is failure detection and run identity. | `my_work/phase5_pretrain/ssl_finetune.py:55-59`; `scripts/train_ssl_finetune.py:91-98`; `scripts/train_contrastive.py:92-99` |
| RT-08 | Medium, experiment evidence preservation | Baseline training writes `training_metrics.csv` inside the same directory passed to `save_model`; the vendored save method deletes every existing file in that directory. Executing the actual cleanup method against a harmless temporary fixture deletes the CSV. | `fraud/scripts/train_baseline.py:208-211,233`; `src/pytorch_tabular/tabular_model.py:1532-1535` |

`scripts/...` in the Phase 5 rows abbreviates `my_work/phase5_pretrain/scripts/...`.

## Further limitations and checks

- **Dependencies:** `requests` is imported eagerly by `src/pytorch_tabular/utils/data_utils.py:8`, and `pytorch-tabnet` by `src/pytorch_tabular/models/tabnet/tabnet_model.py:10-11`. Neither is declared in `requirements.txt` or project metadata. The audit environment already contains those supplements. Its import/test success is not evidence that the declared dependencies create a complete clean environment. Declare required direct imports or make unrelated model imports optional, then test the installed wheel outside the checkout. PyYAML is in requirements and available transitively through existing packages, so it is not presented here as a separately demonstrated clean-install blocker.
- **Actual unlabeled input is unsupported by the SSL loader:** the objective ignores the target, but `ParquetStreamingDataset` adds the target to read columns and unconditionally encodes it (`fraud/data/parquet_loader.py:113-115,149-150`). A synthetic frame without `frd_tag` produces `KeyError('frd_tag')`. This does not stop the current labeled-bank-data recipe; it limits the claim that the implementation can directly exploit a separate unlabeled pool. Add an explicit target-optional data contract before using such a pool.
- **Zero masked tokens can produce a non-differentiable loss:** the initialization at `ssl_pretrain.py:72` is a detached scalar; with no masks, `.backward()` fails. Confirmed for `mask_ratio=0`. Random empty-mask batches are improbable at the default large batch, but become possible with small partial batches. Validate the ratio and deliberately handle empty masks or guarantee at least one masked token.
- **Frozen-probe representations are stochastic during training:** `freeze_encoder=True` uses `no_grad` but does not put the encoder into eval mode (`ssl_finetune.py:61-67`). Repeated identical input gave logit difference 0.2442235 with the default dropout. This does freeze learned parameters; decide whether dropout is intentional probe regularization or whether the intended baseline requires a fixed representation, and document/test the choice.
- **Accumulation settings differ across recipes:** SSL pretraining forwards `accumulate_grad_batches` (`pretrain_ssl.py:107`), while SupCon pretraining and both fine-tuning scripts do not (`pretrain_contrastive.py:102-106`, `train_ssl_finetune.py:110-114`, `train_contrastive.py:111-115`). Defaults currently use one; overrides should not silently change only one recipe's effective batch size.
- **Original sequential causal mode remains imperfect:** the unchanged original model allows an information path through its global CLS token across layers. The earlier probe is reconfirmed: first-layer past-token change 0, second-layer change 0.0248523 when later input tokens change. This is a causal-ablation violation; it does not establish leakage from transactions later than the target. The Phase 4 implementation has its own separate review.
- **Original baseline attention importances remain misindexed:** the unchanged importance extraction selects the wrong attention axis; the synthetic observed values `[0.1, 0.3]` differ from the intended CLS-query values `[0.55, 0.4]`. Do not use these importance outputs as validated explanations (`src/pytorch_tabular/models/ft_transformer/ft_transformer.py:85`).
- **Default SupCon memory requires real-device measurement:** batch 8192 with two views creates 16384×16384 pairwise matrices; each float32 matrix alone is 1 GiB. Multiple masks/logits/probability tensors and encoder activations coexist. This is an arithmetic sizing warning, not a measured out-of-memory claim. Reduce or chunk batches only with a stated contrastive objective/effective-batch design; ordinary gradient accumulation does not reproduce a larger set of contrastive negatives.
- **Downstream capability is narrower than comments imply:** the SSL exporter works for SSL checkpoints; it reconstructs `MaskedTabularAutoencoder` and strictly loads its full state (`export_embeddings.py:60-69`), so it is not a generic SupCon exporter. Its referenced `train_gbm_on_embeddings.py` does not exist. Phase 5 does not directly initialize either history model. Phase 5 scoring writes IDs, binary target and score, but no amount; dollar-based reporting needs correctly joined amount metadata. Metric/ensemble semantics are reviewed separately.

## What is verified to work

`phase5_local_roundtrip.py` exercises the actual unchanged `main()` bodies for SSL and SupCon pretraining, fine-tuning and scoring. The only external-service replacement is the S3 filesystem factory returning a local filesystem. It uses small local predictor/artifact files, one 32-row synthetic Parquet file per split, zero workers, 32-bit CPU execution, width 8, one encoder block and one epoch. This deliberately does not test multi-worker sharding, default batch memory, distributed training or bank integration.

| Tested property | Result |
|---|---|
| SSL pretrain → fine-tune → checkpoint → score | Completes; all 32 evaluation IDs preserved; finite scores; output equals directly restored model with maximum absolute difference 0 |
| SupCon pretrain → fine-tune → checkpoint → score | Completes; all 32 evaluation IDs preserved; finite scores; output equals directly restored model with maximum absolute difference 0 |
| SSL checkpoint → embedding export | 32×8 finite embeddings; exact equality with directly restored encoder |
| Complete SSL encoder transfer to fine-tuner | Every encoder tensor matches exactly; only expected classifier-head keys absent |
| Complete SupCon encoder transfer to fine-tuner | Every encoder tensor matches exactly; only expected classifier-head keys absent |
| Original sequential padding invariance | Arbitrarily changing padded values does not change target logits; maximum difference 0 |

The tiny synthetic class balance and metrics are not realistic fraud-performance estimates. Execution success cannot repair RT-01/RT-02 or establish bank performance.

## Repair and acceptance order

1. For clean reproduction, correct packaging, explicit dependencies and configuration inclusion; align the original streaming library API and supported checkpoint serialization. Build/install a wheel in a clean environment and run imports and baseline train/save/load/score outside the source directory. Preserve logs through model saving.
2. Before Phase 5 compute, correct SupCon view/label alignment and retain numeric field identity under masking. Add independent-reference SupCon tests, joint sample-permutation invariance and a masked-field-distinction test. Verify finite gradients as an additional check, not the whole test.
3. Make pretraining mode explicit: scratch versus required checkpoint; reject missing or incomplete encoder states. Preserve source checkpoint identity, feature order and artifact fingerprints in run manifests. Rebuild scoring models from verified checkpoint hyperparameters and exercise a non-default class weight.
4. Then rerun the local round trip and a small bank-side canary with exact production dependency versions, GPU, precision, file/row counts and label-maturity assumptions. Benchmark the configured SupCon batch before a full run. No such bank-side verification occurred in this review.
5. Compare scratch, corrected SSL and corrected SupCon with the same architecture, data cohorts, preprocessing and evaluation protocol. Add history-model transfer or GBM-on-embeddings only as separate implemented experiments; do not label today's exporter as a complete downstream model pipeline.

## Evidence and reproducibility

Environment: Python 3.12.14; PyTorch 2.14.1+cpu; pytorch-lightning 2.6.6; torchmetrics 1.8.2; NumPy 2.5.3; pandas 2.3.3; scikit-learn 1.9.1; PyArrow 25.0.1; OmegaConf 2.3.1. Full versions in `environment.json`.

Run from the pinned source root with `PYTHONPATH=.:src` and `<audit-python>`:

```text
<private-evidence>/runtime_pretrain/phase5_repro.py
<private-evidence>/runtime_pretrain/phase5_local_roundtrip.py
CH/checks/original_integration_repros.py
CH/checks/original_baseline_smoke.py
CH/checks/original_project_repros.py
```

The first harness is a copy of the frozen earlier probe so its JSON output is written to scratch rather than the original audit folder. The original three probes create only temporary files or stdout. Results are stored in `phase5_repro_results.json`, `phase5_local_roundtrip_results.json` and the corresponding `.log` files.

Packaging was tested using a new standard-library venv in `build_venv`, with `pip wheel --no-deps --wheel-dir <scratch>/wheels <scratch>/build_source`. `build_source` initially copied only the current pyproject, requirements and the three source trees. `wheel_build.log` is the unchanged-project failure. The subsequent **diagnostic copy only** changes the backend string; `backend_only_build.log` and `backend_only_wheel_inspection.json` document the resulting incomplete wheel. The shared audit environment was not changed.
