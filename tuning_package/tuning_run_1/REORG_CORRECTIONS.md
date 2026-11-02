# Four corrections to REORGANIZATION.md (reviewer, 2 November)

Apply them now, including to any step already done.

1. **Receipts.** In `comparisons/chunk_vs_sliding_inputs/`, set `receipt_path` to `models/<name>/seed<NN>/scores/<population>.receipt.json`, and copy each referenced scorer receipt there unchanged. `compare_models.py` opens the receipt and checks its hashes. These receipts are the one exception to "no receipts in the tree".
2. **Scale.** Move `waqas/v2/scale_runs.py` and `waqas/v2/data_size_specs.py` to `archive/code/`, together with the three `run_ablations.py` runners they depend on. Drop their import-map rows. `fraud/experiments/scale/README.md` says: train on 100 files with the chosen model's trainer, its base config and `data.train_end=100`.
3. **Store.** Only the 14-day account consumer goes to the archive. Any embedding-store code (a vector store with snapshots and earlier-time lookup) goes unchanged to `fraud/experiments/embedding/store/`, marked built, not run. Say in `REORG_REPORT.md` whether you found any.
4. **Dependencies.** Keep `xgboost` exactly as the delivered dependency file names it. Add `[project.optional-dependencies] tuning = ["optuna"]` and the line `pip install -e ".[tuning]"` under the tuning command in `README.md`.
