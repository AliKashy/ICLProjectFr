# For Copilot: tuning run 1 on the pod

The reviewers wrote this run sheet (`README.md`) and the four scripts from records made before the owner reorganized the pod. Treat them as the specification, not as exact paths, and adapt them to the pod as it is now.

1. **Find the three recorded runs the screen starts from.** Each run used learning rate 3e-4 and has a `training_metrics.csv`. `locate.py` finds them by their settings, or find them yourself:
   - the chosen model `sequential_hier_allpos_h20_chunk`, seed 43: V2 trainer, `resolved_train_v2.yaml`, all-position, non-overlapping chunks, history 20;
   - `sequential_hier_h20`, seed 43: V2 trainer, `resolved_train_v2.yaml`, one target per window, history 20;
   - `sequential_h15`, seed 42: V1 flat trainer, `resolved_train.yaml`, history 15.
2. **Confirm the two trainers' current paths and how they are invoked:**
   - `python -m waqas.v2.scripts.train_v2 --config <yaml> [overrides] [--resume]`;
   - `python waqas/phase4a/scripts/train_sequential.py --config <yaml> [overrides]`.

   If the code has moved, update the commands, not the trainers.
3. **Check every local file the three configs read** (the predictor list, `test.yaml`, any measured-width file). If one has moved, change the path in a copy of the config only, and list each change.
4. **Keep the runs exactly as specified:**
   - learning rates 1e-4 and 1e-3 for each model, with `training.max_epochs=5`, and everything else as in the recorded config;
   - one folder per run, `runs/tuning/<model>_lr<rate>`;
   - one run after another, in the background (`nohup`);
   - never overwrite an existing run folder or log.
5. **Do not score any 5-epoch checkpoint.** The V2 trainer refuses to resume a run that has been scored.
6. **Keep the decision rule and the morning report** as `README.md` gives them.
7. **Write the final commands into `handoff/TUNING_RUN_1.md`.** Give the owner the exact lines to paste on the pod, and list every path you changed.
